"""Paper trading simulator — simulates trades without real money."""

import json
import logging
import uuid
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from bot.file_security import (
    atomic_write_private,
    tighten_file_permissions,
    validate_sensitive_file,
)

logger = logging.getLogger(__name__)

PAPER_TRADES_FILE = "paper_trades.json"


class PaperTrader:
    """Simulates order execution and position tracking for paper trading.

    Maintains state in a local JSON file so positions persist across restarts.
    """

    def __init__(self, initial_balance: float = 100_000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions: list[dict] = []
        self.closed_trades: list[dict] = []
        self.orders: list[dict] = []
        self._load_state()

    # ── State Persistence ────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load paper trading state from disk."""
        path = Path(PAPER_TRADES_FILE)
        if path.exists():
            try:
                validate_sensitive_file(
                    path,
                    label="paper trading state file",
                    allow_missing=False,
                )
                tighten_file_permissions(path, label="paper trading state file")
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.balance = data.get("balance", self.initial_balance)
                self.positions = data.get("positions", [])
                self.closed_trades = data.get("closed_trades", [])
                self.orders = data.get("orders", [])
                logger.info(
                    "Loaded paper trading state: balance=$%.2f, %d open positions",
                    self.balance, len(self.positions),
                )
            except RuntimeError:
                raise
            except (json.JSONDecodeError, KeyError, FileNotFoundError, OSError) as e:
                logger.warning("Failed to load paper state: %s. Starting fresh.", e)

    def _save_state(self) -> None:
        """Save paper trading state to disk."""
        data = {
            "balance": self.balance,
            "positions": self.positions,
            "closed_trades": self.closed_trades,
            "orders": self.orders,
            "last_updated": datetime.now().isoformat(),
        }
        path = Path(PAPER_TRADES_FILE)
        payload = json.dumps(data, indent=2, default=str)
        atomic_write_private(path, payload, label="paper trading state file")

    # ── Account Info ─────────────────────────────────────────────────

    def get_account_balance(self) -> float:
        return self.balance

    def get_positions(self) -> list:
        return self.positions

    def get_open_position_count(self) -> int:
        return len(self.positions)

    def get_daily_pnl(self) -> float:
        """Sum of P&L from trades closed today."""
        today = date.today().isoformat()
        daily = sum(
            t.get("pnl", 0) for t in self.closed_trades
            if t.get("close_date", "").startswith(today)
        )
        return daily

    # ── Order Execution ──────────────────────────────────────────────

    def execute_open(
        self,
        strategy: str,
        symbol: str,
        credit: float,
        max_loss: float,
        quantity: int,
        details: dict,
    ) -> dict:
        """Simulate opening a new position."""
        order_id = str(uuid.uuid4())[:8]
        position_id = f"paper_{order_id}"

        position = {
            "position_id": position_id,
            "order_id": order_id,
            "strategy": strategy,
            "symbol": symbol,
            "entry_credit": credit,
            "current_value": credit,  # Starts at the credit received
            "max_loss": max_loss,
            "quantity": quantity,
            "open_date": datetime.now().isoformat(),
            "expiration": details.get("expiration", ""),
            "dte_remaining": details.get("dte", 0),
            "details": details,
            "status": "open",
        }

        self.positions.append(position)

        # Credit is received when opening
        total_credit = credit * quantity * 100
        self.balance += total_credit

        order = {
            "order_id": order_id,
            "type": "open",
            "strategy": strategy,
            "symbol": symbol,
            "credit": credit,
            "quantity": quantity,
            "timestamp": datetime.now().isoformat(),
            "status": "FILLED",
        }
        self.orders.append(order)

        self._save_state()

        logger.info(
            "PAPER OPEN: %s %s on %s | %d contracts | Credit: $%.2f ($%.2f total)",
            strategy, position_id, symbol, quantity, credit, total_credit,
        )

        return {"order_id": order_id, "position_id": position_id, "status": "FILLED"}

    def execute_close(
        self,
        position_id: str,
        close_value: Optional[float] = None,
        reason: str = "",
        quantity: Optional[int] = None,
    ) -> dict:
        """Simulate closing a position."""
        position = None
        for p in self.positions:
            if p["position_id"] == position_id:
                position = p
                break

        if position is None:
            logger.warning("Position %s not found for close.", position_id)
            return {"status": "NOT_FOUND"}

        # If no close value provided, simulate at current value
        if close_value is None:
            close_value = position.get("current_value", 0)

        entry_credit = position["entry_credit"]
        open_quantity = max(1, int(position.get("quantity", 1)))
        close_quantity = max(1, min(open_quantity, int(quantity or open_quantity)))
        pnl_per_contract = entry_credit - close_value
        total_pnl = pnl_per_contract * close_quantity * 100

        # Debit to close
        total_debit = close_value * close_quantity * 100
        self.balance -= total_debit

        # Record the closed trade
        closed = {
            **position,
            "close_date": datetime.now().isoformat(),
            "close_value": close_value,
            "pnl_per_contract": pnl_per_contract,
            "pnl": total_pnl,
            "reason": reason,
            "status": "closed" if close_quantity >= open_quantity else "partial_closed",
            "closed_quantity": close_quantity,
        }
        self.closed_trades.append(closed)

        if close_quantity >= open_quantity:
            # Remove from open positions
            self.positions = [p for p in self.positions if p["position_id"] != position_id]
            remaining_quantity = 0
        else:
            remaining_quantity = open_quantity - close_quantity
            position["quantity"] = remaining_quantity
            position["partial_closed"] = True

        order_id = str(uuid.uuid4())[:8]
        self.orders.append({
            "order_id": order_id,
            "type": "close",
            "position_id": position_id,
            "symbol": position["symbol"],
            "debit": close_value,
            "quantity": close_quantity,
            "pnl": total_pnl,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "status": "FILLED",
        })

        self._save_state()

        logger.info(
            "PAPER CLOSE: %s on %s | P/L: $%.2f | Reason: %s",
            position_id, position["symbol"], total_pnl, reason,
        )

        return {
            "order_id": order_id,
            "position_id": position_id,
            "pnl": total_pnl,
            "status": "FILLED",
            "closed_quantity": close_quantity,
            "remaining_quantity": remaining_quantity,
        }

    def update_position_values(
        self,
        market_prices: dict,
        position_meta: Optional[dict] = None,
    ) -> None:
        """Update current values of open positions based on market data.

        market_prices: dict mapping position_id -> current mid price of the spread.
        """
        for position in self.positions:
            pid = position["position_id"]
            if pid in market_prices:
                position["current_value"] = market_prices[pid]
            if position_meta and pid in position_meta:
                meta = position_meta.get(pid, {})
                if isinstance(meta, dict):
                    position.update(meta)

            # Update DTE
            exp = position.get("expiration", "")
            if exp:
                try:
                    exp_date = datetime.strptime(exp.split("T")[0], "%Y-%m-%d").date()
                    position["dte_remaining"] = (exp_date - date.today()).days
                except ValueError:
                    pass

        self._save_state()

    # ── Reporting ────────────────────────────────────────────────────

    def get_performance_summary(self) -> dict:
        """Get overall performance stats."""
        total_trades = len(self.closed_trades)
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
                "balance": self.balance,
                "open_positions": len(self.positions),
            }

        wins = sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in self.closed_trades)
        avg_pnl = total_pnl / total_trades

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": round(wins / total_trades * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "balance": round(self.balance, 2),
            "return_pct": round(
                (self.balance - self.initial_balance) / self.initial_balance * 100, 2
            ),
            "open_positions": len(self.positions),
        }
