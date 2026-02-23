"""Persistent ledger for live strategy positions and order lifecycle state."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from bot.file_security import (
    atomic_write_private,
    tighten_file_permissions,
    validate_sensitive_file,
)
from bot.number_utils import safe_float, safe_int

logger = logging.getLogger(__name__)

LIVE_TRADES_FILE = "live_trades.json"


class LiveTradeLedger:
    """Track live strategy positions and pending broker orders on disk."""

    def __init__(self, state_file: str = LIVE_TRADES_FILE):
        self.state_file = Path(state_file)
        self.positions: list[dict] = []
        self._load_state()

    # ── Persistence ────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load live strategy state from disk."""
        if not self.state_file.exists():
            return

        try:
            validate_sensitive_file(
                self.state_file,
                label="live trade ledger file",
                allow_missing=False,
            )
            tighten_file_permissions(self.state_file, label="live trade ledger file")
            with open(self.state_file, encoding="utf-8") as f:
                data = json.load(f)
            positions = data.get("positions", [])
            if isinstance(positions, list):
                self.positions = [p for p in positions if isinstance(p, dict)]
            logger.info(
                "Loaded live trade ledger: %d tracked positions",
                len(self.positions),
            )
        except RuntimeError:
            raise
        except (json.JSONDecodeError, FileNotFoundError, OSError, KeyError) as exc:
            logger.warning("Failed to load live trade ledger: %s. Starting empty.", exc)
            self.positions = []

    def _save_state(self) -> None:
        """Persist live strategy state to disk."""
        payload = {
            "positions": self.positions,
            "last_updated": datetime.now().isoformat(),
        }
        atomic_write_private(
            self.state_file,
            json.dumps(payload, indent=2, default=str),
            label="live trade ledger file",
        )

    # ── Position Lifecycle ─────────────────────────────────────────

    def register_entry_order(
        self,
        *,
        strategy: str,
        symbol: str,
        quantity: int,
        max_loss: float,
        entry_credit: float,
        details: dict,
        entry_order_id: str,
        entry_order_status: str = "PLACED",
        opened_at: Optional[str] = None,
    ) -> str:
        """Register a newly submitted live entry order."""
        position_id = f"live_{str(uuid.uuid4())[:8]}"
        now_iso = datetime.now().isoformat()

        record = {
            "position_id": position_id,
            "strategy": strategy,
            "symbol": symbol,
            "quantity": max(1, safe_int(quantity, 1)),
            "max_loss": max(0.0, safe_float(max_loss, 0.0)),
            "entry_credit": max(0.0, safe_float(entry_credit, 0.0)),
            "current_value": max(0.0, safe_float(entry_credit, 0.0)),
            "details": details or {},
            "expiration": str((details or {}).get("expiration", "")),
            "status": "open" if not entry_order_id else "opening",
            "entry_order_id": entry_order_id,
            "entry_order_status": str(entry_order_status or "PLACED").upper(),
            "entry_order_time": now_iso,
            "open_date": opened_at or "",
            "exit_order_id": "",
            "exit_order_status": "",
            "exit_reason": "",
            "close_date": "",
            "close_value": 0.0,
            "realized_pnl": 0.0,
            "last_reconciled": now_iso,
        }

        self.positions.append(record)
        self._save_state()
        return position_id

    def register_exit_order(
        self,
        *,
        position_id: str,
        exit_order_id: str,
        reason: str,
    ) -> bool:
        """Attach an exit order to an open position."""
        position = self.get_position(position_id)
        if not position:
            return False

        position["status"] = "closing"
        position["exit_order_id"] = exit_order_id
        position["exit_order_status"] = "PLACED"
        position["exit_reason"] = reason
        position["last_reconciled"] = datetime.now().isoformat()
        self._save_state()
        return True

    def get_position(self, position_id: str) -> Optional[dict]:
        """Return a tracked position by ID."""
        for position in self.positions:
            if position.get("position_id") == position_id:
                return position
        return None

    def list_positions(self, statuses: Optional[set[str]] = None) -> list[dict]:
        """List tracked positions, optionally filtered by status."""
        if not statuses:
            return [dict(p) for p in self.positions]
        normalized = {s.lower() for s in statuses}
        return [
            dict(p)
            for p in self.positions
            if str(p.get("status", "")).lower() in normalized
        ]

    def update_position_quote(
        self,
        position_id: str,
        *,
        current_value: Optional[float] = None,
        dte_remaining: Optional[int] = None,
    ) -> None:
        """Update mark and DTE for a tracked position."""
        position = self.get_position(position_id)
        if not position:
            return

        if current_value is not None:
            position["current_value"] = round(max(0.0, safe_float(current_value, 0.0)), 2)
        if dte_remaining is not None:
            position["dte_remaining"] = safe_int(dte_remaining, 0)
        position["last_reconciled"] = datetime.now().isoformat()
        self._save_state()

    # ── Reconciliation ──────────────────────────────────────────────

    def pending_entry_order_ids(self) -> list[str]:
        """Return order IDs for entries awaiting terminal status."""
        pending = []
        for position in self.positions:
            if position.get("status") != "opening":
                continue
            order_id = str(position.get("entry_order_id", "")).strip()
            if order_id:
                pending.append(order_id)
        return pending

    def pending_exit_order_ids(self) -> list[str]:
        """Return order IDs for exits awaiting terminal status."""
        pending = []
        for position in self.positions:
            if position.get("status") != "closing":
                continue
            order_id = str(position.get("exit_order_id", "")).strip()
            if order_id:
                pending.append(order_id)
        return pending

    def reconcile_entry_order(
        self,
        order_id: str,
        *,
        status: str,
        filled_at: Optional[str] = None,
        entry_credit: Optional[float] = None,
        filled_quantity: Optional[int] = None,
    ) -> bool:
        """Update a pending entry order status."""
        normalized = str(status or "").upper()
        changed = False

        for position in self.positions:
            if str(position.get("entry_order_id", "")) != order_id:
                continue
            if position.get("status") != "opening":
                continue

            position["entry_order_status"] = normalized
            position["last_reconciled"] = datetime.now().isoformat()

            if normalized == "FILLED":
                position["status"] = "open"
                position["open_date"] = filled_at or datetime.now().isoformat()
                if entry_credit is not None:
                    position["entry_credit"] = round(max(0.0, safe_float(entry_credit, 0.0)), 2)
                if filled_quantity is not None:
                    position["quantity"] = max(1, safe_int(filled_quantity, 1))
            elif normalized in {"CANCELED", "REJECTED", "EXPIRED"}:
                position["status"] = normalized.lower()
                position["close_date"] = datetime.now().isoformat()

            changed = True

        if changed:
            self._save_state()
        return changed

    def reconcile_exit_order(
        self,
        order_id: str,
        *,
        status: str,
        filled_at: Optional[str] = None,
        close_value: Optional[float] = None,
    ) -> bool:
        """Update a pending exit order status."""
        normalized = str(status or "").upper()
        changed = False

        for position in self.positions:
            if str(position.get("exit_order_id", "")) != order_id:
                continue
            if position.get("status") != "closing":
                continue

            position["exit_order_status"] = normalized
            position["last_reconciled"] = datetime.now().isoformat()

            if normalized == "FILLED":
                quantity = max(1, safe_int(position.get("quantity", 1), 1))
                entry_credit = safe_float(position.get("entry_credit", 0.0), 0.0)
                debit = safe_float(close_value, safe_float(position.get("current_value", 0.0), 0.0))
                position["close_value"] = round(max(0.0, debit), 2)
                position["realized_pnl"] = round((entry_credit - debit) * quantity * 100, 2)
                position["status"] = "closed"
                position["close_date"] = filled_at or datetime.now().isoformat()
            elif normalized in {"CANCELED", "REJECTED", "EXPIRED"}:
                position["status"] = "open"
                position["exit_order_id"] = ""
                position["exit_order_status"] = normalized
                position["exit_reason"] = ""

            changed = True

        if changed:
            self._save_state()
        return changed

    def close_missing_from_broker(
        self,
        *,
        open_strategy_symbols: set[str],
        position_symbol_resolver,
    ) -> int:
        """Mark tracked positions closed if none of their option legs remain."""
        changed = 0
        now_iso = datetime.now().isoformat()

        for position in self.positions:
            status = str(position.get("status", "")).lower()
            if status not in {"open", "closing"}:
                continue

            symbols = position_symbol_resolver(position)
            if not symbols:
                continue
            if symbols & open_strategy_symbols:
                continue

            position["status"] = "closed_external"
            position["close_date"] = now_iso
            position["exit_order_status"] = position.get("exit_order_status", "EXTERNAL")
            position["last_reconciled"] = now_iso
            changed += 1

        if changed:
            self._save_state()
        return changed

    # ── Reporting ────────────────────────────────────────────────────

    def summary(self, *, today_iso: Optional[str] = None) -> dict:
        """Return aggregate ledger metrics."""
        today = today_iso or datetime.now().date().isoformat()
        out = {
            "total_tracked": len(self.positions),
            "opening": 0,
            "open": 0,
            "closing": 0,
            "closed": 0,
            "closed_today": 0,
            "realized_pnl_today": 0.0,
        }

        for position in self.positions:
            status = str(position.get("status", "")).lower()
            if status in out:
                out[status] += 1

            close_date = str(position.get("close_date", ""))
            if close_date.startswith(today):
                out["closed_today"] += 1
                out["realized_pnl_today"] += safe_float(position.get("realized_pnl", 0.0), 0.0)

        out["realized_pnl_today"] = round(out["realized_pnl_today"], 2)
        return out
