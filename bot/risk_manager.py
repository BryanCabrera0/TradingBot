"""Risk management system — position sizing, portfolio limits, daily loss tracking."""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from bot.config import RiskConfig
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk calculations."""
    account_balance: float = 0.0
    open_positions: list = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_date: Optional[date] = None
    total_risk_deployed: float = 0.0  # Total max loss across all positions


class RiskManager:
    """Enforces risk limits before any trade is executed."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.portfolio = PortfolioState()

    def update_portfolio(
        self,
        account_balance: float,
        open_positions: list,
        daily_pnl: float = 0.0,
    ) -> None:
        """Update the portfolio state with latest data."""
        self.portfolio.account_balance = account_balance
        self.portfolio.open_positions = open_positions

        today = date.today()
        if self.portfolio.daily_pnl_date != today:
            self.portfolio.daily_pnl_date = today
        self.portfolio.daily_pnl = daily_pnl

        # Calculate total risk deployed
        total_risk = 0.0
        for pos in open_positions:
            max_loss = pos.get("max_loss", 0)
            quantity = pos.get("quantity", 1)
            total_risk += max_loss * quantity * 100  # per contract
        self.portfolio.total_risk_deployed = total_risk

    def can_open_more_positions(self) -> bool:
        """Return whether the portfolio has capacity for another position."""
        return len(self.portfolio.open_positions) < self.config.max_open_positions

    def register_open_position(
        self,
        symbol: str,
        max_loss_per_contract: float,
        quantity: int,
    ) -> None:
        """Track a newly opened position immediately for intra-cycle risk checks."""
        position = {
            "symbol": symbol,
            "max_loss": max(0.0, float(max_loss_per_contract)),
            "quantity": max(1, int(quantity)),
        }
        self.portfolio.open_positions.append(position)
        self.portfolio.total_risk_deployed += (
            position["max_loss"] * position["quantity"] * 100
        )

    def approve_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """Check if a trade is allowed under current risk limits.

        Returns (approved: bool, reason: str).
        """
        if signal.action != "open":
            # Always allow closing trades
            return True, "Close trades are always permitted"

        analysis = signal.analysis
        if analysis is None:
            return False, "No analysis data attached to signal"

        balance = self.portfolio.account_balance

        # ── Check 1: Minimum account balance ──────────────────────
        if balance < self.config.min_account_balance:
            return False, (
                f"Account balance ${balance:,.2f} below minimum "
                f"${self.config.min_account_balance:,.2f}"
            )

        # ── Check 2: Max open positions ───────────────────────────
        num_open = len(self.portfolio.open_positions)
        if num_open >= self.config.max_open_positions:
            return False, (
                f"Max open positions reached: {num_open}/{self.config.max_open_positions}"
            )

        # ── Check 3: Max positions per symbol ─────────────────────
        symbol_count = sum(
            1 for p in self.portfolio.open_positions
            if p.get("symbol") == signal.symbol
        )
        if symbol_count >= self.config.max_positions_per_symbol:
            return False, (
                f"Max positions per symbol reached for {signal.symbol}: "
                f"{symbol_count}/{self.config.max_positions_per_symbol}"
            )

        # ── Check 4: Single position risk ─────────────────────────
        position_max_loss = analysis.max_loss * signal.quantity * 100
        max_position_risk = balance * (self.config.max_position_risk_pct / 100)
        if position_max_loss > max_position_risk:
            return False, (
                f"Position risk ${position_max_loss:,.2f} exceeds max "
                f"${max_position_risk:,.2f} ({self.config.max_position_risk_pct}% of account)"
            )

        # ── Check 5: Total portfolio risk ─────────────────────────
        new_total_risk = self.portfolio.total_risk_deployed + position_max_loss
        max_portfolio_risk = balance * (self.config.max_portfolio_risk_pct / 100)
        if new_total_risk > max_portfolio_risk:
            return False, (
                f"Total portfolio risk ${new_total_risk:,.2f} would exceed max "
                f"${max_portfolio_risk:,.2f} ({self.config.max_portfolio_risk_pct}% of account)"
            )

        # ── Check 6: Daily loss limit ─────────────────────────────
        max_daily_loss = balance * (self.config.max_daily_loss_pct / 100)
        if self.portfolio.daily_pnl < 0 and abs(self.portfolio.daily_pnl) >= max_daily_loss:
            return False, (
                f"Daily loss limit reached: ${abs(self.portfolio.daily_pnl):,.2f} >= "
                f"${max_daily_loss:,.2f} ({self.config.max_daily_loss_pct}% of account)"
            )

        # ── Check 7: Minimum quality score ────────────────────────
        if analysis.score < 40:
            return False, f"Trade score {analysis.score} below minimum threshold 40"

        # ── Check 8: Probability of profit ────────────────────────
        if analysis.probability_of_profit < 0.50:
            return False, (
                f"POP {analysis.probability_of_profit:.1%} below minimum 50%"
            )

        logger.info(
            "Trade APPROVED: %s %s on %s | Risk: $%.2f | Portfolio risk: $%.2f/$%.2f",
            signal.action, signal.strategy, signal.symbol,
            position_max_loss, new_total_risk, max_portfolio_risk,
        )
        return True, "Approved"

    def calculate_position_size(self, signal: TradeSignal) -> int:
        """Calculate the number of contracts to trade.

        Uses a fixed-fraction approach: risk a fixed % of account per trade.
        """
        if signal.analysis is None or signal.analysis.max_loss <= 0:
            return 1

        balance = self.portfolio.account_balance
        max_risk_per_trade = balance * (self.config.max_position_risk_pct / 100)
        risk_per_contract = signal.analysis.max_loss * 100  # multiply by 100 for contract size

        if risk_per_contract <= 0:
            return 1

        contracts = int(max_risk_per_trade / risk_per_contract)
        return max(1, min(contracts, 10))  # Floor of 1, cap at 10 contracts
