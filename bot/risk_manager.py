"""Risk management system — position sizing, portfolio limits, and guardrails."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bot.config import RiskConfig
from bot.data_store import load_json
from bot.earnings_calendar import EarningsCalendar
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

SECTOR_MAP_PATH = Path("bot/data/sector_map.json")


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk calculations."""

    account_balance: float = 0.0
    open_positions: list = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_date: Optional[date] = None
    total_risk_deployed: float = 0.0
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0
    sector_risk: dict = field(default_factory=dict)


class RiskManager:
    """Enforces risk limits before any trade is executed."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.portfolio = PortfolioState()
        self.earnings_calendar = EarningsCalendar()
        self.sector_map = self._load_sector_map()
        self._price_history_provider: Optional[Callable[[str, int], list[dict]]] = None
        self._returns_cache: dict[str, np.ndarray] = {}

    def set_price_history_provider(self, provider: Optional[Callable[[str, int], list[dict]]]) -> None:
        """Register a callable used for correlation checks."""
        self._price_history_provider = provider
        self._returns_cache.clear()

    def update_portfolio(
        self,
        account_balance: float,
        open_positions: list,
        daily_pnl: float = 0.0,
    ) -> None:
        """Update the portfolio state with latest data."""
        self.portfolio.account_balance = float(account_balance)
        self.portfolio.open_positions = open_positions

        today = date.today()
        if self.portfolio.daily_pnl_date != today:
            self.portfolio.daily_pnl_date = today
        self.portfolio.daily_pnl = float(daily_pnl)

        total_risk = 0.0
        net_delta = 0.0
        net_theta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        sector_risk: dict[str, float] = {}

        for pos in open_positions:
            quantity = max(1, int(pos.get("quantity", 1)))
            max_loss = max(0.0, float(pos.get("max_loss", 0.0)))
            pos_risk = max_loss * quantity * 100.0
            total_risk += pos_risk

            details = pos.get("details", {}) if isinstance(pos.get("details"), dict) else {}
            net_delta += float(details.get("net_delta", pos.get("net_delta", 0.0)) or 0.0) * quantity
            net_theta += float(details.get("net_theta", pos.get("net_theta", 0.0)) or 0.0) * quantity
            net_gamma += float(details.get("net_gamma", pos.get("net_gamma", 0.0)) or 0.0) * quantity
            net_vega += float(details.get("net_vega", pos.get("net_vega", 0.0)) or 0.0) * quantity * 100.0

            symbol = str(pos.get("symbol", "")).upper()
            sector = self.sector_map.get(symbol, "Unknown")
            sector_risk[sector] = sector_risk.get(sector, 0.0) + pos_risk

        self.portfolio.total_risk_deployed = total_risk
        self.portfolio.net_delta = round(net_delta, 4)
        self.portfolio.net_theta = round(net_theta, 4)
        self.portfolio.net_gamma = round(net_gamma, 4)
        self.portfolio.net_vega = round(net_vega, 4)
        self.portfolio.sector_risk = sector_risk

    def can_open_more_positions(self) -> bool:
        """Return whether the portfolio has capacity for another position."""
        return len(self.portfolio.open_positions) < self.config.max_open_positions

    def register_open_position(
        self,
        symbol: str,
        max_loss_per_contract: float,
        quantity: int,
        strategy: str = "",
        greeks: Optional[dict] = None,
    ) -> None:
        """Track a newly opened position immediately for intra-cycle risk checks."""
        position = {
            "symbol": symbol,
            "max_loss": max(0.0, float(max_loss_per_contract)),
            "quantity": max(1, int(quantity)),
            "strategy": strategy,
            "details": {
                "net_delta": float((greeks or {}).get("net_delta", 0.0)),
                "net_theta": float((greeks or {}).get("net_theta", 0.0)),
                "net_gamma": float((greeks or {}).get("net_gamma", 0.0)),
                "net_vega": float((greeks or {}).get("net_vega", 0.0)),
            },
        }
        self.portfolio.open_positions.append(position)
        self.portfolio.total_risk_deployed += (
            position["max_loss"] * position["quantity"] * 100
        )
        self.portfolio.net_delta += position["details"]["net_delta"] * position["quantity"]
        self.portfolio.net_theta += position["details"]["net_theta"] * position["quantity"]
        self.portfolio.net_gamma += position["details"]["net_gamma"] * position["quantity"]
        self.portfolio.net_vega += position["details"]["net_vega"] * position["quantity"] * 100.0

    def effective_max_loss_per_contract(self, signal: TradeSignal) -> float:
        """Return risk-model max loss per contract for a signal."""
        analysis = signal.analysis
        if analysis is None:
            return 0.0

        raw_max_loss = max(0.0, float(analysis.max_loss))
        if signal.strategy != "covered_call" or raw_max_loss > 0:
            return raw_max_loss

        short_strike = max(0.0, float(analysis.short_strike))
        notional_proxy = short_strike
        downside_pct = self.config.covered_call_notional_risk_pct / 100.0
        risk_proxy = (notional_proxy * downside_pct) - max(0.0, float(analysis.credit))
        return round(max(risk_proxy, max(0.25, float(analysis.credit))), 2)

    def approve_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """Check if a trade is allowed under current risk limits."""
        if signal.action != "open":
            return True, "Close/roll trades are always permitted"

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
        direct_symbol_count = sum(
            1 for p in self.portfolio.open_positions if p.get("symbol") == signal.symbol
        )
        if direct_symbol_count >= self.config.max_positions_per_symbol:
            return False, (
                f"Max positions per symbol reached for {signal.symbol}: "
                f"{direct_symbol_count}/{self.config.max_positions_per_symbol}"
            )

        # ── Check 4: Single position risk ─────────────────────────
        loss_per_contract = self.effective_max_loss_per_contract(signal)
        position_max_loss = loss_per_contract * signal.quantity * 100
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
            return False, f"POP {analysis.probability_of_profit:.1%} below minimum 50%"

        # ── Check 9: Earnings in trade window ─────────────────────
        in_window, earnings_date = self.earnings_calendar.earnings_within_window(
            signal.symbol,
            analysis.expiration,
        )
        if in_window:
            logger.info(
                "Skipping %s: earnings on %s falls within %s",
                signal.symbol,
                earnings_date,
                analysis.expiration,
            )
            return False, (
                f"Earnings event on {earnings_date} before expiration {analysis.expiration}"
            )

        # ── Check 10: Portfolio net-delta guard ───────────────────
        projected_delta = self.portfolio.net_delta + (float(analysis.net_delta) * signal.quantity)
        delta_limit = max(0.0, float(self.config.max_portfolio_delta_abs))
        if abs(projected_delta) > delta_limit:
            same_direction = np.sign(projected_delta) == np.sign(self.portfolio.net_delta or projected_delta)
            if same_direction:
                return False, (
                    f"Portfolio delta limit exceeded: {projected_delta:.2f} "
                    f"vs ±{delta_limit:.2f}"
                )

        # ── Check 11: Portfolio net-vega guard ────────────────────
        vega_limit = balance * (self.config.max_portfolio_vega_pct_of_account / 100.0)
        projected_vega = self.portfolio.net_vega + (float(analysis.net_vega) * signal.quantity * 100.0)
        if abs(projected_vega) > vega_limit and abs(float(analysis.net_vega)) > 0:
            return False, (
                f"Portfolio vega limit exceeded: {projected_vega:.2f} "
                f"vs ±{vega_limit:.2f}"
            )

        # ── Check 12: Sector concentration + correlation guard ────
        sector = self.sector_map.get(signal.symbol.upper(), "Unknown")
        sector_risk_after = self.portfolio.sector_risk.get(sector, 0.0) + position_max_loss
        max_sector_fraction = self.config.max_sector_risk_pct / 100.0
        if new_total_risk > 0 and (sector_risk_after / new_total_risk) > max_sector_fraction:
            return False, (
                f"Sector concentration limit exceeded in {sector}: "
                f"{sector_risk_after / new_total_risk:.1%} > {max_sector_fraction:.1%}"
            )

        correlated_count = self._count_correlated_positions(signal.symbol)
        effective_count = direct_symbol_count + correlated_count
        if effective_count >= self.config.max_positions_per_symbol:
            return False, (
                f"Correlation guard: {signal.symbol} is highly correlated with "
                f"{correlated_count} existing positions; effective limit "
                f"{effective_count}/{self.config.max_positions_per_symbol}"
            )

        logger.info(
            "Trade APPROVED: %s %s on %s | Risk: $%.2f | Portfolio risk: $%.2f/$%.2f",
            signal.action,
            signal.strategy,
            signal.symbol,
            position_max_loss,
            new_total_risk,
            max_portfolio_risk,
        )
        return True, "Approved"

    def calculate_position_size(self, signal: TradeSignal) -> int:
        """Calculate the number of contracts to trade."""
        if signal.analysis is None or signal.analysis.max_loss <= 0:
            return 1

        balance = self.portfolio.account_balance
        max_risk_per_trade = balance * (self.config.max_position_risk_pct / 100)
        risk_per_contract = signal.analysis.max_loss * 100
        if risk_per_contract <= 0:
            return 1

        contracts = int(max_risk_per_trade / risk_per_contract)
        return max(1, min(contracts, 10))

    def get_portfolio_greeks(self) -> dict:
        """Return current net portfolio Greeks."""
        return {
            "delta": round(self.portfolio.net_delta, 4),
            "theta": round(self.portfolio.net_theta, 4),
            "gamma": round(self.portfolio.net_gamma, 4),
            "vega": round(self.portfolio.net_vega, 4),
        }

    def _load_sector_map(self) -> dict[str, str]:
        payload = load_json(SECTOR_MAP_PATH, {})
        if not isinstance(payload, dict):
            return {}
        out: dict[str, str] = {}
        for symbol, sector in payload.items():
            out[str(symbol).upper()] = str(sector)
        return out

    def _count_correlated_positions(self, symbol: str) -> int:
        if self._price_history_provider is None:
            return 0

        target_returns = self._load_returns(symbol)
        if target_returns is None or len(target_returns) < 20:
            return 0

        count = 0
        for position in self.portfolio.open_positions:
            other_symbol = str(position.get("symbol", "")).upper()
            if not other_symbol or other_symbol == symbol.upper():
                continue
            other_returns = self._load_returns(other_symbol)
            if other_returns is None or len(other_returns) < 20:
                continue
            size = min(len(target_returns), len(other_returns))
            if size < 20:
                continue
            corr = float(np.corrcoef(target_returns[-size:], other_returns[-size:])[0, 1])
            if np.isnan(corr):
                continue
            if corr >= float(self.config.correlation_threshold):
                count += 1
        return count

    def _load_returns(self, symbol: str) -> Optional[np.ndarray]:
        symbol_key = symbol.upper().strip()
        if symbol_key in self._returns_cache:
            return self._returns_cache[symbol_key]

        if self._price_history_provider is None:
            return None

        try:
            bars = self._price_history_provider(
                symbol_key,
                int(self.config.correlation_lookback_days) + 5,
            )
        except Exception:
            return None
        if not isinstance(bars, list):
            return None

        closes = np.array(
            [float(item.get("close", 0.0) or 0.0) for item in bars if isinstance(item, dict)],
            dtype=float,
        )
        closes = closes[closes > 0]
        if len(closes) < 20:
            return None

        returns = np.diff(closes) / closes[:-1]
        self._returns_cache[symbol_key] = returns
        return returns
