"""Risk management system — position sizing, portfolio limits, and guardrails."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bot.config import RiskConfig
from bot.data_store import load_json
from bot.earnings_calendar import EarningsCalendar
from bot.number_utils import safe_float
from bot.strategies.base import TradeSignal

logger = logging.getLogger(__name__)

SECTOR_MAP_PATH = Path(__file__).resolve().parent / "data" / "sector_map.json"


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
        self.sizing_config: dict = {
            "method": "fixed",
            "kelly_fraction": 0.5,
            "kelly_min_trades": 30,
            "drawdown_decay_threshold": 0.05,
            "equity_curve_scaling": True,
            "equity_curve_lookback": 20,
            "max_scale_up": 1.25,
            "max_scale_down": 0.50,
        }
        self.strategy_allocation_config: dict = {
            "enabled": True,
            "lookback_trades": 30,
            "min_sharpe_for_boost": 1.5,
            "cold_start_penalty": 0.75,
            "cold_start_window_days": 60,
            "cold_start_min_trades": 10,
        }
        self.greeks_budget_config: dict = {
            "enabled": True,
            "reduce_size_to_fit": True,
            "limits": {
                "BULL_TREND": {"delta_min": -50.0, "delta_max": 80.0, "vega_min": -200.0, "vega_max": 100.0},
                "BEAR_TREND": {"delta_min": -80.0, "delta_max": 30.0, "vega_min": -100.0, "vega_max": 200.0},
                "HIGH_VOL_CHOP": {"delta_min": -30.0, "delta_max": 30.0, "vega_min": -300.0, "vega_max": -50.0},
                "LOW_VOL_GRIND": {"delta_min": -40.0, "delta_max": 60.0, "vega_min": -150.0, "vega_max": 50.0},
                "CRASH/CRISIS": {"delta_min": -20.0, "delta_max": 10.0, "vega_min": 0.0, "vega_max": 500.0},
                "NORMAL": {"delta_min": -50.0, "delta_max": 50.0, "vega_min": -200.0, "vega_max": 200.0},
            },
        }
        self.portfolio = PortfolioState()
        self.earnings_calendar = EarningsCalendar()
        self.sector_map = self._load_sector_map()
        self._price_history_provider: Optional[Callable[[str, int], list[dict]]] = None
        self._returns_cache: dict[str, np.ndarray] = {}
        self._closed_trades: list[dict] = []
        self._equity_peak: float = 0.0
        self._correlation_matrix: dict[str, dict[str, float]] = {}
        self._var_metrics: dict[str, float] = {"var95": 0.0, "var99": 0.0}

    def set_sizing_config(self, sizing: object) -> None:
        if sizing is None:
            return
        if isinstance(sizing, dict):
            self.sizing_config.update(sizing)
            return
        for key in self.sizing_config:
            if hasattr(sizing, key):
                self.sizing_config[key] = getattr(sizing, key)

    def set_strategy_allocation_config(self, cfg: object) -> None:
        """Configure strategy-level allocation scaling controls."""
        if cfg is None:
            return
        if isinstance(cfg, dict):
            self.strategy_allocation_config.update(cfg)
            return
        for key in self.strategy_allocation_config:
            if hasattr(cfg, key):
                self.strategy_allocation_config[key] = getattr(cfg, key)

    def set_greeks_budget_config(self, cfg: object) -> None:
        """Configure regime-aware Greeks budget controls."""
        if cfg is None:
            return
        if isinstance(cfg, dict):
            self.greeks_budget_config.update(cfg)
            return
        for key in self.greeks_budget_config:
            if hasattr(cfg, key):
                self.greeks_budget_config[key] = getattr(cfg, key)

    def update_trade_history(self, closed_trades: list[dict]) -> None:
        self._closed_trades = [item for item in (closed_trades or []) if isinstance(item, dict)]

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
        self._equity_peak = max(self._equity_peak, self.portfolio.account_balance)
        self._refresh_correlation_matrix()
        self._refresh_var_metrics()

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
        if signal.strategy == "naked_put":
            strike = max(0.0, float(analysis.short_strike))
            credit = max(0.0, float(analysis.credit))
            return round(max(strike - credit, raw_max_loss), 2)

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
        is_hedge = bool((signal.metadata or {}).get("is_hedge"))

        balance = self.portfolio.account_balance

        # ── Check 1: Minimum account balance ──────────────────────
        if balance < self.config.min_account_balance:
            return False, (
                f"Account balance ${balance:,.2f} below minimum "
                f"${self.config.min_account_balance:,.2f}"
            )

        # ── Check 2: Max open positions ───────────────────────────
        num_open = len(self.portfolio.open_positions)
        if not is_hedge and num_open >= self.config.max_open_positions:
            return False, (
                f"Max open positions reached: {num_open}/{self.config.max_open_positions}"
            )

        # ── Check 3: Max positions per symbol ─────────────────────
        direct_symbol_count = sum(
            1 for p in self.portfolio.open_positions if p.get("symbol") == signal.symbol
        )
        if not is_hedge and direct_symbol_count >= self.config.max_positions_per_symbol:
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
        if not is_hedge and analysis.score < 40:
            return False, f"Trade score {analysis.score} below minimum threshold 40"

        # ── Check 8: Probability of profit ────────────────────────
        if not is_hedge and analysis.probability_of_profit < 0.50:
            return False, f"POP {analysis.probability_of_profit:.1%} below minimum 50%"

        # ── Check 9: Earnings in trade window ─────────────────────
        if not is_hedge:
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

        # ── Check 11b: Regime-adaptive Greeks budget ───────────────
        signal_delta = abs(safe_float(getattr(analysis, "net_delta", 0.0), 0.0))
        signal_vega = abs(safe_float(getattr(analysis, "net_vega", 0.0), 0.0))
        if (signal_delta + signal_vega) > 0:
            budget_ok, _, budget_reason = self.evaluate_greeks_budget(
                signal,
                regime=(signal.metadata or {}).get("regime", "NORMAL"),
                quantity=signal.quantity,
                allow_resize=False,
            )
            if not budget_ok:
                return False, budget_reason

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
        if not is_hedge and effective_count >= self.config.max_positions_per_symbol:
            return False, (
                f"Correlation guard: {signal.symbol} is highly correlated with "
                f"{correlated_count} existing positions; effective limit "
                f"{effective_count}/{self.config.max_positions_per_symbol}"
            )

        if self.config.var_enabled:
            var95 = float(self._var_metrics.get("var95", 0.0))
            var99 = float(self._var_metrics.get("var99", 0.0))
            limit95 = balance * (float(self.config.var_limit_pct_95) / 100.0)
            limit99 = balance * (float(self.config.var_limit_pct_99) / 100.0)
            if var95 > limit95 or var99 > limit99:
                return False, (
                    f"Portfolio VaR exceeded limits (95%: {var95:,.2f}/{limit95:,.2f}, "
                    f"99%: {var99:,.2f}/{limit99:,.2f})"
                )

        projected_corr = self._projected_portfolio_correlation(signal.symbol)
        if projected_corr > float(self.config.max_portfolio_correlation):
            return False, (
                f"Portfolio correlation too high: {projected_corr:.2f} > "
                f"{float(self.config.max_portfolio_correlation):.2f}"
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

    def evaluate_greeks_budget(
        self,
        signal: TradeSignal,
        *,
        regime: Optional[str] = None,
        quantity: Optional[int] = None,
        allow_resize: bool = True,
    ) -> tuple[bool, int, str]:
        """Check regime-aware delta/vega budgets and optionally downsize quantity."""
        if signal.action != "open":
            qty = max(1, int(quantity or signal.quantity or 1))
            return True, qty, "Greeks budget not applicable"
        analysis = signal.analysis
        if analysis is None:
            qty = max(1, int(quantity or signal.quantity or 1))
            return False, qty, "No analysis data attached to signal"
        cfg = self.greeks_budget_config
        if not bool(cfg.get("enabled", True)):
            qty = max(1, int(quantity or signal.quantity or 1))
            return True, qty, "Greeks budget disabled"

        limits = self._greeks_limits_for_regime(regime or "NORMAL")
        if not limits:
            qty = max(1, int(quantity or signal.quantity or 1))
            return True, qty, "No Greeks budget limits for regime"

        requested_qty = max(1, int(quantity or signal.quantity or 1))
        delta_unit = safe_float(getattr(analysis, "net_delta", 0.0), 0.0)
        vega_unit = safe_float(getattr(analysis, "net_vega", 0.0), 0.0) * 100.0
        delta_min = safe_float(limits.get("delta_min"), -1e9)
        delta_max = safe_float(limits.get("delta_max"), 1e9)
        vega_min = safe_float(limits.get("vega_min"), -1e9)
        vega_max = safe_float(limits.get("vega_max"), 1e9)

        def _fits(qty: int) -> bool:
            projected_delta = self.portfolio.net_delta + (delta_unit * qty)
            projected_vega = self.portfolio.net_vega + (vega_unit * qty)
            return (
                (delta_min <= projected_delta <= delta_max)
                and (vega_min <= projected_vega <= vega_max)
            )

        if _fits(requested_qty):
            return True, requested_qty, "Greeks budget within limits"

        regime_key = self._normalize_regime_key(regime or "NORMAL")
        if not allow_resize or not bool(cfg.get("reduce_size_to_fit", True)):
            return (
                False,
                0,
                (
                    f"Greeks budget breach in {regime_key}: requested qty {requested_qty} "
                    f"would exceed delta[{delta_min:.1f},{delta_max:.1f}] / "
                    f"vega[{vega_min:.1f},{vega_max:.1f}]"
                ),
            )

        for candidate_qty in range(requested_qty - 1, 0, -1):
            if _fits(candidate_qty):
                return (
                    True,
                    candidate_qty,
                    (
                        f"Greeks budget resized in {regime_key}: {requested_qty} -> {candidate_qty} "
                        f"to stay within delta[{delta_min:.1f},{delta_max:.1f}] / "
                        f"vega[{vega_min:.1f},{vega_max:.1f}]"
                    ),
                )

        return (
            False,
            0,
            (
                f"Greeks budget breach in {regime_key}: minimum qty 1 still exceeds "
                f"delta[{delta_min:.1f},{delta_max:.1f}] / vega[{vega_min:.1f},{vega_max:.1f}]"
            ),
        )

    def _greeks_limits_for_regime(self, regime: str) -> dict:
        limits = self.greeks_budget_config.get("limits", {})
        if not isinstance(limits, dict):
            return {}
        regime_key = self._normalize_regime_key(regime)
        row = limits.get(regime_key)
        if isinstance(row, dict):
            return row
        fallback = limits.get("NORMAL")
        return fallback if isinstance(fallback, dict) else {}

    @staticmethod
    def _normalize_regime_key(regime: str) -> str:
        raw = str(regime or "").strip().upper()
        if not raw:
            return "NORMAL"
        if raw in {"CRASH", "CRISIS", "CRASH_CRISIS", "CRASH/CIRISIS", "CRASH_CRIISIS"}:
            return "CRASH/CRISIS"
        return raw

    def calculate_position_size(self, signal: TradeSignal) -> int:
        """Calculate the number of contracts to trade."""
        if signal.analysis is None or signal.analysis.max_loss <= 0:
            return 1

        balance = self.portfolio.account_balance
        risk_scalar = self._equity_curve_risk_scalar()
        risk_scalar *= self.strategy_allocation_scalar(signal.strategy)
        adjusted_max_position_risk_pct = float(self.config.max_position_risk_pct) * risk_scalar
        max_risk_per_trade = balance * (adjusted_max_position_risk_pct / 100.0)
        risk_per_contract = signal.analysis.max_loss * 100
        if risk_per_contract <= 0:
            return 1

        method = str(self.sizing_config.get("method", "fixed")).strip().lower()
        if method == "kelly":
            fraction = self._kelly_fraction(signal)
            contracts = int((max_risk_per_trade * fraction) / risk_per_contract)
        else:
            contracts = int(max_risk_per_trade / risk_per_contract)
        return max(1, min(contracts, 10))

    def strategy_allocation_scalar(self, strategy_name: str) -> float:
        """Return strategy-level size scalar from rolling Sharpe/cold-start heuristics."""
        cfg = self.strategy_allocation_config
        if not bool(cfg.get("enabled", True)):
            return 1.0
        strategy_key = str(strategy_name or "").strip().lower()
        if not strategy_key:
            return 1.0
        lookback = max(5, int(cfg.get("lookback_trades", 30)))
        min_sharpe_for_boost = float(cfg.get("min_sharpe_for_boost", 1.5))
        cold_penalty = max(0.1, min(1.0, float(cfg.get("cold_start_penalty", 0.75))))
        cold_window = max(7, int(cfg.get("cold_start_window_days", 60)))
        cold_min_trades = max(1, int(cfg.get("cold_start_min_trades", 10)))

        strategy_trades = [
            item
            for item in self._closed_trades
            if isinstance(item, dict) and str(item.get("strategy", "")).strip().lower() == strategy_key
        ]
        if not strategy_trades:
            return cold_penalty

        recent = strategy_trades[-lookback:]
        pnls = np.array(
            [safe_float(item.get("pnl", item.get("realized_pnl", 0.0)), 0.0) for item in recent],
            dtype=float,
        )
        sharpe = 0.0
        if len(pnls) >= 2:
            std = float(np.std(pnls, ddof=1))
            if std > 0:
                sharpe = float((np.mean(pnls) / std) * np.sqrt(len(pnls)))

        scalar = 1.0
        if sharpe < 0.0:
            scalar = 0.5
        elif sharpe > min_sharpe_for_boost:
            scalar = 1.25

        cutoff = datetime.utcnow().date() - timedelta(days=cold_window)
        recent_window_count = 0
        for trade in strategy_trades:
            close_date = str(trade.get("close_date", "")).split("T", 1)[0]
            try:
                trade_day = datetime.strptime(close_date, "%Y-%m-%d").date()
            except Exception:
                continue
            if trade_day >= cutoff:
                recent_window_count += 1
        if recent_window_count < cold_min_trades:
            scalar = min(scalar, cold_penalty)
        return max(0.1, min(1.25, float(scalar)))

    def _equity_curve_slope(self) -> float:
        """Slope of normalized rolling cumulative P&L across recent closed trades."""
        if not bool(self.sizing_config.get("equity_curve_scaling", True)):
            return 0.0
        lookback = max(5, int(self.sizing_config.get("equity_curve_lookback", 20)))
        if len(self._closed_trades) < 2:
            return 0.0
        pnls = [float(item.get("pnl", 0.0) or 0.0) for item in self._closed_trades[-lookback:]]
        if len(pnls) < 2:
            return 0.0
        account = max(1.0, float(self.portfolio.account_balance or 0.0))
        curve = np.cumsum(np.array(pnls, dtype=float) / account)
        try:
            slope = float(np.polyfit(np.arange(len(curve), dtype=float), curve, 1)[0])
        except Exception as exc:
            logger.debug("Equity curve polyfit failed: %s", exc)
            return 0.0
        return slope

    def _equity_curve_risk_scalar(self) -> float:
        """Adaptive risk scalar from recent equity-curve slope (anti-fragile sizing)."""
        if not bool(self.sizing_config.get("equity_curve_scaling", True)):
            return 1.0
        slope = self._equity_curve_slope()
        max_up = max(1.0, float(self.sizing_config.get("max_scale_up", 1.25)))
        max_down = max(0.1, min(1.0, float(self.sizing_config.get("max_scale_down", 0.50))))
        if slope > 0:
            return min(max_up, 1.0 + min(max_up - 1.0, slope * 2.0))
        if slope < 0:
            return max(max_down, 1.0 + (slope * 3.0))
        return 1.0

    def get_portfolio_greeks(self) -> dict:
        """Return current net portfolio Greeks."""
        return {
            "delta": round(self.portfolio.net_delta, 4),
            "theta": round(self.portfolio.net_theta, 4),
            "gamma": round(self.portfolio.net_gamma, 4),
            "vega": round(self.portfolio.net_vega, 4),
        }

    def get_correlation_matrix(self) -> dict:
        return self._correlation_matrix

    def get_var_metrics(self) -> dict:
        return dict(self._var_metrics)

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
        except Exception as exc:
            logger.debug("Correlation price history failed for %s: %s", symbol_key, exc)
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

    def _kelly_fraction(self, signal: TradeSignal) -> float:
        kelly_weight = max(0.0, min(1.0, float(self.sizing_config.get("kelly_fraction", 0.5))))
        min_trades = max(1, int(self.sizing_config.get("kelly_min_trades", 30)))

        win_rate, avg_win, avg_loss = self._historical_edge()
        if win_rate is None or len(self._closed_trades) < min_trades:
            analysis = signal.analysis
            if analysis is None:
                return 1.0
            win_rate = max(0.01, min(0.99, float(analysis.probability_of_profit)))
            avg_win = max(0.01, float(analysis.credit))
            avg_loss = max(0.01, float(analysis.max_loss))

        # Kelly = (p*W - (1-p)*L)/W.
        kelly_raw = ((win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)) / max(avg_win, 1e-9)
        kelly_raw = max(0.0, min(1.0, kelly_raw))
        fractional = kelly_raw * kelly_weight

        drawdown = self._current_drawdown()
        decay_threshold = max(0.0, float(self.sizing_config.get("drawdown_decay_threshold", 0.05)))
        if drawdown > decay_threshold > 0:
            # Anti-martingale: decay sizing when drawdown is elevated.
            fractional *= max(0.25, 1.0 - ((drawdown - decay_threshold) / max(decay_threshold, 1e-9)))

        return max(0.1, min(1.0, fractional))

    def _historical_edge(self) -> tuple[Optional[float], float, float]:
        if not self._closed_trades:
            return None, 0.0, 0.0
        pnls = [float(item.get("pnl", 0.0) or 0.0) for item in self._closed_trades]
        winners = [value for value in pnls if value > 0]
        losers = [abs(value) for value in pnls if value < 0]
        if not winners or not losers:
            return None, 0.0, 0.0
        win_rate = len(winners) / max(1, len(pnls))
        return win_rate, float(np.mean(winners)), float(np.mean(losers))

    def _current_drawdown(self) -> float:
        if self._equity_peak <= 0:
            return 0.0
        current = max(0.0, float(self.portfolio.account_balance))
        return max(0.0, (self._equity_peak - current) / self._equity_peak)

    def _refresh_correlation_matrix(self) -> None:
        symbols = sorted(
            {str(item.get("symbol", "")).upper() for item in self.portfolio.open_positions if item.get("symbol")}
        )
        matrix: dict[str, dict[str, float]] = {}
        if len(symbols) < 2:
            self._correlation_matrix = matrix
            return
        returns = {symbol: self._load_returns(symbol) for symbol in symbols}
        for left in symbols:
            matrix[left] = {}
            for right in symbols:
                if left == right:
                    matrix[left][right] = 1.0
                    continue
                l = returns.get(left)
                r = returns.get(right)
                if l is None or r is None:
                    matrix[left][right] = 0.0
                    continue
                size = min(len(l), len(r))
                if size < 20:
                    matrix[left][right] = 0.0
                    continue
                corr = float(np.corrcoef(l[-size:], r[-size:])[0, 1])
                matrix[left][right] = 0.0 if np.isnan(corr) else round(corr, 6)
        self._correlation_matrix = matrix

    def _refresh_var_metrics(self) -> None:
        if not self.config.var_enabled:
            self._var_metrics = {"var95": 0.0, "var99": 0.0}
            return
        symbols = [str(item.get("symbol", "")).upper() for item in self.portfolio.open_positions if item.get("symbol")]
        if not symbols:
            self._var_metrics = {"var95": 0.0, "var99": 0.0}
            return

        returns = []
        weights = []
        total_risk = max(1.0, float(self.portfolio.total_risk_deployed))
        for position in self.portfolio.open_positions:
            symbol = str(position.get("symbol", "")).upper()
            series = self._load_returns(symbol)
            if series is None or len(series) < 20:
                continue
            risk = max(0.0, float(position.get("max_loss", 0.0))) * max(1, int(position.get("quantity", 1))) * 100.0
            returns.append(series)
            weights.append(risk / total_risk)

        if len(returns) < 1:
            self._var_metrics = {"var95": 0.0, "var99": 0.0}
            return

        min_len = min(len(series) for series in returns)
        if min_len < 20:
            self._var_metrics = {"var95": 0.0, "var99": 0.0}
            return

        aligned = np.vstack([series[-min_len:] for series in returns])
        weighted = np.average(aligned, axis=0, weights=np.array(weights))
        sigma = float(np.std(weighted, ddof=1))
        account = max(0.0, float(self.portfolio.account_balance))
        self._var_metrics = {
            "var95": round(account * 1.645 * sigma, 4),
            "var99": round(account * 2.326 * sigma, 4),
        }

    def _projected_portfolio_correlation(self, symbol: str) -> float:
        symbol = str(symbol).upper().strip()
        if not symbol:
            return 0.0
        target = self._load_returns(symbol)
        if target is None or len(target) < 20:
            return 0.0
        corrs = []
        for position in self.portfolio.open_positions:
            other = str(position.get("symbol", "")).upper().strip()
            if not other or other == symbol:
                continue
            series = self._load_returns(other)
            if series is None or len(series) < 20:
                continue
            size = min(len(target), len(series))
            if size < 20:
                continue
            corr = float(np.corrcoef(target[-size:], series[-size:])[0, 1])
            if np.isnan(corr):
                continue
            corrs.append(corr)
        if not corrs:
            return 0.0
        return float(np.mean(corrs))
