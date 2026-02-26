"""Iron condor strategy — sell OTM put spread + OTM call spread."""

import logging
from typing import Optional

from bot.analysis import analyze_iron_condor, find_option_by_delta, find_spread_wing
from bot.iv_history import IVHistory
from bot.number_utils import safe_float, safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

logger = logging.getLogger(__name__)


class IronCondorStrategy(BaseStrategy):
    """Automated iron-condor strategy with adaptive IV and range filters."""

    def __init__(self, config: dict):
        super().__init__("iron_condors", config)
        self.iv_history = IVHistory()

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        """Scan for iron condor opportunities."""
        signals = []
        min_dte = int(self.config.get("min_dte", 25))
        max_dte = int(self.config.get("max_dte", 50))
        target_delta = float(self.config.get("short_delta", 0.16))
        spread_width = float(self.config.get("spread_width", 5))
        min_credit_pct = float(self.config.get("min_credit_pct", 0.30))

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})
        iv = self._average_chain_iv(calls, puts)
        iv_rank = self.iv_history.update_and_rank(symbol, iv) if iv > 0 else 50.0

        if iv_rank > 70:
            min_dte, max_dte = 20, 30
            spread_width += 1.0
            min_credit_pct *= 1.20
        elif iv_rank < 30:
            min_dte, max_dte = 35, 50
            spread_width = max(1.0, spread_width - 1.0)

        vol_surface = (market_context or {}).get("vol_surface", {})
        if isinstance(vol_surface, dict) and vol_surface:
            vol_of_vol = float(vol_surface.get("vol_of_vol", 0.0) or 0.0)
            max_vov = float(self.config.get("max_vol_of_vol", self.config.get("max_vol_of_vol_for_condors", 0.20)))
            if vol_of_vol > max_vov:
                return []

        # Iron condors prefer range-bound markets.
        if technical_context and not technical_context.between_bands:
            return []

        common_exps = set(calls.keys()) & set(puts.keys())
        for exp_date in common_exps:
            exp_puts = puts[exp_date]
            exp_calls = calls[exp_date]
            if not exp_puts or not exp_calls:
                continue

            dte = int(exp_puts[0].get("dte", 0))
            if dte < min_dte or dte > max_dte:
                continue

            short_put = find_option_by_delta(exp_puts, target_delta)
            if not short_put:
                continue
            long_put = find_spread_wing(exp_puts, short_put["strike"], spread_width, "lower")
            if not long_put:
                continue

            short_call = find_option_by_delta(exp_calls, target_delta)
            if not short_call:
                continue
            long_call = find_spread_wing(exp_calls, short_call["strike"], spread_width, "higher")
            if not long_call:
                continue

            if short_put["strike"] >= short_call["strike"]:
                continue

            analysis = analyze_iron_condor(
                underlying_price, short_put, long_put, short_call, long_call
            )
            analysis.symbol = symbol

            if analysis.credit_pct_of_width < min_credit_pct:
                continue
            if iv_rank > 70:
                analysis.score = min(100.0, analysis.score + 10.0)
            if self.meets_minimum_quality(analysis):
                signals.append(
                    TradeSignal(
                        action="open",
                        strategy="iron_condor",
                        symbol=symbol,
                        analysis=analysis,
                        metadata={"iv_rank": iv_rank},
                    )
                )

        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        max_signals = int(
            (market_context or {}).get(
                "max_signals_per_symbol_per_strategy",
                self.config.get("max_signals_per_symbol_per_strategy", 2),
            )
            or 2
        )
        return signals[: max(1, max_signals)]

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open iron condors for adaptive exits and defend mode."""
        signals = []
        base_stop_loss_pct = float(self.config.get("stop_loss_pct", 2.0))
        adaptive_targets = bool(self.config.get("adaptive_targets", True))
        trailing_enabled = bool(self.config.get("trailing_stop_enabled", False))
        trail_activation = float(self.config.get("trailing_stop_activation_pct", 0.25))
        trail_floor = float(self.config.get("trailing_stop_floor_pct", 0.10))

        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") != "iron_condor":
                continue

            entry_credit = float(pos.get("entry_credit", 0) or 0)
            current_value = float(pos.get("current_value", 0) or 0)
            quantity = max(1, int(pos.get("quantity", 1)))
            dte_remaining = safe_int(pos.get("dte_remaining"), 999)

            pnl = entry_credit - current_value
            pnl_pct = (pnl / entry_credit) if entry_credit > 0 else 0.0
            stop_loss_pct = self._stop_loss_for_position(pos, base_stop_loss_pct)
            if self._is_short_strike_tested(pos):
                stop_loss_pct = min(stop_loss_pct, 1.5)

            if quantity >= 2 and not pos.get("partial_closed", False) and pnl_pct >= 0.40:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy="iron_condor",
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=f"Partial profit at {pnl_pct:.1%}",
                        quantity=max(1, quantity // 2),
                    )
                )
                continue

            details = pos.get("details", {}) or {}
            max_seen = float(details.get("max_profit_pct_seen", pos.get("max_profit_pct_seen", pnl_pct)) or pnl_pct)
            max_seen = max(max_seen, pnl_pct)
            details["max_profit_pct_seen"] = max_seen
            pos["max_profit_pct_seen"] = max_seen

            target_pct = (
                self._profit_target_for_dte(dte_remaining)
                if adaptive_targets
                else float(self.config.get("profit_target_pct", 0.50))
            )
            if pos.get("partial_closed", False):
                target_pct = max(target_pct, 0.65)

            exit_reason = ""
            if entry_credit > 0:
                if trailing_enabled and max_seen >= trail_activation and pnl_pct > 0:
                    trail_lock = self._trailing_lock_pct(
                        max_seen=max_seen,
                        activation=trail_activation,
                        floor=trail_floor,
                    )
                    if pnl_pct <= trail_lock:
                        exit_reason = f"Trailing stop hit ({pnl_pct:.1%} <= {trail_lock:.1%})"

                if not exit_reason and pnl_pct >= target_pct:
                    exit_reason = f"Profit target reached ({pnl_pct:.1%})"
                elif not exit_reason and pnl < 0 and abs(pnl) >= entry_credit * stop_loss_pct:
                    exit_reason = f"Stop loss triggered (loss {abs(pnl):.2f})"

            if not exit_reason and dte_remaining <= 5:
                exit_reason = f"Approaching expiration ({dte_remaining} DTE)"

            if exit_reason:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy="iron_condor",
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=exit_reason,
                        quantity=quantity,
                    )
                )

        return signals

    @staticmethod
    def _profit_target_for_dte(dte_remaining: int) -> float:
        if dte_remaining < 10:
            return 0.20
        if dte_remaining <= 20:
            return 0.30
        if dte_remaining <= 30:
            return 0.40
        return 0.50

    @staticmethod
    def _trailing_lock_pct(*, max_seen: float, activation: float, floor: float) -> float:
        gain_above_activation = max(0.0, max_seen - activation)
        lock = floor + (gain_above_activation * 0.50)
        lock = min(max_seen - 0.01, lock)
        return max(floor, min(max_seen, lock))

    @staticmethod
    def _stop_loss_for_position(position: dict, base_stop_loss_pct: float) -> float:
        details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
        override = safe_float(details.get("stop_loss_override_multiple"), 0.0)
        regime = str(position.get("regime", details.get("regime", ""))).upper()
        iv_rank = float(position.get("iv_rank", details.get("iv_rank", 50.0)) or 50.0)
        result = max(1.0, base_stop_loss_pct)
        if regime in {"CRASH/CRISIS", "CRISIS"}:
            result = 1.5
        elif regime in {"HIGH_VOL_CHOP", "ELEVATED"} or iv_rank >= 70.0:
            result = max(base_stop_loss_pct, 2.5)
        if override > 0:
            result = min(result, override)
        return max(1.0, result)

    @staticmethod
    def _is_short_strike_tested(position: dict) -> bool:
        details = position.get("details", {}) or {}
        underlying = float(position.get("underlying_price", 0.0) or 0.0)
        if underlying <= 0:
            return False
        put_short = float(details.get("put_short_strike", 0.0) or 0.0)
        call_short = float(details.get("call_short_strike", 0.0) or 0.0)
        if put_short <= 0 or call_short <= 0:
            return False
        put_prox = abs(underlying - put_short) / underlying
        call_prox = abs(underlying - call_short) / underlying
        return (
            (underlying <= put_short and put_prox <= 0.01)
            or (underlying >= call_short and call_prox <= 0.01)
        )

    @staticmethod
    def _average_chain_iv(calls: dict, puts: dict) -> float:
        ivs = []
        for exp_options in calls.values():
            ivs.extend(
                float(option.get("iv", 0.0) or 0.0)
                for option in exp_options
                if float(option.get("iv", 0.0) or 0.0) > 0
            )
        for exp_options in puts.values():
            ivs.extend(
                float(option.get("iv", 0.0) or 0.0)
                for option in exp_options
                if float(option.get("iv", 0.0) or 0.0) > 0
            )
        if not ivs:
            return 0.0
        return float(sum(ivs) / len(ivs))
