"""Short strangles / straddles strategy for high-IV range regimes."""

from __future__ import annotations

from typing import Optional

from bot.analysis import SpreadAnalysis, find_option_by_delta, find_option_by_strike
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

ALLOWED_REGIMES = {"HIGH_VOL_CHOP", "LOW_VOL_GRIND"}


class StranglesStrategy(BaseStrategy):
    """Sell high-IV premium using short strangles/straddles under strict filters."""

    def __init__(self, config: dict):
        super().__init__("strangles", config)

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        if underlying_price <= 0:
            return []

        iv_rank = float((market_context or {}).get("iv_rank", 0.0) or 0.0)
        if iv_rank < float(self.config.get("min_iv_rank", 70.0)):
            return []

        regime = str((market_context or {}).get("regime", "")).upper()
        if regime == "CRASH/CRISIS":
            return []
        if regime and regime not in ALLOWED_REGIMES:
            return []
        vol_surface = (market_context or {}).get("vol_surface", {})
        if isinstance(vol_surface, dict) and vol_surface:
            vol_of_vol = float(vol_surface.get("vol_of_vol", 0.0) or 0.0)
            max_vov = float(self.config.get("max_vol_of_vol", self.config.get("max_vol_of_vol_for_condors", 0.20)))
            if vol_of_vol > max_vov:
                return []

        account_balance = float((market_context or {}).get("account_balance", 0.0) or 0.0)
        min_account_balance = float(self.config.get("min_account_balance", 25_000.0))
        if account_balance > 0 and account_balance < min_account_balance:
            return []

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})
        if not calls or not puts:
            return []

        min_dte = int(self.config.get("min_dte", 20))
        max_dte = int(self.config.get("max_dte", 45))
        target_delta = float(self.config.get("short_delta", 0.16))
        allow_straddles = {str(item).upper() for item in self.config.get("allow_straddles_on", ["SPY", "QQQ", "IWM"])}

        signals: list[TradeSignal] = []
        for exp in sorted(set(calls.keys()) & set(puts.keys())):
            exp_calls = calls.get(exp, [])
            exp_puts = puts.get(exp, [])
            if not exp_calls or not exp_puts:
                continue
            dte = int(exp_calls[0].get("dte", 0) or 0)
            if dte < min_dte or dte > max_dte:
                continue

            short_call = find_option_by_delta(exp_calls, target_delta)
            short_put = find_option_by_delta(exp_puts, target_delta)
            if not short_call or not short_put:
                continue
            if float(short_put.get("strike", 0.0)) >= float(short_call.get("strike", 0.0)):
                continue

            credit = float(short_call.get("mid", 0.0) or 0.0) + float(short_put.get("mid", 0.0) or 0.0)
            if credit <= 0:
                continue
            max_loss_proxy = max(underlying_price * 0.20 - credit, credit)
            pop = max(0.0, min(1.0, 1.0 - (abs(float(short_put.get("delta", 0.0))) + abs(float(short_call.get("delta", 0.0)))))
            )

            analysis = SpreadAnalysis(
                symbol=symbol,
                strategy="short_strangle",
                expiration=exp,
                dte=dte,
                short_strike=float(short_put.get("strike", 0.0)),
                long_strike=float(short_call.get("strike", 0.0)),
                put_short_strike=float(short_put.get("strike", 0.0)),
                call_short_strike=float(short_call.get("strike", 0.0)),
                credit=round(credit, 4),
                max_loss=round(max_loss_proxy, 4),
                max_profit=round(credit, 4),
                probability_of_profit=round(pop, 4),
                net_delta=round(
                    -float(short_put.get("delta", 0.0) or 0.0) - float(short_call.get("delta", 0.0) or 0.0),
                    4,
                ),
                net_theta=round(
                    -float(short_put.get("theta", 0.0) or 0.0) - float(short_call.get("theta", 0.0) or 0.0),
                    4,
                ),
                net_gamma=round(
                    -float(short_put.get("gamma", 0.0) or 0.0) - float(short_call.get("gamma", 0.0) or 0.0),
                    4,
                ),
                net_vega=round(
                    -float(short_put.get("vega", 0.0) or 0.0) - float(short_call.get("vega", 0.0) or 0.0),
                    4,
                ),
                score=round(min(100.0, pop * 100.0 + min(credit / 2.0, 15.0)), 1),
            )
            signals.append(
                TradeSignal(
                    action="open",
                    strategy="short_strangle",
                    symbol=symbol,
                    analysis=analysis,
                    metadata={"iv_rank": iv_rank, "regime": regime},
                )
            )

            if symbol.upper() in allow_straddles:
                atm_strike = min(
                    [float(item.get("strike", 0.0) or 0.0) for item in exp_calls if float(item.get("strike", 0.0) or 0.0) > 0],
                    key=lambda strike: abs(strike - underlying_price),
                    default=0.0,
                )
                if atm_strike > 0:
                    atm_call = find_option_by_strike(exp_calls, atm_strike)
                    atm_put = find_option_by_strike(exp_puts, atm_strike)
                    if atm_call and atm_put:
                        straddle_credit = float(atm_call.get("mid", 0.0) or 0.0) + float(atm_put.get("mid", 0.0) or 0.0)
                        if straddle_credit > 0:
                            straddle_analysis = SpreadAnalysis(
                                symbol=symbol,
                                strategy="short_straddle",
                                expiration=exp,
                                dte=dte,
                                short_strike=atm_strike,
                                long_strike=atm_strike,
                                put_short_strike=atm_strike,
                                call_short_strike=atm_strike,
                                credit=round(straddle_credit, 4),
                                max_loss=round(max(underlying_price * 0.25 - straddle_credit, straddle_credit), 4),
                                max_profit=round(straddle_credit, 4),
                                probability_of_profit=0.50,
                                score=round(min(100.0, 55.0 + min(straddle_credit, 10.0)), 1),
                            )
                            signals.append(
                                TradeSignal(
                                    action="open",
                                    strategy="short_straddle",
                                    symbol=symbol,
                                    analysis=straddle_analysis,
                                    metadata={"iv_rank": iv_rank, "regime": regime},
                                )
                            )

        signals.sort(key=lambda item: item.analysis.score if item.analysis else 0.0, reverse=True)
        max_signals = int(
            (market_context or {}).get(
                "max_signals_per_symbol_per_strategy",
                self.config.get("max_signals_per_symbol_per_strategy", 2),
            )
            or 2
        )
        return signals[: max(1, max_signals)]

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        out: list[TradeSignal] = []
        adaptive_targets = bool(self.config.get("adaptive_targets", True))
        trailing_enabled = bool(self.config.get("trailing_stop_enabled", False))
        trail_activation = float(self.config.get("trailing_stop_activation_pct", 0.25))
        trail_floor = float(self.config.get("trailing_stop_floor_pct", 0.10))
        stop_multiple = float(self.config.get("stop_loss_multiple", 2.0))
        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") not in {"short_strangle", "short_straddle"}:
                continue
            entry_credit = float(pos.get("entry_credit", 0.0) or 0.0)
            current_value = float(pos.get("current_value", 0.0) or 0.0)
            dte_remaining = int(pos.get("dte_remaining", 999) or 999)
            if entry_credit <= 0:
                continue
            details = pos.get("details", {}) if isinstance(pos.get("details"), dict) else {}
            stop_override = float(details.get("stop_loss_override_multiple", 0.0) or 0.0)
            effective_stop_multiple = stop_multiple
            if stop_override > 0:
                effective_stop_multiple = min(effective_stop_multiple, stop_override)
            pnl_pct = (entry_credit - current_value) / entry_credit
            target = (
                self._profit_target_for_dte(dte_remaining)
                if adaptive_targets
                else float(self.config.get("profit_target_pct", 0.50))
            )
            trailing_high = float(pos.get("trailing_stop_high", details.get("trailing_stop_high", 0.0)) or 0.0)
            if trailing_enabled and pnl_pct >= trail_activation:
                trailing_high = max(trailing_high, pnl_pct)
                pos["trailing_stop_high"] = trailing_high
                details["trailing_stop_high"] = trailing_high
                pos["details"] = details

            trailing_triggered = (
                trailing_enabled
                and trailing_high > 0
                and pnl_pct <= max(0.0, trailing_high - trail_floor)
            )
            if pnl_pct >= target or trailing_triggered or current_value >= entry_credit * effective_stop_multiple:
                if trailing_triggered:
                    reason = "Trailing stop"
                elif pnl_pct >= target:
                    reason = "Profit target"
                else:
                    reason = "Stop loss"
                out.append(
                    TradeSignal(
                        action="close",
                        strategy=str(pos.get("strategy", "")),
                        symbol=str(pos.get("symbol", "")),
                        position_id=pos.get("position_id"),
                        reason=reason,
                        quantity=max(1, int(pos.get("quantity", 1))),
                    )
                )
        return out

    @staticmethod
    def _profit_target_for_dte(dte_remaining: int) -> float:
        if dte_remaining < 10:
            return 0.20
        if dte_remaining <= 20:
            return 0.30
        if dte_remaining <= 30:
            return 0.40
        return 0.50
