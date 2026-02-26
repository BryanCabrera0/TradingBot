"""Calendar spread strategy (debit spread)."""

from __future__ import annotations

import logging
from typing import Optional

from bot.analysis import SpreadAnalysis, find_option_by_strike
from bot.number_utils import safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

logger = logging.getLogger(__name__)


class CalendarSpreadStrategy(BaseStrategy):
    """ATM call calendar spread: short front month, long back month."""

    def __init__(self, config: dict):
        super().__init__("calendar_spreads", config)

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        calls = chain_data.get("calls", {})
        if not calls or underlying_price <= 0:
            return []
        vol_surface = (market_context or {}).get("vol_surface", {})
        if isinstance(vol_surface, dict) and vol_surface:
            regime = str(vol_surface.get("term_structure_regime", "")).lower()
            if regime and regime != "backwardation":
                return []

        front_min_dte = int(self.config.get("front_min_dte", 20))
        front_max_dte = int(self.config.get("front_max_dte", 30))
        back_min_dte = int(self.config.get("back_min_dte", 50))
        back_max_dte = int(self.config.get("back_max_dte", 60))

        fronts = self._expirations_in_range(calls, front_min_dte, front_max_dte)
        backs = self._expirations_in_range(calls, back_min_dte, back_max_dte)
        if not fronts or not backs:
            return []

        signals: list[TradeSignal] = []
        for front_exp in fronts:
            front_options = calls.get(front_exp, [])
            front_atm = self._find_atm_option(front_options, underlying_price)
            if not front_atm:
                continue

            strike = float(front_atm.get("strike", 0.0) or 0.0)
            if strike <= 0:
                continue

            for back_exp in backs:
                if back_exp == front_exp:
                    continue
                back_options = calls.get(back_exp, [])
                back_option = find_option_by_strike(back_options, strike)
                if not back_option:
                    continue

                front_mid = float(front_atm.get("mid", 0.0) or 0.0)
                back_mid = float(back_option.get("mid", 0.0) or 0.0)
                if front_mid <= 0 or back_mid <= 0:
                    continue

                debit = back_mid - front_mid
                if debit <= 0:
                    continue
                if debit >= 0.5 * front_mid:
                    continue

                front_iv = float(front_atm.get("iv", 0.0) or 0.0)
                back_iv = float(back_option.get("iv", 0.0) or 0.0)
                if front_iv <= back_iv:
                    continue

                front_dte = int(front_atm.get("dte", 0) or 0)
                back_dte = int(back_option.get("dte", 0) or 0)
                mark_value = back_mid - front_mid
                if mark_value <= 0:
                    continue

                analysis = SpreadAnalysis(
                    symbol=symbol,
                    strategy="calendar_spread",
                    expiration=front_exp,
                    dte=front_dte,
                    short_strike=strike,
                    long_strike=strike,
                    credit=round(-debit, 4),  # debit paid (negative credit)
                    max_loss=round(debit, 4),
                    max_profit=round(debit * 3.0, 4),
                    risk_reward_ratio=round((debit * 3.0) / debit, 4),
                    credit_pct_of_width=round(min(debit / front_mid, 1.0), 4),
                    probability_of_profit=0.62,
                    net_delta=round(
                        float(back_option.get("delta", 0.0) or 0.0)
                        - float(front_atm.get("delta", 0.0) or 0.0),
                        4,
                    ),
                    net_theta=round(
                        float(back_option.get("theta", 0.0) or 0.0)
                        - float(front_atm.get("theta", 0.0) or 0.0),
                        4,
                    ),
                    net_gamma=round(
                        float(back_option.get("gamma", 0.0) or 0.0)
                        - float(front_atm.get("gamma", 0.0) or 0.0),
                        4,
                    ),
                    net_vega=round(
                        float(back_option.get("vega", 0.0) or 0.0)
                        - float(front_atm.get("vega", 0.0) or 0.0),
                        4,
                    ),
                    score=self._score_setup(
                        debit=debit,
                        front_mid=front_mid,
                        front_iv=front_iv,
                        back_iv=back_iv,
                    ),
                )
                if not self.meets_minimum_quality(analysis):
                    continue

                signals.append(
                    TradeSignal(
                        action="open",
                        strategy="calendar_spread",
                        symbol=symbol,
                        analysis=analysis,
                        metadata={
                            "position_details": {
                                "front_expiration": front_exp,
                                "back_expiration": back_exp,
                                "front_dte": front_dte,
                                "back_dte": back_dte,
                                "strike": strike,
                                # For paper execution, mark the opening debit with negative value.
                                "mark_sign": -1,
                            },
                            "front_iv": front_iv,
                            "back_iv": back_iv,
                        },
                    )
                )

        signals.sort(key=lambda item: item.analysis.score if item.analysis else 0.0, reverse=True)
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        signals: list[TradeSignal] = []
        profit_target_pct = float(self.config.get("profit_target_pct", 0.25))
        exit_dte = int(self.config.get("exit_dte", 7))

        for position in positions:
            if str(position.get("status", "open")).lower() != "open":
                continue
            if str(position.get("strategy", "")).lower() != "calendar_spread":
                continue

            entry_debit = abs(float(position.get("entry_credit", 0.0) or 0.0))
            current_mark = abs(float(position.get("current_value", 0.0) or 0.0))
            dte_remaining = safe_int(position.get("dte_remaining"), 999)
            quantity = max(1, int(position.get("quantity", 1)))

            profit_target = entry_debit * (1.0 + profit_target_pct)
            should_close = dte_remaining <= exit_dte or (
                entry_debit > 0 and current_mark >= profit_target
            )
            if not should_close:
                continue

            reason = (
                f"Profit target reached ({current_mark:.2f} >= {profit_target:.2f})"
                if entry_debit > 0 and current_mark >= profit_target
                else f"Front DTE reached ({dte_remaining})"
            )
            signals.append(
                TradeSignal(
                    action="close",
                    strategy="calendar_spread",
                    symbol=str(position.get("symbol", "")),
                    position_id=position.get("position_id"),
                    reason=reason,
                    quantity=quantity,
                )
            )

        return signals

    def meets_minimum_quality(self, analysis: SpreadAnalysis) -> bool:
        debit = abs(float(analysis.credit))
        if debit <= 0:
            return False
        if analysis.probability_of_profit < 0.55:
            return False
        if analysis.score < 40:
            return False
        return True

    @staticmethod
    def _expirations_in_range(calls: dict, min_dte: int, max_dte: int) -> list[str]:
        matches: list[str] = []
        for expiration, options in calls.items():
            if not options:
                continue
            dte = int(options[0].get("dte", 0) or 0)
            if min_dte <= dte <= max_dte:
                matches.append(expiration)
        return matches

    @staticmethod
    def _find_atm_option(options: list[dict], underlying_price: float) -> Optional[dict]:
        best = None
        best_diff = float("inf")
        for option in options:
            strike = float(option.get("strike", 0.0) or 0.0)
            if strike <= 0:
                continue
            diff = abs(strike - underlying_price)
            if diff < best_diff:
                best_diff = diff
                best = option
        return best

    @staticmethod
    def _score_setup(*, debit: float, front_mid: float, front_iv: float, back_iv: float) -> float:
        score = 0.0
        cost_ratio = debit / max(front_mid, 0.01)
        score += max(0.0, 1.0 - min(cost_ratio / 0.5, 1.0)) * 45.0
        term_structure = max(0.0, (front_iv - back_iv) / max(front_iv, 1e-9))
        score += min(term_structure / 0.3, 1.0) * 35.0
        score += 20.0
        return round(min(score, 100.0), 1)
