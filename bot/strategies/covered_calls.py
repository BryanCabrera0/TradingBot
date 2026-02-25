"""Covered call strategy — sell calls against owned stock positions."""

import logging
from typing import Optional

from bot.analysis import SpreadAnalysis, find_option_by_delta
from bot.dividend_calendar import DividendCalendar
from bot.iv_history import IVHistory
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext

logger = logging.getLogger(__name__)


class CoveredCallStrategy(BaseStrategy):
    """Automated covered-call strategy with adaptive IV and trend-aware delta."""

    def __init__(self, config: dict):
        super().__init__("covered_calls", config)
        self.iv_history = IVHistory()
        self.dividend_calendar = DividendCalendar()

    def scan_for_entries(
        self,
        symbol: str,
        chain_data: dict,
        underlying_price: float,
        technical_context: Optional[TechnicalContext] = None,
        market_context: Optional[dict] = None,
    ) -> list[TradeSignal]:
        """Scan for covered call opportunities on stocks we own."""
        signals = []
        min_dte = int(self.config.get("min_dte", 20))
        max_dte = int(self.config.get("max_dte", 45))
        target_delta = float(self.config.get("short_delta", 0.30))
        size_multiplier = 1.0

        allowed_tickers = self.config.get("tickers", [])
        if allowed_tickers and symbol not in allowed_tickers:
            return signals

        calls = chain_data.get("calls", {})
        iv = self._average_call_iv(calls)
        iv_rank = self.iv_history.update_and_rank(symbol, iv) if iv > 0 else 50.0

        if iv_rank > 70:
            min_dte, max_dte = 20, 30
        elif iv_rank < 30:
            min_dte, max_dte = 35, 50
            size_multiplier = 0.70

        if technical_context and technical_context.close > technical_context.sma50:
            target_delta = min(0.45, target_delta + 0.10)

        for exp_date, exp_calls in calls.items():
            if not exp_calls:
                continue
            dte = int(exp_calls[0].get("dte", 0) or 0)
            if dte < min_dte or dte > max_dte:
                continue

            short_call = find_option_by_delta(exp_calls, target_delta)
            if not short_call:
                continue

            premium = float(short_call["mid"])
            if premium <= 0.05:
                continue

            roc = premium / underlying_price if underlying_price > 0 else 0.0
            annualized_roc = roc * (365 / dte) if dte > 0 else 0.0
            if annualized_roc < 0.05:
                continue

            analysis = SpreadAnalysis(
                symbol=symbol,
                strategy="covered_call",
                expiration=exp_date,
                dte=dte,
                short_strike=float(short_call["strike"]),
                long_strike=0,
                credit=premium,
                max_profit=premium,
                max_loss=0,  # Managed by risk manager as notional proxy.
                probability_of_profit=1.0 - abs(float(short_call["delta"])),
                net_delta=float(short_call["delta"]),
                net_theta=float(short_call["theta"]),
                net_gamma=float(short_call.get("gamma", 0.0) or 0.0),
                net_vega=float(short_call["vega"]),
                score=self._score_covered_call(short_call, premium, annualized_roc, dte),
            )

            div_risk = self.dividend_calendar.assess_trade_risk(
                symbol=symbol,
                strategy="covered_call",
                expiration=exp_date,
                short_strike=float(short_call["strike"]),
                underlying_price=float(underlying_price),
                is_call_side=True,
            )
            analysis.score = round(
                max(0.0, min(100.0, analysis.score + float(div_risk.get("score_adjustment", 0.0)))),
                1,
            )

            signals.append(
                TradeSignal(
                    action="open",
                    strategy="covered_call",
                    symbol=symbol,
                    analysis=analysis,
                    size_multiplier=size_multiplier,
                    metadata={
                        "iv_rank": iv_rank,
                        "dividend_warning": div_risk.get("warning"),
                    },
                )
            )

        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open covered calls for adaptive profit targets and partial exits."""
        signals = []

        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") != "covered_call":
                continue

            entry_credit = float(pos.get("entry_credit", 0) or 0)
            current_value = float(pos.get("current_value", 0) or 0)
            quantity = max(1, int(pos.get("quantity", 1)))
            dte_remaining = int(pos.get("dte_remaining", 999) or 999)
            pnl = entry_credit - current_value
            pnl_pct = (pnl / entry_credit) if entry_credit > 0 else 0.0

            if quantity >= 2 and not pos.get("partial_closed", False) and pnl_pct >= 0.40:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy="covered_call",
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=f"Partial profit at {pnl_pct:.1%}",
                        quantity=max(1, quantity // 2),
                    )
                )
                continue

            target_pct = self._profit_target_for_dte(dte_remaining)
            if pos.get("partial_closed", False):
                target_pct = max(target_pct, 0.65)

            exit_reason = ""
            if entry_credit > 0 and pnl_pct >= target_pct:
                exit_reason = f"Profit target reached ({pnl_pct:.1%})"
            elif dte_remaining <= 3:
                exit_reason = f"Approaching expiration ({dte_remaining} DTE)"

            if exit_reason:
                signals.append(
                    TradeSignal(
                        action="close",
                        strategy="covered_call",
                        symbol=pos.get("symbol", ""),
                        position_id=pos.get("position_id"),
                        reason=exit_reason,
                        quantity=quantity,
                    )
                )

        return signals

    @staticmethod
    def _profit_target_for_dte(dte_remaining: int) -> float:
        if dte_remaining <= 14:
            return 0.25
        if dte_remaining <= 30:
            return 0.50
        return 0.65

    @staticmethod
    def _score_covered_call(
        call_option: dict, premium: float, annualized_roc: float, dte: int
    ) -> float:
        """Score a covered call from 0-100."""
        score = 0.0

        score += min(annualized_roc / 0.30, 1.0) * 35
        prob_otm = 1.0 - abs(float(call_option.get("delta", 0.3)))
        score += prob_otm * 30

        oi = float(call_option.get("open_interest", 0) or 0)
        vol = float(call_option.get("volume", 0) or 0)
        liquidity = min(oi / 500, 1.0) * 0.5 + min(vol / 100, 1.0) * 0.5
        score += liquidity * 15

        if 20 <= dte <= 45:
            score += 10
        elif 14 <= dte < 20 or 45 < dte <= 55:
            score += 5

        ba_spread = float(call_option.get("ask", 0) or 0) - float(call_option.get("bid", 0) or 0)
        tightness = max(0.0, 1.0 - ba_spread / 0.30)
        score += tightness * 10

        return round(min(score, 100), 1)

    @staticmethod
    def _average_call_iv(calls: dict) -> float:
        ivs = []
        for exp_options in calls.values():
            ivs.extend(
                float(option.get("iv", 0.0) or 0.0)
                for option in exp_options
                if float(option.get("iv", 0.0) or 0.0) > 0
            )
        if not ivs:
            return 0.0
        return float(sum(ivs) / len(ivs))
