"""Covered call strategy — sell calls against owned stock positions."""

import logging

from bot.analysis import find_option_by_delta
from bot.strategies.base import BaseStrategy, TradeSignal

logger = logging.getLogger(__name__)


class CoveredCallStrategy(BaseStrategy):
    """Automated covered call trading strategy.

    Sells OTM calls against stock positions you already own.
    Generates income on existing holdings.
    """

    def __init__(self, config: dict):
        super().__init__("covered_calls", config)

    def scan_for_entries(
        self, symbol: str, chain_data: dict, underlying_price: float
    ) -> list[TradeSignal]:
        """Scan for covered call opportunities on stocks we own."""
        signals = []
        min_dte = self.config.get("min_dte", 20)
        max_dte = self.config.get("max_dte", 45)
        target_delta = self.config.get("short_delta", 0.30)

        # Only trade on configured tickers
        allowed_tickers = self.config.get("tickers", [])
        if allowed_tickers and symbol not in allowed_tickers:
            return signals

        calls = chain_data.get("calls", {})

        for exp_date, exp_calls in calls.items():
            if not exp_calls:
                continue

            dte = exp_calls[0].get("dte", 0)
            if dte < min_dte or dte > max_dte:
                continue

            short_call = find_option_by_delta(exp_calls, target_delta)
            if not short_call:
                continue

            premium = short_call["mid"]
            if premium <= 0.05:
                continue

            # Return on capital for the covered call
            roc = premium / underlying_price if underlying_price > 0 else 0
            annualized_roc = roc * (365 / dte) if dte > 0 else 0

            if annualized_roc < 0.05:  # Min 5% annualized return
                continue

            self.logger.info(
                "Covered call on %s: Sell %s call exp %s | Premium: $%.2f | "
                "ROC: %.2f%% (%.1f%% ann.) | Delta: %.2f",
                symbol, short_call["strike"], exp_date, premium,
                roc * 100, annualized_roc * 100, short_call["delta"],
            )

            from bot.analysis import SpreadAnalysis

            analysis = SpreadAnalysis(
                symbol=symbol,
                strategy="covered_call",
                expiration=exp_date,
                dte=dte,
                short_strike=short_call["strike"],
                long_strike=0,
                credit=premium,
                max_profit=premium,
                max_loss=0,  # Covered by stock
                probability_of_profit=1.0 - abs(short_call["delta"]),
                net_delta=short_call["delta"],
                net_theta=short_call["theta"],
                net_vega=short_call["vega"],
                score=self._score_covered_call(
                    short_call, premium, annualized_roc, dte
                ),
            )

            signals.append(TradeSignal(
                action="open",
                strategy="covered_call",
                symbol=symbol,
                analysis=analysis,
            ))

        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open covered calls for exit conditions."""
        signals = []
        profit_target_pct = self.config.get("profit_target_pct", 0.50)

        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") != "covered_call":
                continue

            entry_credit = pos.get("entry_credit", 0)
            current_value = pos.get("current_value", 0)
            exit_reason = ""

            if entry_credit > 0:
                pnl = entry_credit - current_value
                pnl_pct = pnl / entry_credit if entry_credit > 0 else 0

                # Close at profit target to free up shares for a new call
                if pnl_pct >= profit_target_pct:
                    self.logger.info(
                        "PROFIT TARGET hit on %s covered call: P/L %.1f%%",
                        pos.get("symbol"), pnl_pct * 100,
                    )
                    exit_reason = f"Profit target reached ({pnl_pct:.1%})"

            # Close if approaching expiration to avoid assignment.
            if not exit_reason and pos.get("dte_remaining", 999) <= 3:
                dte = pos.get("dte_remaining", 999)
                self.logger.info(
                    "DTE EXIT on %s covered call: %d DTE",
                    pos.get("symbol"), dte,
                )
                exit_reason = f"Approaching expiration ({dte} DTE)"

            if exit_reason:
                signals.append(TradeSignal(
                    action="close",
                    strategy="covered_call",
                    symbol=pos.get("symbol", ""),
                    position_id=pos.get("position_id"),
                    reason=exit_reason,
                ))

        return signals

    @staticmethod
    def _score_covered_call(
        call_option: dict, premium: float, annualized_roc: float, dte: int
    ) -> float:
        """Score a covered call from 0-100."""
        score = 0.0

        # Annualized ROC (35% weight) — higher is better, cap at 30%
        score += min(annualized_roc / 0.30, 1.0) * 35

        # Probability OTM (30% weight)
        prob_otm = 1.0 - abs(call_option.get("delta", 0.3))
        score += prob_otm * 30

        # Liquidity (15% weight)
        oi = call_option.get("open_interest", 0)
        vol = call_option.get("volume", 0)
        liquidity = min(oi / 500, 1.0) * 0.5 + min(vol / 100, 1.0) * 0.5
        score += liquidity * 15

        # DTE sweet spot 25-40 (10% weight)
        if 20 <= dte <= 45:
            score += 10
        elif 14 <= dte < 20 or 45 < dte <= 55:
            score += 5

        # Bid-ask tightness (10% weight)
        ba_spread = call_option.get("ask", 0) - call_option.get("bid", 0)
        tightness = max(0, 1.0 - ba_spread / 0.30)
        score += tightness * 10

        return round(min(score, 100), 1)
