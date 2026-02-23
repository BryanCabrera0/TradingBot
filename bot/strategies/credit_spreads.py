"""Credit spread strategy — bull put spreads and bear call spreads."""

import logging
from typing import Optional

from bot.analysis import (
    SpreadAnalysis,
    analyze_credit_spread,
    find_option_by_delta,
    find_spread_wing,
)
from bot.strategies.base import BaseStrategy, TradeSignal

logger = logging.getLogger(__name__)


class CreditSpreadStrategy(BaseStrategy):
    """Automated credit spread trading strategy.

    Sells OTM credit spreads (bull put or bear call) and collects premium.
    Exits at profit target or stop loss.
    """

    def __init__(self, config: dict):
        super().__init__("credit_spreads", config)

    def scan_for_entries(
        self, symbol: str, chain_data: dict, underlying_price: float
    ) -> list[TradeSignal]:
        """Scan for credit spread opportunities on a symbol."""
        signals = []
        direction = self.config.get("direction", "both")
        min_dte = self.config.get("min_dte", 20)
        max_dte = self.config.get("max_dte", 45)
        target_delta = self.config.get("short_delta", 0.30)
        spread_width = self.config.get("spread_width", 5)

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})

        for exp_date in puts:
            # Bull put spreads
            if direction in ("both", "bull_put"):
                put_signals = self._scan_put_spreads(
                    symbol, underlying_price, puts.get(exp_date, []),
                    exp_date, min_dte, max_dte, target_delta, spread_width,
                )
                signals.extend(put_signals)

        for exp_date in calls:
            # Bear call spreads
            if direction in ("both", "bear_call"):
                call_signals = self._scan_call_spreads(
                    symbol, underlying_price, calls.get(exp_date, []),
                    exp_date, min_dte, max_dte, target_delta, spread_width,
                )
                signals.extend(call_signals)

        # Sort by score descending
        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        return signals

    def _scan_put_spreads(
        self,
        symbol: str,
        underlying_price: float,
        puts: list,
        exp_date: str,
        min_dte: int,
        max_dte: int,
        target_delta: float,
        spread_width: float,
    ) -> list[TradeSignal]:
        """Find bull put spread opportunities."""
        signals = []
        if not puts:
            return signals

        dte = puts[0].get("dte", 0) if puts else 0
        if dte < min_dte or dte > max_dte:
            return signals

        # Find short put near target delta
        short_put = find_option_by_delta(puts, target_delta)
        if not short_put:
            return signals

        # Find long put (lower strike)
        long_put = find_spread_wing(puts, short_put["strike"], spread_width, "lower")
        if not long_put:
            return signals

        analysis = analyze_credit_spread(
            underlying_price, short_put, long_put, "PUT"
        )
        analysis.symbol = symbol

        if self.meets_minimum_quality(analysis):
            self.logger.info(
                "Bull put spread on %s: %s/%s exp %s | Credit: $%.2f | "
                "POP: %.1f%% | Score: %.1f",
                symbol, short_put["strike"], long_put["strike"],
                exp_date, analysis.credit,
                analysis.probability_of_profit * 100, analysis.score,
            )
            signals.append(TradeSignal(
                action="open",
                strategy="bull_put_spread",
                symbol=symbol,
                analysis=analysis,
            ))

        return signals

    def _scan_call_spreads(
        self,
        symbol: str,
        underlying_price: float,
        calls: list,
        exp_date: str,
        min_dte: int,
        max_dte: int,
        target_delta: float,
        spread_width: float,
    ) -> list[TradeSignal]:
        """Find bear call spread opportunities."""
        signals = []
        if not calls:
            return signals

        dte = calls[0].get("dte", 0) if calls else 0
        if dte < min_dte or dte > max_dte:
            return signals

        short_call = find_option_by_delta(calls, target_delta)
        if not short_call:
            return signals

        long_call = find_spread_wing(calls, short_call["strike"], spread_width, "higher")
        if not long_call:
            return signals

        analysis = analyze_credit_spread(
            underlying_price, short_call, long_call, "CALL"
        )
        analysis.symbol = symbol

        if self.meets_minimum_quality(analysis):
            self.logger.info(
                "Bear call spread on %s: %s/%s exp %s | Credit: $%.2f | "
                "POP: %.1f%% | Score: %.1f",
                symbol, short_call["strike"], long_call["strike"],
                exp_date, analysis.credit,
                analysis.probability_of_profit * 100, analysis.score,
            )
            signals.append(TradeSignal(
                action="open",
                strategy="bear_call_spread",
                symbol=symbol,
                analysis=analysis,
            ))

        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open credit spreads for exit conditions."""
        signals = []
        profit_target_pct = self.config.get("profit_target_pct", 0.50)
        stop_loss_pct = self.config.get("stop_loss_pct", 2.0)

        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") not in ("bull_put_spread", "bear_call_spread"):
                continue

            entry_credit = pos.get("entry_credit", 0)
            current_value = pos.get("current_value", 0)
            exit_reason = ""

            if entry_credit > 0:
                # Current P/L per contract
                pnl = entry_credit - current_value
                pnl_pct = pnl / entry_credit if entry_credit > 0 else 0

                # Profit target: close if we've captured X% of max profit
                if pnl_pct >= profit_target_pct:
                    self.logger.info(
                        "PROFIT TARGET hit on %s: P/L %.1f%% >= target %.1f%%",
                        pos.get("symbol"), pnl_pct * 100, profit_target_pct * 100,
                    )
                    exit_reason = f"Profit target reached ({pnl_pct:.1%})"
                # Stop loss: close if loss exceeds X times the credit received
                elif pnl < 0 and abs(pnl) >= entry_credit * stop_loss_pct:
                    self.logger.warning(
                        "STOP LOSS hit on %s: Loss $%.2f >= %.1fx credit $%.2f",
                        pos.get("symbol"), abs(pnl), stop_loss_pct, entry_credit,
                    )
                    exit_reason = f"Stop loss triggered (loss {abs(pnl):.2f})"

            # DTE exit: close if approaching expiration
            if not exit_reason and pos.get("dte_remaining", 999) <= 5:
                dte = pos.get("dte_remaining", 999)
                self.logger.info(
                    "DTE EXIT on %s: %d days to expiration", pos.get("symbol"), dte
                )
                exit_reason = f"Approaching expiration ({dte} DTE)"

            if exit_reason:
                signals.append(TradeSignal(
                    action="close",
                    strategy=pos["strategy"],
                    symbol=pos.get("symbol", ""),
                    position_id=pos.get("position_id"),
                    reason=exit_reason,
                ))

        return signals
