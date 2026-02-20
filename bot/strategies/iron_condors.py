"""Iron condor strategy — sell OTM put spread + OTM call spread."""

import logging

from bot.analysis import (
    analyze_iron_condor,
    find_option_by_delta,
    find_spread_wing,
)
from bot.strategies.base import BaseStrategy, TradeSignal

logger = logging.getLogger(__name__)


class IronCondorStrategy(BaseStrategy):
    """Automated iron condor trading strategy.

    Sells an OTM put spread and an OTM call spread simultaneously,
    profiting when the underlying stays within a range.
    """

    def __init__(self, config: dict):
        super().__init__("iron_condors", config)

    def scan_for_entries(
        self, symbol: str, chain_data: dict, underlying_price: float
    ) -> list[TradeSignal]:
        """Scan for iron condor opportunities."""
        signals = []
        min_dte = self.config.get("min_dte", 25)
        max_dte = self.config.get("max_dte", 50)
        target_delta = self.config.get("short_delta", 0.16)
        spread_width = self.config.get("spread_width", 5)

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})

        # Find matching expirations in both calls and puts
        common_exps = set(calls.keys()) & set(puts.keys())

        for exp_date in common_exps:
            exp_puts = puts[exp_date]
            exp_calls = calls[exp_date]

            if not exp_puts or not exp_calls:
                continue

            dte = exp_puts[0].get("dte", 0)
            if dte < min_dte or dte > max_dte:
                continue

            # Find short put near target delta
            short_put = find_option_by_delta(exp_puts, target_delta)
            if not short_put:
                continue

            # Find long put (lower strike)
            long_put = find_spread_wing(
                exp_puts, short_put["strike"], spread_width, "lower"
            )
            if not long_put:
                continue

            # Find short call near target delta
            short_call = find_option_by_delta(exp_calls, target_delta)
            if not short_call:
                continue

            # Find long call (higher strike)
            long_call = find_spread_wing(
                exp_calls, short_call["strike"], spread_width, "higher"
            )
            if not long_call:
                continue

            # Make sure strikes don't overlap
            if short_put["strike"] >= short_call["strike"]:
                continue

            analysis = analyze_iron_condor(
                underlying_price, short_put, long_put, short_call, long_call
            )
            analysis.symbol = symbol

            if self.meets_minimum_quality(analysis):
                self.logger.info(
                    "Iron condor on %s: P %s/%s | C %s/%s exp %s | "
                    "Credit: $%.2f | POP: %.1f%% | Score: %.1f",
                    symbol,
                    long_put["strike"], short_put["strike"],
                    short_call["strike"], long_call["strike"],
                    exp_date, analysis.credit,
                    analysis.probability_of_profit * 100, analysis.score,
                )
                signals.append(TradeSignal(
                    action="open",
                    strategy="iron_condor",
                    symbol=symbol,
                    analysis=analysis,
                ))

        signals.sort(key=lambda s: s.analysis.score if s.analysis else 0, reverse=True)
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        """Check open iron condors for exit conditions."""
        signals = []
        profit_target_pct = self.config.get("profit_target_pct", 0.50)
        stop_loss_pct = self.config.get("stop_loss_pct", 2.0)

        for pos in positions:
            if pos.get("strategy") != "iron_condor":
                continue

            entry_credit = pos.get("entry_credit", 0)
            current_value = pos.get("current_value", 0)

            if entry_credit <= 0:
                continue

            pnl = entry_credit - current_value
            pnl_pct = pnl / entry_credit if entry_credit > 0 else 0

            if pnl_pct >= profit_target_pct:
                self.logger.info(
                    "PROFIT TARGET hit on %s condor: P/L %.1f%%",
                    pos.get("symbol"), pnl_pct * 100,
                )
                signals.append(TradeSignal(
                    action="close",
                    strategy="iron_condor",
                    symbol=pos.get("symbol", ""),
                    position_id=pos.get("position_id"),
                    reason=f"Profit target reached ({pnl_pct:.1%})",
                ))

            elif pnl < 0 and abs(pnl) >= entry_credit * stop_loss_pct:
                self.logger.warning(
                    "STOP LOSS hit on %s condor: Loss $%.2f",
                    pos.get("symbol"), abs(pnl),
                )
                signals.append(TradeSignal(
                    action="close",
                    strategy="iron_condor",
                    symbol=pos.get("symbol", ""),
                    position_id=pos.get("position_id"),
                    reason=f"Stop loss triggered (loss {abs(pnl):.2f})",
                ))

            # Use elif to avoid duplicate close signals for the same position.
            elif pos.get("dte_remaining", 999) <= 5:
                dte = pos.get("dte_remaining", 999)
                self.logger.info(
                    "DTE EXIT on %s condor: %d days to expiration",
                    pos.get("symbol"), dte,
                )
                signals.append(TradeSignal(
                    action="close",
                    strategy="iron_condor",
                    symbol=pos.get("symbol", ""),
                    position_id=pos.get("position_id"),
                    reason=f"Approaching expiration ({dte} DTE)",
                ))

        return signals
