"""Sell pre-earnings premium to capture post-event IV crush."""

from __future__ import annotations

from typing import Optional

from bot.analysis import analyze_iron_condor, find_option_by_delta, find_spread_wing
from bot.earnings_calendar import EarningsCalendar
from bot.number_utils import safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext


class EarningsVolCrushStrategy(BaseStrategy):
    """Short-dated iron-condor earnings strategy."""

    def __init__(self, config: dict):
        super().__init__("earnings_vol_crush", config)
        self.earnings_calendar = EarningsCalendar()

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
        if iv_rank < float(self.config.get("min_iv_rank", 75.0)):
            return []

        has_earnings, earnings_date = self.earnings_calendar.earnings_within_window(
            symbol,
            (market_context or {}).get("max_expiration", ""),
        )
        if not has_earnings:
            return []

        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})
        if not calls or not puts:
            return []

        min_dte = int(self.config.get("min_dte", 1))
        max_dte = int(self.config.get("max_dte", 3))
        wing_width = float(self.config.get("wing_width", 10.0))
        target_delta = 0.12

        signals: list[TradeSignal] = []
        for exp in sorted(set(calls.keys()) & set(puts.keys())):
            exp_calls = calls.get(exp, [])
            exp_puts = puts.get(exp, [])
            if not exp_calls or not exp_puts:
                continue
            dte = int(exp_calls[0].get("dte", 0) or 0)
            if dte < min_dte or dte > max_dte:
                continue

            short_put = find_option_by_delta(exp_puts, target_delta)
            long_put = find_spread_wing(exp_puts, short_put["strike"], wing_width, "lower") if short_put else None
            short_call = find_option_by_delta(exp_calls, target_delta)
            long_call = find_spread_wing(exp_calls, short_call["strike"], wing_width, "higher") if short_call else None
            if None in (short_put, long_put, short_call, long_call):
                continue
            if float(short_put["strike"]) >= float(short_call["strike"]):
                continue

            analysis = analyze_iron_condor(
                underlying_price,
                short_put,
                long_put,
                short_call,
                long_call,
            )
            analysis.symbol = symbol
            analysis.strategy = "earnings_vol_crush"
            analysis.score = round(min(100.0, analysis.score + 8.0), 1)

            signals.append(
                TradeSignal(
                    action="open",
                    strategy="earnings_vol_crush",
                    symbol=symbol,
                    analysis=analysis,
                    size_multiplier=0.50,  # capped risk profile
                    metadata={
                        "iv_rank": iv_rank,
                        "earnings_date": earnings_date,
                        "position_details": {
                            "expiration": exp,
                            "put_short_strike": analysis.put_short_strike,
                            "put_long_strike": analysis.put_long_strike,
                            "call_short_strike": analysis.call_short_strike,
                            "call_long_strike": analysis.call_long_strike,
                        },
                    },
                )
            )
        signals.sort(key=lambda item: item.analysis.score if item.analysis else 0.0, reverse=True)
        return signals

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        out: list[TradeSignal] = []
        for pos in positions:
            if str(pos.get("status", "open")).lower() != "open":
                continue
            if pos.get("strategy") != "earnings_vol_crush":
                continue
            entry_credit = float(pos.get("entry_credit", 0.0) or 0.0)
            current = float(pos.get("current_value", 0.0) or 0.0)
            dte = safe_int(pos.get("dte_remaining"), 999)
            if entry_credit <= 0:
                continue
            pnl_pct = (entry_credit - current) / entry_credit
            if pnl_pct >= 0.35 or dte <= 0:
                out.append(
                    TradeSignal(
                        action="close",
                        strategy="earnings_vol_crush",
                        symbol=str(pos.get("symbol", "")),
                        position_id=pos.get("position_id"),
                        reason="Post-earnings crush capture" if pnl_pct >= 0.35 else "Event complete",
                        quantity=max(1, int(pos.get("quantity", 1))),
                    )
                )
        return out
