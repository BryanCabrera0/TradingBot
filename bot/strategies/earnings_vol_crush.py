"""Sell pre-earnings premium to capture post-event IV crush."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bot.analysis import analyze_iron_condor, find_option_by_delta, find_spread_wing
from bot.data_store import load_json
from bot.earnings_calendar import EarningsCalendar
from bot.number_utils import safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext


class EarningsVolCrushStrategy(BaseStrategy):
    """Short-dated iron-condor earnings strategy."""

    def __init__(self, config: dict):
        super().__init__("earnings_vol_crush", config)
        self.earnings_calendar = EarningsCalendar()
        self.moves_path = Path(
            self.config.get("earnings_moves_file", "bot/data/earnings_moves.json")
        )

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

        move_profile = self._earnings_move_profile(symbol)
        boost_score = 0.0
        if move_profile:
            total_events = max(1, int(move_profile.get("total_events", 0) or 0))
            times_exceeded = int(move_profile.get("times_exceeded_implied", 0) or 0)
            exceeded_ratio = times_exceeded / total_events
            if exceeded_ratio > 0.50:
                return []
            avg_move = float(move_profile.get("avg_earnings_move_pct", 0.0) or 0.0)
            implied_move = float(move_profile.get("implied_move_pct", 0.0) or 0.0)
            if implied_move > 0 and avg_move < (0.70 * implied_move):
                boost_score = 8.0

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
            long_put = (
                find_spread_wing(exp_puts, short_put["strike"], wing_width, "lower")
                if short_put
                else None
            )
            short_call = find_option_by_delta(exp_calls, target_delta)
            long_call = (
                find_spread_wing(exp_calls, short_call["strike"], wing_width, "higher")
                if short_call
                else None
            )
            if not short_put or not long_put or not short_call or not long_call:
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
            analysis.score = round(min(100.0, analysis.score + 8.0 + boost_score), 1)

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
                        "earnings_move_profile": move_profile,
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
        signals.sort(
            key=lambda item: item.analysis.score if item.analysis else 0.0, reverse=True
        )
        return signals

    def _earnings_move_profile(self, symbol: str) -> dict:
        payload = load_json(self.moves_path, {})
        if not isinstance(payload, dict):
            return {}
        row = payload.get(str(symbol).upper(), {})
        return row if isinstance(row, dict) else {}

    def check_exits(self, positions: list, market_client) -> list[TradeSignal]:
        out: list[TradeSignal] = []
        adaptive_targets = bool(self.config.get("adaptive_targets", True))
        trailing_enabled = bool(self.config.get("trailing_stop_enabled", False))
        trail_activation = float(self.config.get("trailing_stop_activation_pct", 0.25))
        trail_floor = float(self.config.get("trailing_stop_floor_pct", 0.10))
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
            target_pct = self._profit_target_for_dte(dte) if adaptive_targets else 0.35
            details = (
                pos.get("details", {}) if isinstance(pos.get("details"), dict) else {}
            )
            trailing_high = float(
                pos.get("trailing_stop_high", details.get("trailing_stop_high", 0.0))
                or 0.0
            )
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
            if pnl_pct >= target_pct or trailing_triggered or dte <= 0:
                out.append(
                    TradeSignal(
                        action="close",
                        strategy="earnings_vol_crush",
                        symbol=str(pos.get("symbol", "")),
                        position_id=pos.get("position_id"),
                        reason=(
                            "Trailing stop"
                            if trailing_triggered
                            else (
                                "Post-earnings crush capture"
                                if pnl_pct >= target_pct
                                else "Event complete"
                            )
                        ),
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
