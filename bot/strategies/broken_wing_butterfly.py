"""Broken-wing butterfly strategy with directional bias."""

from __future__ import annotations

from typing import Optional

from bot.analysis import SpreadAnalysis, find_option_by_delta, find_spread_wing
from bot.number_utils import safe_int
from bot.strategies.base import BaseStrategy, TradeSignal
from bot.technicals import TechnicalContext


class BrokenWingButterflyStrategy(BaseStrategy):
    """Directional broken-wing butterfly entry generator."""

    def __init__(self, config: dict):
        super().__init__("broken_wing_butterfly", config)

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
        calls = chain_data.get("calls", {})
        puts = chain_data.get("puts", {})
        if not calls or not puts:
            return []

        regime = str((market_context or {}).get("regime", "")).upper()
        bullish = regime in {"BULL_TREND", "MEAN_REVERSION", "LOW_VOL_GRIND"}

        min_dte = int(self.config.get("min_dte", 20))
        max_dte = int(self.config.get("max_dte", 45))
        target_delta = float(self.config.get("short_delta", 0.30))
        near_wing = float(self.config.get("near_wing_width", 5.0))
        far_wing = float(self.config.get("far_wing_width", 10.0))
        min_credit = float(self.config.get("min_credit", 0.10))

        signals: list[TradeSignal] = []
        expirations = sorted(set(calls.keys()) & set(puts.keys()))
        for exp in expirations:
            options = puts.get(exp, []) if bullish else calls.get(exp, [])
            if not options:
                continue
            dte = int(options[0].get("dte", 0) or 0)
            if dte < min_dte or dte > max_dte:
                continue
            short = find_option_by_delta(options, target_delta)
            if not short:
                continue

            # Near wing on the low-risk side, far wing on the risk side.
            if bullish:
                near = find_spread_wing(options, short["strike"], near_wing, "higher")
                far = find_spread_wing(options, short["strike"], far_wing, "lower")
            else:
                near = find_spread_wing(options, short["strike"], near_wing, "lower")
                far = find_spread_wing(options, short["strike"], far_wing, "higher")
            if not near or not far:
                continue

            credit = float(short.get("mid", 0.0) or 0.0) * 2.0 - float(near.get("mid", 0.0) or 0.0) - float(far.get("mid", 0.0) or 0.0)
            if credit < min_credit:
                continue

            risk_side = abs(float(far.get("strike", 0.0)) - float(short.get("strike", 0.0)))
            max_loss = max(risk_side - credit, credit)
            score = 60.0
            score += 10.0 if bullish else 8.0
            score += min(15.0, credit * 10.0)

            analysis = SpreadAnalysis(
                symbol=symbol,
                strategy="broken_wing_butterfly",
                expiration=exp,
                dte=dte,
                short_strike=float(short.get("strike", 0.0)),
                long_strike=float(far.get("strike", 0.0)),
                credit=round(credit, 4),
                max_loss=round(max_loss, 4),
                max_profit=round(max(credit + near_wing, credit), 4),
                probability_of_profit=0.58,
                score=round(min(100.0, score), 1),
            )
            signals.append(
                TradeSignal(
                    action="open",
                    strategy="broken_wing_butterfly",
                    symbol=symbol,
                    analysis=analysis,
                    metadata={
                        "regime": regime,
                        "position_details": {
                            "expiration": exp,
                            "short_strike": float(short.get("strike", 0.0)),
                            "near_wing_strike": float(near.get("strike", 0.0)),
                            "far_wing_strike": float(far.get("strike", 0.0)),
                            "direction": "bullish" if bullish else "bearish",
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
            if pos.get("strategy") != "broken_wing_butterfly":
                continue
            entry_credit = float(pos.get("entry_credit", 0.0) or 0.0)
            current_value = float(pos.get("current_value", 0.0) or 0.0)
            dte = safe_int(pos.get("dte_remaining"), 999)
            if entry_credit <= 0:
                continue
            pnl_pct = (entry_credit - current_value) / entry_credit
            if pnl_pct >= 0.40 or dte <= 7:
                out.append(
                    TradeSignal(
                        action="close",
                        strategy="broken_wing_butterfly",
                        symbol=str(pos.get("symbol", "")),
                        position_id=pos.get("position_id"),
                        reason="Profit target" if pnl_pct >= 0.40 else "DTE threshold",
                        quantity=max(1, int(pos.get("quantity", 1))),
                    )
                )
        return out
