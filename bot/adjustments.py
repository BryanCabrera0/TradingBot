"""Multi-leg adjustment decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bot.number_utils import safe_int
from bot.strategies.base import TradeSignal


@dataclass
class AdjustmentPlan:
    action: str  # none | add_wing | roll_tested_side | add_hedge
    reason: str
    score: float = 0.0


class AdjustmentEngine:
    """Create adjustment plans when short strikes are tested."""

    def __init__(self, config: dict):
        self.config = config

    def evaluate(
        self,
        *,
        position: dict,
        regime: str,
        iv_change_since_entry: float = 0.0,
    ) -> AdjustmentPlan:
        if not bool(self.config.get("enabled", False)):
            return AdjustmentPlan(action="none", reason="disabled", score=0.0)

        if str(position.get("status", "open")).lower() != "open":
            return AdjustmentPlan(action="none", reason="position not open", score=0.0)

        dte = safe_int(position.get("dte_remaining"), 999)
        min_dte = int(self.config.get("min_dte_remaining", 7))
        if dte < min_dte:
            return AdjustmentPlan(action="none", reason="too close to expiry", score=0.0)

        if int((position.get("details", {}) or {}).get("adjustment_count", 0) or 0) >= int(
            self.config.get("max_adjustments_per_position", 2)
        ):
            return AdjustmentPlan(action="none", reason="max adjustments reached", score=0.0)

        if not _is_short_strike_tested(position):
            return AdjustmentPlan(action="none", reason="short strike not tested", score=0.0)

        pnl = _pnl_pct(position)
        regime_key = str(regime or "").upper()
        if regime_key in {"CRASH/CRISIS", "HIGH_VOL_CHOP"} or iv_change_since_entry > 0.15:
            return AdjustmentPlan(
                action="add_wing",
                reason="Elevated volatility while strike tested",
                score=0.80,
            )
        if pnl < -0.20:
            return AdjustmentPlan(
                action="roll_tested_side",
                reason="Losing position with tested short strike",
                score=0.75,
            )
        return AdjustmentPlan(
            action="add_hedge",
            reason="Strike tested; adding directional hedge",
            score=0.65,
        )

    def to_signal(self, *, position: dict, plan: AdjustmentPlan) -> Optional[TradeSignal]:
        if plan.action == "none":
            return None
        return TradeSignal(
            action="roll" if plan.action == "roll_tested_side" else "close",
            strategy=str(position.get("strategy", "")),
            symbol=str(position.get("symbol", "")),
            position_id=position.get("position_id"),
            reason=f"Adjustment: {plan.action} ({plan.reason})",
            quantity=max(1, int(position.get("quantity", 1))),
            metadata={"adjustment_plan": plan.action, "adjustment_score": plan.score},
        )


def _pnl_pct(position: dict) -> float:
    entry_credit = float(position.get("entry_credit", 0.0) or 0.0)
    current_value = float(position.get("current_value", 0.0) or 0.0)
    if entry_credit <= 0:
        return 0.0
    return (entry_credit - current_value) / entry_credit


def _is_short_strike_tested(position: dict) -> bool:
    details = position.get("details", {}) if isinstance(position.get("details"), dict) else {}
    underlying = float(position.get("underlying_price", 0.0) or 0.0)
    if underlying <= 0:
        return False
    threshold = 0.01
    short_strikes = []
    for key in ("short_strike", "put_short_strike", "call_short_strike"):
        strike = float(details.get(key, 0.0) or 0.0)
        if strike > 0:
            short_strikes.append(strike)
    if not short_strikes:
        return False
    return min(abs(underlying - strike) / underlying for strike in short_strikes) <= threshold
