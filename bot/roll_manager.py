"""Position rolling decision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bot.number_utils import safe_int
from bot.strategies.base import TradeSignal


@dataclass
class RollDecision:
    should_roll: bool
    reason: str = ""
    roll_type: str = "none"  # profit | defensive | none
    min_credit_required: float = 0.0


class RollManager:
    """Determine when and how positions should be rolled."""

    def __init__(self, config: dict):
        self.config = config

    def evaluate(self, position: dict, *, regime: str = "normal") -> RollDecision:
        if not bool(self.config.get("enabled", False)):
            return RollDecision(False, "rolling disabled")

        dte = safe_int(position.get("dte_remaining"), 999)
        entry_credit = float(position.get("entry_credit", 0.0) or 0.0)
        current_value = float(position.get("current_value", 0.0) or 0.0)
        if entry_credit <= 0:
            return RollDecision(False, "missing entry credit")

        roll_count = int((position.get("details", {}) or {}).get("roll_count", 0) or 0)
        if roll_count >= int(self.config.get("max_rolls_per_position", 2)):
            return RollDecision(False, "max rolls reached")

        min_dte = int(self.config.get("min_dte_trigger", 7))
        min_credit = float(self.config.get("min_credit_for_roll", 0.15))
        pnl_pct = (entry_credit - current_value) / entry_credit

        if dte <= min_dte and pnl_pct > 0:
            return RollDecision(
                True,
                reason=f"{dte} DTE with unrealized profit {pnl_pct:.1%}",
                roll_type="profit",
                min_credit_required=min_credit,
            )

        if bool(self.config.get("allow_defensive_rolls", True)) and dte <= max(min_dte + 3, 10):
            if pnl_pct < -0.30 or regime.upper() in {"CRASH/CRISIS", "BEAR_TREND"}:
                return RollDecision(
                    True,
                    reason=f"Defensive roll candidate at {dte} DTE",
                    roll_type="defensive",
                    min_credit_required=min_credit,
                )

        return RollDecision(False, "no roll condition met")

    @staticmethod
    def annotate_roll_metadata(source_position: dict, target_signal: TradeSignal) -> None:
        details = source_position.get("details", {}) if isinstance(source_position.get("details"), dict) else {}
        prev_rolls = int(details.get("roll_count", 0) or 0)
        target_signal.metadata.setdefault("position_details", {})
        target_signal.metadata["position_details"]["rolled_from"] = source_position.get("position_id")
        target_signal.metadata["position_details"]["roll_count"] = prev_rolls + 1
