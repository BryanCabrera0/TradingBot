"""Options flow intelligence derived from already-fetched chain snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bot.number_utils import safe_float


@dataclass
class FlowContext:
    """Symbol-level options flow diagnostics for strategy/LLM context."""

    symbol: str
    directional_bias: str = "neutral"  # bullish | bearish | neutral
    unusual_activity_flag: bool = False
    institutional_flow_direction: str = "neutral"
    put_call_volume_ratio: float = 1.0
    sweep_score: float = 0.0
    open_interest_change: float = 0.0
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "directional_bias": self.directional_bias,
            "unusual_activity_flag": self.unusual_activity_flag,
            "institutional_flow_direction": self.institutional_flow_direction,
            "put_call_volume_ratio": self.put_call_volume_ratio,
            "sweep_score": self.sweep_score,
            "open_interest_change": self.open_interest_change,
            "metrics": self.metrics,
        }


class OptionsFlowAnalyzer:
    """Analyze unusual options activity from Schwab option chain payloads."""

    def __init__(self, *, unusual_volume_multiple: float = 5.0):
        self.unusual_volume_multiple = max(1.0, float(unusual_volume_multiple))

    def analyze(
        self,
        *,
        symbol: str,
        chain_data: dict,
        previous_chain_data: Optional[dict] = None,
    ) -> FlowContext:
        calls = chain_data.get("calls", {}) if isinstance(chain_data, dict) else {}
        puts = chain_data.get("puts", {}) if isinstance(chain_data, dict) else {}

        call_volumes = _collect_metric(calls, "volume")
        put_volumes = _collect_metric(puts, "volume")
        call_oi = _collect_metric(calls, "open_interest")
        put_oi = _collect_metric(puts, "open_interest")

        call_total = float(np.sum(call_volumes)) if call_volumes else 0.0
        put_total = float(np.sum(put_volumes)) if put_volumes else 0.0
        put_call_ratio = put_total / max(call_total, 1.0)

        unusual_flag = self._detect_unusual_volume(call_volumes, put_volumes)
        sweep_score = self._sweep_score(calls, puts)
        oi_change = self._open_interest_change(
            current_calls=call_oi,
            current_puts=put_oi,
            previous=previous_chain_data,
        )

        directional_bias = "neutral"
        if put_call_ratio > 1.25:
            directional_bias = "bearish"
        elif put_call_ratio < 0.80:
            directional_bias = "bullish"

        if unusual_flag and sweep_score >= 0.60:
            institutional = (
                directional_bias if directional_bias != "neutral" else "two_way"
            )
        elif sweep_score >= 0.50:
            institutional = directional_bias
        else:
            institutional = "neutral"

        return FlowContext(
            symbol=symbol.upper(),
            directional_bias=directional_bias,
            unusual_activity_flag=bool(unusual_flag),
            institutional_flow_direction=institutional,
            put_call_volume_ratio=round(put_call_ratio, 4),
            sweep_score=round(sweep_score, 4),
            open_interest_change=round(oi_change, 4),
            metrics={
                "call_volume_total": round(call_total, 2),
                "put_volume_total": round(put_total, 2),
                "call_open_interest_total": round(
                    float(np.sum(call_oi)) if call_oi else 0.0, 2
                ),
                "put_open_interest_total": round(
                    float(np.sum(put_oi)) if put_oi else 0.0, 2
                ),
            },
        )

    def _detect_unusual_volume(
        self, call_volumes: list[float], put_volumes: list[float]
    ) -> bool:
        combined = [volume for volume in (call_volumes + put_volumes) if volume > 0]
        if len(combined) < 4:
            return False
        avg = float(np.mean(combined))
        peak = float(np.max(combined))
        return peak >= (avg * self.unusual_volume_multiple)

    @staticmethod
    def _sweep_score(calls: dict, puts: dict) -> float:
        # Approximate "sweep urgency" from concentration of extreme-volume contracts.
        volumes = []
        for side in (calls, puts):
            for options in (side or {}).values():
                for row in options or []:
                    volume = safe_float(row.get("volume"), 0.0)
                    oi = safe_float(row.get("open_interest"), 0.0)
                    if volume <= 0:
                        continue
                    # Contracts trading volume above OI are often indicative of fresh urgency.
                    ratio = volume / max(oi, 1.0)
                    volumes.append(min(2.0, ratio))
        if not volumes:
            return 0.0
        top = sorted(volumes, reverse=True)[:10]
        return float(np.mean(top)) / 2.0

    @staticmethod
    def _open_interest_change(
        *,
        current_calls: list[float],
        current_puts: list[float],
        previous: Optional[dict],
    ) -> float:
        if not isinstance(previous, dict):
            return 0.0
        prev_calls = _collect_metric(previous.get("calls", {}), "open_interest")
        prev_puts = _collect_metric(previous.get("puts", {}), "open_interest")
        prev_total = (
            float(np.sum(prev_calls) + np.sum(prev_puts))
            if prev_calls or prev_puts
            else 0.0
        )
        curr_total = (
            float(np.sum(current_calls) + np.sum(current_puts))
            if current_calls or current_puts
            else 0.0
        )
        if prev_total <= 0:
            return 0.0
        return (curr_total - prev_total) / prev_total


def _collect_metric(exp_map: dict, field: str) -> list[float]:
    out: list[float] = []
    if not isinstance(exp_map, dict):
        return out
    for options in exp_map.values():
        for row in options or []:
            value = safe_float(row.get(field), 0.0)
            if value >= 0:
                out.append(value)
    return out
