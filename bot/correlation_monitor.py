"""Cross-asset rolling-correlation monitor for portfolio risk overlays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from bot.number_utils import safe_float

NORMAL = "normal"
STRESSED = "stressed"
CRISIS = "crisis"


@dataclass
class CorrelationState:
    """Computed cross-asset correlation regime and diagnostics."""

    correlation_regime: str = NORMAL
    correlations: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    lookback_days: int = 20

    def to_dict(self) -> dict:
        return {
            "correlation_regime": self.correlation_regime,
            "correlations": dict(self.correlations),
            "flags": dict(self.flags),
            "lookback_days": int(self.lookback_days),
        }


class CrossAssetCorrelationMonitor:
    """Track rolling cross-asset correlations and classify stress regime."""

    def __init__(
        self,
        *,
        get_price_history: Optional[Callable[[str, int], list[dict]]] = None,
        lookback_days: int = 20,
        crisis_threshold: float = 0.95,
        stress_threshold: float = 0.85,
    ):
        self.get_price_history = get_price_history
        self.lookback_days = max(10, int(lookback_days))
        self.crisis_threshold = max(0.0, min(1.0, float(crisis_threshold)))
        self.stress_threshold = max(0.0, min(1.0, float(stress_threshold)))

    def get_correlation_state(self) -> dict:
        """Return cross-asset correlation regime and diagnostic flags."""
        pairs = {
            "SPY_QQQ": ("SPY", "QQQ"),
            "SPY_IWM": ("SPY", "IWM"),
            "SPY_VIX": ("SPY", "^VIX"),
            "HYG_SPY": ("HYG", "SPY"),
            "TLT_SPY": ("TLT", "SPY"),
            "GLD_SPY": ("GLD", "SPY"),
        }
        values: dict[str, float] = {}
        for key, (left, right) in pairs.items():
            values[key] = self._returns_correlation(left, right)

        spy_vix_corr = safe_float(values.get("SPY_VIX"), 0.0)
        spy_qqq = abs(safe_float(values.get("SPY_QQQ"), 0.0))
        spy_iwm = abs(safe_float(values.get("SPY_IWM"), 0.0))
        hyg_spy = safe_float(values.get("HYG_SPY"), 0.0)

        flags = {
            "spy_vix_positive": spy_vix_corr > 0.0,
            "hyg_spy_decoupled": abs(hyg_spy) < 0.20,
            "equity_corr_spike": max(spy_qqq, spy_iwm) >= self.stress_threshold,
            "systemic_equity_corr": min(spy_qqq, spy_iwm) >= self.crisis_threshold,
        }

        regime = NORMAL
        if (
            (spy_vix_corr > 0.20)
            or flags["systemic_equity_corr"]
            or abs(hyg_spy) < 0.10
        ):
            regime = CRISIS
        elif (
            flags["equity_corr_spike"]
            or flags["hyg_spy_decoupled"]
            or spy_vix_corr > -0.10
        ):
            regime = STRESSED

        state = CorrelationState(
            correlation_regime=regime,
            correlations={k: round(v, 4) for k, v in values.items()},
            flags=flags,
            lookback_days=self.lookback_days,
        )
        return state.to_dict()

    def _returns_correlation(self, left_symbol: str, right_symbol: str) -> float:
        left = self._returns(left_symbol)
        right = self._returns(right_symbol)
        if left.size < self.lookback_days or right.size < self.lookback_days:
            return 0.0
        size = min(left.size, right.size)
        if size < self.lookback_days:
            return 0.0
        corr = float(np.corrcoef(left[-size:], right[-size:])[0, 1])
        if np.isnan(corr):
            return 0.0
        return corr

    def _returns(self, symbol: str) -> np.ndarray:
        if self.get_price_history is None:
            return np.array([], dtype=float)
        try:
            rows = self.get_price_history(symbol, self.lookback_days + 10)
        except Exception:
            return np.array([], dtype=float)
        closes = [
            safe_float(row.get("close"), 0.0)
            for row in (rows or [])
            if isinstance(row, dict)
        ]
        closes = [value for value in closes if value > 0]
        if len(closes) < (self.lookback_days + 1):
            return np.array([], dtype=float)
        arr = np.array(closes[-(self.lookback_days + 1) :], dtype=float)
        rets = np.diff(arr) / arr[:-1]
        return rets[np.isfinite(rets)]
