"""Volatility-surface analytics for options strategy filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bot.iv_history import IVHistory
from bot.number_utils import safe_float


@dataclass
class VolSurfaceContext:
    """Normalized volatility-surface readings for a symbol."""

    symbol: str
    term_structure_front_back: float = 0.0
    term_structure_regime: str = "flat"
    put_call_skew: float = 0.0
    skew_regime: str = "flat"
    iv_rank: float = 50.0
    iv_percentile: float = 50.0
    realized_vol: float = 0.0
    implied_vol: float = 0.0
    realized_implied_spread: float = 0.0
    vol_of_vol: float = 0.0
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "term_structure_front_back": self.term_structure_front_back,
            "term_structure_regime": self.term_structure_regime,
            "put_call_skew": self.put_call_skew,
            "skew_regime": self.skew_regime,
            "iv_rank": self.iv_rank,
            "iv_percentile": self.iv_percentile,
            "realized_vol": self.realized_vol,
            "implied_vol": self.implied_vol,
            "realized_implied_spread": self.realized_implied_spread,
            "vol_of_vol": self.vol_of_vol,
            "flags": self.flags,
        }


class VolSurfaceAnalyzer:
    """Build a strategy-facing volatility context from chain + price history."""

    def __init__(self, iv_history: Optional[IVHistory] = None):
        self.iv_history = iv_history or IVHistory()

    def analyze(
        self,
        *,
        symbol: str,
        chain_data: dict,
        price_history: Optional[list[dict]] = None,
    ) -> VolSurfaceContext:
        calls = chain_data.get("calls", {}) if isinstance(chain_data, dict) else {}
        puts = chain_data.get("puts", {}) if isinstance(chain_data, dict) else {}
        implied = _average_iv(calls, puts)
        iv_rank = self.iv_history.update_and_rank(symbol, implied) if implied > 0 else 50.0
        iv_percentile = self._iv_percentile(symbol=symbol, value=implied)

        term = self._term_structure(calls, puts)
        skew = self._skew(calls, puts)
        realized = _realized_vol(price_history or [])
        realized_implied = implied - realized
        vol_of_vol = _vol_of_vol(calls, puts)

        flags = {
            "positive_vol_risk_premium": realized < implied if implied > 0 else False,
            "front_richer_than_back": term["spread"] > 0.0,
            "stable_iv": vol_of_vol < 0.20,
        }
        return VolSurfaceContext(
            symbol=symbol.upper(),
            term_structure_front_back=round(term["spread"], 4),
            term_structure_regime=term["regime"],
            put_call_skew=round(skew["spread"], 4),
            skew_regime=skew["regime"],
            iv_rank=round(iv_rank, 2),
            iv_percentile=round(iv_percentile, 2),
            realized_vol=round(realized, 4),
            implied_vol=round(implied, 4),
            realized_implied_spread=round(realized_implied, 4),
            vol_of_vol=round(vol_of_vol, 4),
            flags=flags,
        )

    def _term_structure(self, calls: dict, puts: dict) -> dict:
        points = []
        for side in (calls, puts):
            if not isinstance(side, dict):
                continue
            for options in side.values():
                if not options:
                    continue
                dte = int(safe_float(options[0].get("dte"), 0))
                ivs = [safe_float(item.get("iv"), 0.0) for item in options if safe_float(item.get("iv"), 0.0) > 0]
                if not ivs or dte <= 0:
                    continue
                points.append((dte, float(np.mean(ivs))))

        if len(points) < 2:
            return {"spread": 0.0, "regime": "flat"}

        points.sort(key=lambda row: row[0])
        front = points[0][1]
        back = points[-1][1]
        spread = front - back
        if spread > 2.0:
            regime = "backwardation"
        elif spread < -2.0:
            regime = "contango"
        else:
            regime = "flat"
        return {"spread": spread, "regime": regime}

    def _skew(self, calls: dict, puts: dict) -> dict:
        call_ivs = _collect_delta_bucket_ivs(calls, target_delta=0.25)
        put_ivs = _collect_delta_bucket_ivs(puts, target_delta=0.25)
        if not call_ivs or not put_ivs:
            return {"spread": 0.0, "regime": "flat"}

        call_iv = float(np.mean(call_ivs))
        put_iv = float(np.mean(put_ivs))
        spread = put_iv - call_iv
        if spread > 3.0:
            regime = "put_skew_fear"
        elif spread < -3.0:
            regime = "call_skew_speculation"
        else:
            regime = "flat"
        return {"spread": spread, "regime": regime}

    def _iv_percentile(self, *, symbol: str, value: float) -> float:
        # Percentile is derived as z-score CDF to complement rank.
        state = self.iv_history.path
        try:
            from bot.data_store import load_json

            payload = load_json(state, {"symbols": {}})
            series = payload.get("symbols", {}).get(symbol.upper(), []) if isinstance(payload, dict) else []
            ivs = [safe_float(item.get("iv"), 0.0) for item in series if isinstance(item, dict)]
            ivs = [x for x in ivs if x > 0]
            if len(ivs) < 5:
                return 50.0
            mean = float(np.mean(ivs))
            std = float(np.std(ivs, ddof=1))
            if std <= 0:
                return 50.0
            z = (safe_float(value, mean) - mean) / std
            # Standard normal CDF approximation.
            percentile = 50.0 * (1.0 + float(np.math.erf(z / np.sqrt(2.0))))
            return max(0.0, min(100.0, percentile))
        except Exception:
            return 50.0


def _average_iv(calls: dict, puts: dict) -> float:
    ivs = []
    for side in (calls, puts):
        if not isinstance(side, dict):
            continue
        for options in side.values():
            for option in options or []:
                iv = safe_float(option.get("iv"), 0.0)
                if iv > 0:
                    ivs.append(iv)
    if not ivs:
        return 0.0
    return float(np.mean(ivs))


def _collect_delta_bucket_ivs(exp_map: dict, target_delta: float) -> list[float]:
    out = []
    if not isinstance(exp_map, dict):
        return out
    for options in exp_map.values():
        if not options:
            continue
        best = None
        best_diff = float("inf")
        for option in options:
            delta = abs(safe_float(option.get("delta"), 0.0))
            iv = safe_float(option.get("iv"), 0.0)
            if iv <= 0:
                continue
            diff = abs(delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = iv
        if best is not None:
            out.append(best)
    return out


def _realized_vol(price_history: list[dict]) -> float:
    closes = [safe_float(row.get("close"), 0.0) for row in price_history if isinstance(row, dict)]
    closes = [value for value in closes if value > 0]
    if len(closes) < 10:
        return 0.0
    arr = np.array(closes[-30:], dtype=float)
    returns = np.diff(arr) / arr[:-1]
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(252.0) * 100.0)


def _vol_of_vol(calls: dict, puts: dict) -> float:
    values = []
    for side in (calls, puts):
        if not isinstance(side, dict):
            continue
        for options in side.values():
            ivs = [safe_float(item.get("iv"), 0.0) for item in options or []]
            ivs = [value for value in ivs if value > 0]
            if len(ivs) < 2:
                continue
            values.append(float(np.std(ivs, ddof=1)))
    if not values:
        return 0.0
    return float(np.mean(values))
