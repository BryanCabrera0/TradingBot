"""Market regime detection from price, volatility, breadth, and risk proxies."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bot.data_store import dump_json, load_json
from bot.number_utils import safe_float

logger = logging.getLogger(__name__)

BULL_TREND = "BULL_TREND"
BEAR_TREND = "BEAR_TREND"
HIGH_VOL_CHOP = "HIGH_VOL_CHOP"
LOW_VOL_GRIND = "LOW_VOL_GRIND"
CRASH_CRISIS = "CRASH/CRISIS"
# Backwards-compatible alias (historical typo kept for imports/tests).
CRASH_CRIISIS = CRASH_CRISIS
MEAN_REVERSION = "MEAN_REVERSION"

ALL_REGIMES = {
    BULL_TREND,
    BEAR_TREND,
    HIGH_VOL_CHOP,
    LOW_VOL_GRIND,
    CRASH_CRISIS,
    MEAN_REVERSION,
}

DEFAULT_STRATEGY_WEIGHTS = {
    "credit_spreads": 1.0,
    "iron_condors": 1.0,
    "covered_calls": 1.0,
    "naked_puts": 1.0,
    "calendar_spreads": 1.0,
    "strangles": 1.0,
    "broken_wing_butterfly": 1.0,
    "earnings_vol_crush": 1.0,
}


@dataclass
class RegimeState:
    """Result of a regime classification pass."""

    regime: str
    confidence: float
    sub_signals: dict = field(default_factory=dict)
    recommended_strategy_weights: dict = field(default_factory=lambda: dict(DEFAULT_STRATEGY_WEIGHTS))
    recommended_position_size_scalar: float = 1.0

    def as_context(self) -> dict:
        return {
            "regime": self.regime,
            "confidence": self.confidence,
            "sub_signals": self.sub_signals,
            "recommended_strategy_weights": self.recommended_strategy_weights,
            "recommended_position_size_scalar": self.recommended_position_size_scalar,
        }


class MarketRegimeDetector:
    """Classify current market state into actionable options-trading regimes."""

    def __init__(
        self,
        *,
        get_price_history: Optional[Callable[[str, int], list[dict]]] = None,
        get_quote: Optional[Callable[[str], dict]] = None,
        get_option_chain: Optional[Callable[[str], dict]] = None,
        config: Optional[dict] = None,
    ):
        self.get_price_history = get_price_history
        self.get_quote = get_quote
        self.get_option_chain = get_option_chain
        self.config = config or {}
        self.cache_seconds = max(0, int(self.config.get("cache_seconds", 1800) or 1800))
        self.history_path = Path(self.config.get("history_file", "bot/data/regime_history.json"))
        self._cached_state: Optional[RegimeState] = None
        self._cached_at: Optional[datetime] = None
        self._last_regime: Optional[str] = None

    def detect(self) -> RegimeState:
        """Collect market proxies from data providers and classify regime."""
        now = datetime.utcnow()
        if (
            self._cached_state is not None
            and self._cached_at is not None
            and self.cache_seconds > 0
            and (now - self._cached_at) < timedelta(seconds=self.cache_seconds)
        ):
            return self._cached_state

        inputs = self._collect_inputs()
        state = self.detect_from_inputs(inputs)
        state = self._apply_regime_momentum(state)
        if self._last_regime and self._last_regime != state.regime:
            state.sub_signals["transition_from"] = self._last_regime
        self._last_regime = state.regime
        self._persist_history(state)
        self._cached_state = state
        self._cached_at = now
        return state

    def detect_from_inputs(self, inputs: dict) -> RegimeState:
        """Classify regime from normalized input readings."""
        vix = safe_float(inputs.get("vix_level"), 20.0)
        vix_ratio = safe_float(inputs.get("vix_term_ratio"), 1.0)
        trend_score = _clamp(safe_float(inputs.get("spy_trend_score"), 0.0), -1.0, 1.0)
        breadth = _clamp(safe_float(inputs.get("breadth_above_50ma"), 0.5), 0.0, 1.0)
        put_call = max(0.01, safe_float(inputs.get("put_call_ratio"), 1.0))
        spread_proxy = safe_float(inputs.get("risk_appetite_spread"), 0.0)
        vol_rp = safe_float(inputs.get("realized_vs_implied_spread"), 0.0)
        vol_of_vol = max(0.0, safe_float(inputs.get("vol_of_vol"), 0.0))
        vix3m = safe_float(inputs.get("vix_3m"), 0.0)
        vix6m = safe_float(inputs.get("vix_6m"), 0.0)
        vix1y = safe_float(inputs.get("vix_1y"), 0.0)
        steepness = safe_float(inputs.get("term_structure_steepness"), 0.0)
        momentum = safe_float(inputs.get("term_structure_momentum"), 0.0)
        flattening_rapid = momentum <= -0.03

        regime = HIGH_VOL_CHOP
        if vix >= 35 or vix_ratio >= 1.05 or (steepness < 0.0 and vix >= 20):
            regime = CRASH_CRISIS
        elif trend_score >= 0.35 and breadth >= 0.55 and vix < 24:
            regime = BULL_TREND
        elif trend_score <= -0.35 and breadth <= 0.45 and vix >= 17:
            regime = BEAR_TREND
        elif (put_call >= 1.15 or put_call <= 0.75) and abs(trend_score) < 0.2 and vix < 28:
            regime = MEAN_REVERSION
        elif vix <= 15.5 and vol_of_vol < 0.12:
            regime = LOW_VOL_GRIND
        elif vix >= 24 and vol_of_vol >= 0.18:
            regime = HIGH_VOL_CHOP
        elif abs(trend_score) < 0.15 and 0.45 <= breadth <= 0.60:
            regime = LOW_VOL_GRIND

        confidence = self._confidence_for(
            regime=regime,
            trend_score=trend_score,
            vix=vix,
            vix_ratio=vix_ratio,
            breadth=breadth,
            put_call=put_call,
            vol_of_vol=vol_of_vol,
            vol_rp=vol_rp,
            steepness=steepness,
            momentum=momentum,
        )
        weights = self._strategy_weights_for_regime(regime, inputs)
        size_scalar = self._size_scalar_for_regime(regime, confidence, vol_of_vol)

        sub_signals = {
            "spy_trend_score": round(trend_score, 4),
            "vix_level": round(vix, 4),
            "vix_term_ratio": round(vix_ratio, 4),
            "vix_3m": round(vix3m, 4),
            "vix_6m": round(vix6m, 4),
            "vix_1y": round(vix1y, 4),
            "term_structure_steepness": round(steepness, 6),
            "term_structure_momentum": round(momentum, 6),
            "term_structure_flattening_warning": bool(flattening_rapid),
            "put_call_ratio": round(put_call, 4),
            "risk_appetite_spread": round(spread_proxy, 6),
            "breadth_above_50ma": round(breadth, 4),
            "realized_vs_implied_spread": round(vol_rp, 4),
            "vol_of_vol": round(vol_of_vol, 4),
            "as_of": datetime.utcnow().isoformat() + "Z",
        }
        return RegimeState(
            regime=regime,
            confidence=round(confidence, 4),
            sub_signals=sub_signals,
            recommended_strategy_weights=weights,
            recommended_position_size_scalar=round(size_scalar, 4),
        )

    def _apply_regime_momentum(self, state: RegimeState) -> RegimeState:
        payload = load_json(self.history_path, {"entries": []})
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return state
        streak = 0
        for row in reversed(entries):
            if not isinstance(row, dict):
                continue
            if str(row.get("regime", "")) != state.regime:
                break
            streak += 1
            if streak >= 3:
                break
        if streak >= 3:
            state.confidence = round(_clamp(state.confidence + 0.10, 0.05, 0.99), 4)
            state.sub_signals["regime_momentum"] = streak
        return state

    def _persist_history(self, state: RegimeState) -> None:
        payload = load_json(self.history_path, {"entries": []})
        if not isinstance(payload, dict):
            payload = {"entries": []}
        entries = payload.get("entries")
        if not isinstance(entries, list):
            entries = []
            payload["entries"] = entries

        now = datetime.now(timezone.utc)
        entries.append(
            {
                "timestamp": now.isoformat().replace("+00:00", "Z"),
                "regime": state.regime,
                "confidence": state.confidence,
                "sub_signals": dict(state.sub_signals),
            }
        )
        cutoff = now - timedelta(days=30)
        compact: list[dict] = []
        for row in entries:
            if not isinstance(row, dict):
                continue
            ts_raw = str(row.get("timestamp", "")).replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                compact.append(row)
        payload["entries"] = compact[-5000:]
        dump_json(self.history_path, payload)

    def _collect_inputs(self) -> dict:
        """Collect lightweight regime inputs from available providers."""
        spy_prices = self._close_series("SPY", 260)
        trend_score = self._trend_score(spy_prices)

        vix_level = self._quote_price("$VIX") or self._quote_price("^VIX") or self._quote_price("VIX")
        vix3m = self._quote_price("$VIX3M") or self._quote_price("^VIX3M") or self._quote_price("VIX3M")
        vix6m = self._quote_price("$VIX6M") or self._quote_price("^VIX6M") or self._quote_price("VIX6M")
        vix1y = self._quote_price("$VIX1Y") or self._quote_price("^VIX1Y") or self._quote_price("VIX1Y")
        spy_chain = self._spy_option_chain()
        if vix3m <= 0:
            vix3m = self._estimate_term_iv_from_chain(spy_chain, target_dte=90)
        if vix6m <= 0:
            vix6m = self._estimate_term_iv_from_chain(spy_chain, target_dte=180)
        if vix1y <= 0:
            vix1y = self._estimate_term_iv_from_chain(spy_chain, target_dte=360)
        vix_term_ratio = (vix_level / vix3m) if vix_level > 0 and vix3m > 0 else 1.0
        term_structure_steepness = ((vix3m / vix_level) - 1.0) if vix_level > 0 and vix3m > 0 else 0.0
        term_structure_momentum = self._term_structure_momentum(term_structure_steepness)

        put_call_ratio = self._estimate_put_call_ratio()
        risk_appetite_spread = self._risk_appetite_spread()
        breadth = self._breadth_proxy()
        realized_vs_implied = self._realized_vs_implied(spy_prices)
        vol_of_vol = self._vol_of_vol_proxy()

        return {
            "spy_trend_score": trend_score,
            "vix_level": vix_level,
            "vix_term_ratio": vix_term_ratio,
            "vix_3m": vix3m,
            "vix_6m": vix6m,
            "vix_1y": vix1y,
            "term_structure_steepness": term_structure_steepness,
            "term_structure_momentum": term_structure_momentum,
            "put_call_ratio": put_call_ratio,
            "risk_appetite_spread": risk_appetite_spread,
            "breadth_above_50ma": breadth,
            "realized_vs_implied_spread": realized_vs_implied,
            "vol_of_vol": vol_of_vol,
        }

    def _spy_option_chain(self) -> dict:
        if self.get_option_chain is None:
            return {}
        try:
            raw = self.get_option_chain("SPY")
        except Exception:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _estimate_term_iv_from_chain(self, chain: dict, *, target_dte: int) -> float:
        if not isinstance(chain, dict):
            return 0.0
        iv_values: list[float] = []
        for map_key in ("callExpDateMap", "putExpDateMap"):
            exp_map = chain.get(map_key)
            if not isinstance(exp_map, dict):
                continue
            for expiry_key, strike_map in exp_map.items():
                if not isinstance(strike_map, dict):
                    continue
                dte = _parse_dte_from_expiry_key(expiry_key)
                if dte is None or abs(dte - target_dte) > 40:
                    continue
                for contracts in strike_map.values():
                    if not isinstance(contracts, list):
                        continue
                    for contract in contracts:
                        if not isinstance(contract, dict):
                            continue
                        iv = safe_float(contract.get("volatility", contract.get("iv", 0.0)), 0.0)
                        if iv <= 0:
                            continue
                        iv_values.append(iv * 100.0 if iv <= 1.0 else iv)
        if not iv_values:
            return 0.0
        return float(np.mean(np.array(iv_values, dtype=float)))

    def _term_structure_momentum(self, current_steepness: float) -> float:
        payload = load_json(self.history_path, {"entries": []})
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list) or not entries:
            return 0.0
        previous = None
        previous_ts = None
        for row in reversed(entries):
            if not isinstance(row, dict):
                continue
            sub = row.get("sub_signals", {}) if isinstance(row.get("sub_signals"), dict) else {}
            prev = safe_float(sub.get("term_structure_steepness"), None)
            if prev is None:
                continue
            ts_raw = str(row.get("timestamp", "")).replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                continue
            previous = prev
            previous_ts = ts
            break
        if previous is None or previous_ts is None:
            return 0.0
        now = datetime.now(timezone.utc)
        if previous_ts.tzinfo is None:
            previous_ts = previous_ts.replace(tzinfo=timezone.utc)
        elapsed_days = max(1e-6, (now - previous_ts).total_seconds() / 86400.0)
        return (current_steepness - previous) / elapsed_days

    def _close_series(self, symbol: str, days: int) -> list[float]:
        if self.get_price_history is None:
            return []
        try:
            rows = self.get_price_history(symbol, days)
        except Exception:
            return []
        closes = [safe_float(row.get("close"), 0.0) for row in (rows or []) if isinstance(row, dict)]
        return [value for value in closes if value > 0]

    def _quote_price(self, symbol: str) -> float:
        if self.get_quote is None:
            return 0.0
        try:
            quote = self.get_quote(symbol)
        except Exception:
            return 0.0
        if not isinstance(quote, dict):
            return 0.0
        ref = quote.get("quote", quote)
        if not isinstance(ref, dict):
            return 0.0
        return safe_float(ref.get("lastPrice", ref.get("mark", 0.0)), 0.0)

    def _trend_score(self, closes: list[float]) -> float:
        if len(closes) < 200:
            return 0.0

        close = closes[-1]
        sma20 = float(np.mean(closes[-20:]))
        sma50 = float(np.mean(closes[-50:]))
        sma200 = float(np.mean(closes[-200:]))
        slope20 = _slope(closes[-20:])
        slope50 = _slope(closes[-50:])
        slope200 = _slope(closes[-200:])

        score = 0.0
        score += 0.30 if close > sma20 else -0.30
        score += 0.30 if close > sma50 else -0.30
        score += 0.25 if close > sma200 else -0.25
        score += 0.10 if slope20 > 0 else -0.10
        score += 0.10 if slope50 > 0 else -0.10
        score += 0.05 if slope200 > 0 else -0.05
        return _clamp(score, -1.0, 1.0)

    def _estimate_put_call_ratio(self) -> float:
        if self.get_option_chain is None:
            return 1.0
        try:
            raw = self.get_option_chain("SPY")
        except Exception:
            return 1.0
        if not isinstance(raw, dict):
            return 1.0
        put_volume = _sum_chain_field(raw.get("putExpDateMap"), "totalVolume", "volume")
        call_volume = _sum_chain_field(raw.get("callExpDateMap"), "totalVolume", "volume")
        if call_volume <= 0:
            return 1.0
        return put_volume / call_volume

    def _risk_appetite_spread(self) -> float:
        hyg = self._close_series("HYG", 40)
        tlt = self._close_series("TLT", 40)
        if len(hyg) < 2 or len(tlt) < 2:
            return 0.0
        hyg_ret = (hyg[-1] / hyg[0]) - 1.0
        tlt_ret = (tlt[-1] / tlt[0]) - 1.0
        return hyg_ret - tlt_ret

    def _breadth_proxy(self) -> float:
        # Sector ETF approximation for "% above 50DMA".
        sector_etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
        ratios = []
        for symbol in sector_etfs:
            closes = self._close_series(symbol, 90)
            if len(closes) < 50:
                continue
            sma50 = float(np.mean(closes[-50:]))
            if sma50 <= 0:
                continue
            ratios.append(1.0 if closes[-1] > sma50 else 0.0)
        if not ratios:
            return 0.5
        return float(np.mean(ratios))

    def _realized_vs_implied(self, closes: list[float]) -> float:
        if len(closes) < 22:
            return 0.0
        returns = np.diff(np.array(closes[-22:], dtype=float)) / np.array(closes[-22:-1], dtype=float)
        realized = float(np.std(returns, ddof=1) * math.sqrt(252.0) * 100.0) if len(returns) > 1 else 0.0
        # If IV quote is unavailable, fallback to VIX as an implied-vol proxy.
        implied = self._quote_price("$VIX") or self._quote_price("^VIX") or realized
        return implied - realized

    def _vol_of_vol_proxy(self) -> float:
        vix_series = self._close_series("^VIX", 40)
        if len(vix_series) < 10:
            return 0.0
        returns = np.diff(np.array(vix_series[-20:], dtype=float)) / np.array(vix_series[-20:-1], dtype=float)
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns, ddof=1))

    @staticmethod
    def _confidence_for(
        *,
        regime: str,
        trend_score: float,
        vix: float,
        vix_ratio: float,
        breadth: float,
        put_call: float,
        vol_of_vol: float,
        vol_rp: float,
        steepness: float,
        momentum: float,
    ) -> float:
        confidence = 0.50
        if regime in {BULL_TREND, BEAR_TREND}:
            confidence += min(0.35, abs(trend_score) * 0.30)
            confidence += min(0.15, abs(breadth - 0.5))
        elif regime == CRASH_CRISIS:
            confidence += min(0.30, max(0.0, (vix - 30.0) / 20.0))
            confidence += min(0.10, max(0.0, vix_ratio - 1.0))
        elif regime == HIGH_VOL_CHOP:
            confidence += min(0.20, max(0.0, (vix - 20.0) / 20.0))
            confidence += min(0.15, vol_of_vol)
        elif regime == LOW_VOL_GRIND:
            confidence += min(0.20, max(0.0, (16.0 - vix) / 16.0))
            confidence += 0.05 if vol_rp > 0 else 0.0
        elif regime == MEAN_REVERSION:
            confidence += min(0.20, abs(put_call - 1.0))
            confidence += min(0.10, abs(trend_score))
        if steepness > 0.15 and regime in {BULL_TREND, LOW_VOL_GRIND}:
            confidence += 0.05
        if steepness < 0.0 and regime in {HIGH_VOL_CHOP, CRASH_CRISIS, BEAR_TREND}:
            confidence += 0.06
        if momentum <= -0.03 and regime in {HIGH_VOL_CHOP, CRASH_CRISIS, BEAR_TREND}:
            confidence += 0.04
        return _clamp(confidence, 0.05, 0.99)

    @staticmethod
    def _strategy_weights_for_regime(regime: str, inputs: dict) -> dict:
        weights = dict(DEFAULT_STRATEGY_WEIGHTS)
        if regime == CRASH_CRISIS:
            weights.update(
                {
                    "credit_spreads": 0.2,
                    "iron_condors": 0.0,
                    "covered_calls": 0.6,
                    "naked_puts": 0.0,
                    "calendar_spreads": 1.2,
                    "strangles": 0.0,
                    "broken_wing_butterfly": 0.6,
                    "earnings_vol_crush": 0.4,
                }
            )
        elif regime == HIGH_VOL_CHOP:
            weights.update(
                {
                    "credit_spreads": 1.1,
                    "iron_condors": 1.6,
                    "covered_calls": 0.9,
                    "naked_puts": 0.8,
                    "calendar_spreads": 0.9,
                    "strangles": 1.4,
                    "broken_wing_butterfly": 1.0,
                    "earnings_vol_crush": 1.0,
                }
            )
        elif regime == BULL_TREND:
            weights.update(
                {
                    "credit_spreads": 1.4,
                    "iron_condors": 0.7,
                    "covered_calls": 1.2,
                    "naked_puts": 1.3,
                    "calendar_spreads": 0.8,
                    "strangles": 0.6,
                    "broken_wing_butterfly": 1.3,
                    "earnings_vol_crush": 1.0,
                }
            )
        elif regime == BEAR_TREND:
            weights.update(
                {
                    "credit_spreads": 1.2,
                    "iron_condors": 0.7,
                    "covered_calls": 0.8,
                    "naked_puts": 0.5,
                    "calendar_spreads": 0.9,
                    "strangles": 0.6,
                    "broken_wing_butterfly": 1.3,
                    "earnings_vol_crush": 1.0,
                }
            )
        elif regime == LOW_VOL_GRIND:
            weights.update(
                {
                    "credit_spreads": 0.8,
                    "iron_condors": 0.9,
                    "covered_calls": 0.9,
                    "naked_puts": 0.6,
                    "calendar_spreads": 1.6,
                    "strangles": 0.5,
                    "broken_wing_butterfly": 1.0,
                    "earnings_vol_crush": 0.8,
                }
            )
        elif regime == MEAN_REVERSION:
            weights.update(
                {
                    "credit_spreads": 1.5,
                    "iron_condors": 1.1,
                    "covered_calls": 1.0,
                    "naked_puts": 1.0,
                    "calendar_spreads": 0.9,
                    "strangles": 0.9,
                    "broken_wing_butterfly": 1.1,
                    "earnings_vol_crush": 1.0,
                }
            )

        # Optional directional nudges for trend regimes.
        trend = safe_float(inputs.get("spy_trend_score"), 0.0)
        if trend > 0.3:
            weights["naked_puts"] = _clamp(weights["naked_puts"] * 1.05, 0.0, 2.0)
        elif trend < -0.3:
            weights["covered_calls"] = _clamp(weights["covered_calls"] * 0.95, 0.0, 2.0)
        return {name: round(_clamp(value, 0.0, 2.0), 4) for name, value in weights.items()}

    @staticmethod
    def _size_scalar_for_regime(regime: str, confidence: float, vol_of_vol: float) -> float:
        if regime == CRASH_CRISIS:
            base = 0.55
        elif regime == HIGH_VOL_CHOP:
            base = 0.95
        elif regime == LOW_VOL_GRIND:
            base = 0.80
        else:
            base = 1.0

        uncertainty_penalty = 0.12 if confidence < 0.55 else 0.0
        vol_penalty = min(0.15, max(0.0, vol_of_vol - 0.20))
        confidence_boost = min(0.20, max(0.0, confidence - 0.70))
        scalar = base - uncertainty_penalty - vol_penalty + confidence_boost
        return _clamp(scalar, 0.5, 1.5)


def _sum_chain_field(exp_map: object, *field_names: str) -> float:
    if not isinstance(exp_map, dict):
        return 0.0
    total = 0.0
    for strike_map in exp_map.values():
        if not isinstance(strike_map, dict):
            continue
        for contracts in strike_map.values():
            if not isinstance(contracts, list):
                continue
            for contract in contracts:
                if not isinstance(contract, dict):
                    continue
                for name in field_names:
                    value = safe_float(contract.get(name), 0.0)
                    if value > 0:
                        total += value
                        break
    return total


def _parse_dte_from_expiry_key(value: object) -> Optional[int]:
    raw = str(value or "")
    if ":" in raw:
        tail = raw.rsplit(":", 1)[-1]
        try:
            return int(float(tail))
        except ValueError:
            pass
    match = re.search(r"(\d+)", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    idx = np.arange(len(arr), dtype=float)
    x_center = idx - float(np.mean(idx))
    denom = float(np.sum(x_center ** 2))
    if denom <= 0:
        return 0.0
    y_center = arr - float(np.mean(arr))
    slope = float(np.sum(x_center * y_center) / denom)
    base = max(1e-9, float(np.mean(arr)))
    return slope / base


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
