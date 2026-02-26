"""Alternative data layer for institutional-style signal enrichment."""

from __future__ import annotations

import time
from typing import Optional

from bot.config import AltDataConfig
from bot.news_scanner import NewsScanner
from bot.number_utils import safe_float


class AltDataEngine:
    """Compute and cache alternative-data overlays used by entry decisions."""

    def __init__(self, config: AltDataConfig, *, news_scanner: Optional[NewsScanner] = None):
        self.config = config
        self.news_scanner = news_scanner
        self._social_cache: dict[str, tuple[float, dict]] = {}

    def build_signals(
        self,
        *,
        symbol: str,
        chain_data: dict,
        flow_context: Optional[dict] = None,
        previous_chain_data: Optional[dict] = None,
    ) -> dict:
        """Return the full alternative-data signal payload for one symbol."""
        payload: dict = {
            "gex": {},
            "dark_pool_proxy": {},
            "social_sentiment": {},
        }
        if bool(self.config.gex_enabled):
            payload["gex"] = self.estimate_gex(
                chain_data,
                underlying_price=safe_float((chain_data or {}).get("underlying_price"), 0.0),
            )
        if bool(self.config.dark_pool_proxy_enabled):
            payload["dark_pool_proxy"] = self.estimate_dark_pool_proxy(
                chain_data,
                flow_context=flow_context or {},
                previous_chain_data=previous_chain_data,
            )
        if bool(self.config.social_sentiment_enabled):
            payload["social_sentiment"] = self.get_social_sentiment(symbol)
        return payload

    @staticmethod
    def estimate_gex(chain_data: dict, *, underlying_price: float = 0.0) -> dict:
        """Estimate net dealer gamma exposure and the nearest gamma-flip point."""
        per_strike: dict[float, float] = {}

        for option in _iter_contracts(chain_data, "calls"):
            strike = safe_float(option.get("strike"), 0.0)
            gamma = safe_float(option.get("gamma"), 0.0)
            oi = safe_float(option.get("open_interest"), 0.0)
            if strike <= 0 or gamma <= 0 or oi <= 0:
                continue
            # Call OI proxy: dealers typically hedge against call-driven upside exposure.
            per_strike[strike] = per_strike.get(strike, 0.0) + (gamma * oi * 100.0)

        for option in _iter_contracts(chain_data, "puts"):
            strike = safe_float(option.get("strike"), 0.0)
            gamma = safe_float(option.get("gamma"), 0.0)
            oi = safe_float(option.get("open_interest"), 0.0)
            if strike <= 0 or gamma <= 0 or oi <= 0:
                continue
            # Put OI proxy: opposite sign relative to calls for net-gamma pressure.
            per_strike[strike] = per_strike.get(strike, 0.0) - (gamma * oi * 100.0)

        if not per_strike:
            return {
                "gex_flip": round(max(0.0, float(underlying_price)), 2),
                "dealer_gamma_bias": "neutral",
                "magnitude": 0.0,
                "net_gamma_by_strike": {},
            }

        strikes = sorted(per_strike.keys())
        cumulative: list[tuple[float, float]] = []
        running = 0.0
        for strike in strikes:
            running += per_strike[strike]
            cumulative.append((strike, running))

        flip = _gamma_flip_from_cumulative(cumulative)
        if flip is None:
            # Fallback: strike where cumulative gamma is closest to neutral.
            flip = min(cumulative, key=lambda pair: abs(pair[1]))[0]

        net_total = sum(per_strike.values())
        if net_total > 0:
            bias = "long"
        elif net_total < 0:
            bias = "short"
        else:
            bias = "neutral"

        return {
            "gex_flip": round(float(flip), 2),
            "dealer_gamma_bias": bias,
            "magnitude": round(abs(net_total) / 1_000_000.0, 3),
            "net_gamma_by_strike": {
                f"{strike:.2f}": round(per_strike[strike], 2)
                for strike in strikes
            },
        }

    @staticmethod
    def estimate_dark_pool_proxy(
        chain_data: dict,
        *,
        flow_context: Optional[dict] = None,
        previous_chain_data: Optional[dict] = None,
    ) -> dict:
        """Approximate institutional accumulation/distribution pressure in [-1, 1]."""
        curr = _totals(chain_data)
        prev = _totals(previous_chain_data or {})

        call_oi_shift = curr["call_oi"] - prev["call_oi"]
        put_oi_shift = curr["put_oi"] - prev["put_oi"]
        oi_den = abs(call_oi_shift) + abs(put_oi_shift) + 1.0
        oi_skew = (call_oi_shift - put_oi_shift) / oi_den

        vol_den = curr["call_volume"] + curr["put_volume"] + 1.0
        volume_skew = (curr["call_volume"] - curr["put_volume"]) / vol_den

        flow = flow_context if isinstance(flow_context, dict) else {}
        directional_bias = str(flow.get("directional_bias", "neutral")).lower()
        institutional = str(flow.get("institutional_flow_direction", "neutral")).lower()
        flow_bias = 0.0
        if directional_bias == "bullish":
            flow_bias += 0.5
        elif directional_bias == "bearish":
            flow_bias -= 0.5
        if institutional in {"bullish", "buying"}:
            flow_bias += 0.5
        elif institutional in {"bearish", "selling"}:
            flow_bias -= 0.5

        score = _clamp((0.45 * oi_skew) + (0.35 * volume_skew) + (0.20 * flow_bias), -1.0, 1.0)
        pressure = "accumulation" if score > 0.15 else "distribution" if score < -0.15 else "neutral"

        return {
            "dark_pool_proxy_score": round(score, 4),
            "institutional_pressure": pressure,
            "components": {
                "oi_skew": round(oi_skew, 4),
                "volume_skew": round(volume_skew, 4),
                "flow_bias": round(flow_bias, 4),
                "call_oi_shift": round(call_oi_shift, 2),
                "put_oi_shift": round(put_oi_shift, 2),
            },
        }

    def get_social_sentiment(self, symbol: str) -> dict:
        """Return normalized LLM social/news sentiment with per-symbol caching."""
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "catalyst": None,
                "confidence": 0.5,
            }

        ttl_seconds = max(1, int(self.config.social_sentiment_cache_minutes)) * 60
        cached = self._social_cache.get(symbol_key)
        if cached and (time.time() - cached[0]) < ttl_seconds:
            return cached[1]

        scanner = self.news_scanner
        if scanner is None:
            sentiment = {
                "sentiment": "neutral",
                "score": 0.0,
                "catalyst": None,
                "confidence": 0.5,
            }
            self._social_cache[symbol_key] = (time.time(), sentiment)
            return sentiment

        raw = scanner.get_symbol_sentiment(
            symbol_key,
            model=self.config.social_sentiment_model,
            cache_seconds=ttl_seconds,
        )
        sentiment = _normalize_social_payload(raw)
        self._social_cache[symbol_key] = (time.time(), sentiment)
        return sentiment


def _iter_contracts(chain_data: dict, side: str):
    side_map = chain_data.get(side, {}) if isinstance(chain_data, dict) else {}
    if not isinstance(side_map, dict):
        return
    for contracts in side_map.values():
        if not isinstance(contracts, list):
            continue
        for row in contracts:
            if isinstance(row, dict):
                yield row


def _totals(chain_data: dict) -> dict:
    out = {"call_oi": 0.0, "put_oi": 0.0, "call_volume": 0.0, "put_volume": 0.0}
    for option in _iter_contracts(chain_data, "calls"):
        out["call_oi"] += max(0.0, safe_float(option.get("open_interest"), 0.0))
        out["call_volume"] += max(0.0, safe_float(option.get("volume"), 0.0))
    for option in _iter_contracts(chain_data, "puts"):
        out["put_oi"] += max(0.0, safe_float(option.get("open_interest"), 0.0))
        out["put_volume"] += max(0.0, safe_float(option.get("volume"), 0.0))
    return out


def _gamma_flip_from_cumulative(cumulative: list[tuple[float, float]]) -> Optional[float]:
    if len(cumulative) < 2:
        return None
    prev_strike, prev_value = cumulative[0]
    for strike, value in cumulative[1:]:
        if prev_value == 0.0:
            return prev_strike
        if value == 0.0:
            return strike
        if (prev_value < 0.0 < value) or (prev_value > 0.0 > value):
            span = strike - prev_strike
            if span <= 0:
                return strike
            frac = abs(prev_value) / (abs(prev_value) + abs(value))
            return prev_strike + (span * frac)
        prev_strike, prev_value = strike, value
    return None


def _normalize_social_payload(payload: object) -> dict:
    raw = payload if isinstance(payload, dict) else {}
    sentiment = str(raw.get("sentiment", "neutral") or "neutral").lower()
    if sentiment not in {"bullish", "bearish", "neutral"}:
        sentiment = "neutral"

    score = safe_float(raw.get("score"), float("nan"))
    if score != score:  # NaN guard
        score = {"bullish": 0.5, "bearish": -0.5, "neutral": 0.0}.get(sentiment, 0.0)
    score = _clamp(score, -1.0, 1.0)

    confidence = safe_float(raw.get("confidence"), float("nan"))
    if confidence != confidence:
        confidence = 0.5
    if confidence > 1.0:
        confidence /= 100.0
    confidence = _clamp(confidence, 0.0, 1.0)

    catalyst = raw.get("catalyst")
    if catalyst is None:
        catalyst = raw.get("key_event")
    catalyst_text = str(catalyst).strip() if catalyst is not None else ""

    return {
        "sentiment": sentiment,
        "score": round(score, 4),
        "catalyst": catalyst_text or None,
        "confidence": round(confidence, 4),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
