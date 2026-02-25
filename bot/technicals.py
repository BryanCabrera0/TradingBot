"""Technical indicator context generation for strategy filters."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bot.data_store import dump_json, ensure_data_dir, load_json

logger = logging.getLogger(__name__)

TECHNICALS_CACHE_PATH = Path("bot/data/technicals_cache.json")


@dataclass
class TechnicalContext:
    """Multi-timeframe technical snapshot for a symbol."""

    symbol: str
    as_of: str
    close: float
    rsi14: float
    sma20: float
    sma50: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr14: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_position: float
    volume_ratio: float
    between_bands: bool
    return_5d: float
    return_5d_zscore: float

    def to_dict(self) -> dict:
        return asdict(self)


class TechnicalAnalyzer:
    """Computes and caches technical contexts from daily OHLCV bars."""

    def __init__(self, cache_path: Path | str = TECHNICALS_CACHE_PATH):
        self.cache_path = Path(cache_path)
        ensure_data_dir(self.cache_path.parent)

    def get_context(self, symbol: str, schwab_client) -> Optional[TechnicalContext]:
        """Return a cached-or-fresh technical context for ``symbol``."""
        symbol_key = symbol.upper().strip()
        cached = self._load_cached(symbol_key)
        today_iso = date.today().isoformat()
        if cached and cached.get("as_of") == today_iso:
            return _context_from_payload(cached)

        bars = schwab_client.get_price_history(symbol_key, days=120)
        context = build_technical_context(symbol_key, bars)
        if context is None:
            return None

        self._store_cached(symbol_key, context.to_dict())
        return context

    def _load_cached(self, symbol: str) -> Optional[dict]:
        payload = load_json(self.cache_path, {"symbols": {}})
        symbols = payload.get("symbols", {}) if isinstance(payload, dict) else {}
        data = symbols.get(symbol)
        return data if isinstance(data, dict) else None

    def _store_cached(self, symbol: str, context: dict) -> None:
        payload = load_json(self.cache_path, {"symbols": {}, "updated": ""})
        if not isinstance(payload, dict):
            payload = {"symbols": {}, "updated": ""}
        symbols = payload.get("symbols")
        if not isinstance(symbols, dict):
            symbols = {}
            payload["symbols"] = symbols
        symbols[symbol] = context
        payload["updated"] = date.today().isoformat()
        dump_json(self.cache_path, payload)


def build_technical_context(symbol: str, bars: list[dict]) -> Optional[TechnicalContext]:
    """Build technical indicators from a list of daily OHLCV bars."""
    if not bars:
        return None

    df = pd.DataFrame(bars)
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        logger.debug("Missing OHLCV fields for %s. Got columns: %s", symbol, sorted(df.columns))
        return None

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])
    if df.empty:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI(14)
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # SMA(20/50)
    sma20 = close.rolling(20, min_periods=1).mean()
    sma50 = close.rolling(50, min_periods=1).mean()

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()

    # Bollinger Bands(20,2)
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)

    avg_volume20 = df["volume"].rolling(20, min_periods=1).mean()
    returns = close.pct_change().fillna(0.0)
    rolling_std5 = returns.rolling(5, min_periods=5).std(ddof=0)
    cumulative_return_5d = close.pct_change(periods=5).fillna(0.0)

    latest_close = float(close.iloc[-1])
    latest_upper = float(bb_upper.iloc[-1])
    latest_lower = float(bb_lower.iloc[-1])
    band_width = max(1e-9, latest_upper - latest_lower)
    band_position = (latest_close - latest_lower) / band_width
    latest_return5 = float(cumulative_return_5d.iloc[-1])
    std5 = float(rolling_std5.iloc[-1]) if not np.isnan(rolling_std5.iloc[-1]) else 0.0
    zscore5 = latest_return5 / std5 if std5 > 1e-9 else 0.0

    context = TechnicalContext(
        symbol=symbol.upper(),
        as_of=date.today().isoformat(),
        close=round(latest_close, 4),
        rsi14=round(float(rsi.fillna(50.0).iloc[-1]), 2),
        sma20=round(float(sma20.iloc[-1]), 4),
        sma50=round(float(sma50.iloc[-1]), 4),
        macd=round(float(macd.iloc[-1]), 4),
        macd_signal=round(float(macd_signal.iloc[-1]), 4),
        macd_hist=round(float(macd_hist.iloc[-1]), 4),
        atr14=round(float(atr14.iloc[-1]), 4),
        bollinger_upper=round(latest_upper, 4),
        bollinger_middle=round(float(bb_mid.iloc[-1]), 4),
        bollinger_lower=round(latest_lower, 4),
        bollinger_position=round(float(np.clip(band_position, 0.0, 1.0)), 4),
        volume_ratio=round(
            float(
                (
                    float(df["volume"].fillna(0).iloc[-1])
                    / max(1.0, float(avg_volume20.fillna(0).iloc[-1]))
                )
            ),
            4,
        ),
        between_bands=bool(latest_lower <= latest_close <= latest_upper),
        return_5d=round(latest_return5, 4),
        return_5d_zscore=round(float(zscore5), 4),
    )
    return context


def _context_from_payload(payload: dict) -> Optional[TechnicalContext]:
    try:
        return TechnicalContext(
            symbol=str(payload.get("symbol", "")).upper(),
            as_of=str(payload.get("as_of", "")),
            close=float(payload.get("close", 0.0)),
            rsi14=float(payload.get("rsi14", 50.0)),
            sma20=float(payload.get("sma20", 0.0)),
            sma50=float(payload.get("sma50", 0.0)),
            macd=float(payload.get("macd", 0.0)),
            macd_signal=float(payload.get("macd_signal", 0.0)),
            macd_hist=float(payload.get("macd_hist", 0.0)),
            atr14=float(payload.get("atr14", 0.0)),
            bollinger_upper=float(payload.get("bollinger_upper", 0.0)),
            bollinger_middle=float(payload.get("bollinger_middle", 0.0)),
            bollinger_lower=float(payload.get("bollinger_lower", 0.0)),
            bollinger_position=float(payload.get("bollinger_position", 0.5)),
            volume_ratio=float(payload.get("volume_ratio", 1.0)),
            between_bands=bool(payload.get("between_bands", False)),
            return_5d=float(payload.get("return_5d", 0.0)),
            return_5d_zscore=float(payload.get("return_5d_zscore", 0.0)),
        )
    except Exception:
        return None
