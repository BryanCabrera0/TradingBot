"""Earnings calendar lookup with daily local cache."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from bot.data_store import dump_json, ensure_data_dir, load_json

logger = logging.getLogger(__name__)

YAHOO_CALENDAR_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
EARNINGS_CACHE_PATH = Path("bot/data/earnings_cache.json")


class EarningsCalendar:
    """Fetch and cache upcoming earnings dates."""

    def __init__(
        self,
        cache_path: Path | str = EARNINGS_CACHE_PATH,
        request_timeout_seconds: int = 10,
    ):
        self.cache_path = Path(cache_path)
        self.request_timeout_seconds = max(2, int(request_timeout_seconds))
        ensure_data_dir(self.cache_path.parent)

    def earnings_within_window(self, symbol: str, expiration: str) -> tuple[bool, Optional[str]]:
        """Return True when earnings fall on/before trade expiration date."""
        earnings_dt = self.get_earnings_date(symbol)
        exp_date = _parse_date(expiration)
        if earnings_dt is None or exp_date is None:
            return False, None

        if earnings_dt <= exp_date:
            return True, earnings_dt.isoformat()
        return False, None

    def get_earnings_date(self, symbol: str) -> Optional[date]:
        """Return cached or fetched next earnings date for ``symbol``."""
        symbol_key = symbol.upper().strip()
        cache = load_json(self.cache_path, {"as_of": "", "symbols": {}})
        if not isinstance(cache, dict):
            cache = {"as_of": "", "symbols": {}}

        today_iso = date.today().isoformat()
        symbols = cache.get("symbols")
        if not isinstance(symbols, dict):
            symbols = {}
            cache["symbols"] = symbols

        if cache.get("as_of") == today_iso:
            cached = symbols.get(symbol_key)
            parsed = _parse_date(cached) if cached else None
            if parsed:
                return parsed

        fetched = self._fetch_from_yahoo(symbol_key)
        cache["as_of"] = today_iso
        symbols[symbol_key] = fetched.isoformat() if fetched else ""
        dump_json(self.cache_path, cache)
        return fetched

    def _fetch_from_yahoo(self, symbol: str) -> Optional[date]:
        url = YAHOO_CALENDAR_URL.format(symbol=symbol)
        params = {"modules": "calendarEvents"}
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.request_timeout_seconds,
                headers={"User-Agent": "TradingBot-EarningsCalendar/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Failed to fetch earnings for %s: %s", symbol, exc)
            return None

        result = (
            payload.get("quoteSummary", {})
            .get("result", [{}])[0]
            .get("calendarEvents", {})
        )
        earnings_date = result.get("earnings", {}).get("earningsDate", [])
        if not isinstance(earnings_date, list) or not earnings_date:
            return None

        first = earnings_date[0]
        if isinstance(first, dict):
            raw = first.get("fmt") or first.get("raw")
        else:
            raw = first
        return _parse_date(raw)


def _parse_date(raw_value: object) -> Optional[date]:
    if raw_value in (None, ""):
        return None
    if isinstance(raw_value, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(raw_value)).date()
        except Exception:
            return None

    text = str(raw_value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    if "T" in text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None
