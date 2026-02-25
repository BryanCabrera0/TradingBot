"""Dividend calendar utilities for assignment/pin-risk awareness."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import requests

from bot.data_store import dump_json, ensure_data_dir, load_json

logger = logging.getLogger(__name__)

YAHOO_CALENDAR_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
DIVIDEND_CACHE_PATH = Path("bot/data/dividend_cache.json")


class DividendCalendar:
    """Fetch and cache ex-dividend dates used by strategy scoring filters."""

    def __init__(
        self,
        cache_path: Path | str = DIVIDEND_CACHE_PATH,
        request_timeout_seconds: int = 10,
    ):
        self.cache_path = Path(cache_path)
        self.request_timeout_seconds = max(2, int(request_timeout_seconds))
        ensure_data_dir(self.cache_path.parent)

    def get_ex_dividend_date(self, symbol: str) -> Optional[date]:
        symbol_key = symbol.upper().strip()
        cache = load_json(self.cache_path, {"as_of": "", "symbols": {}})
        if not isinstance(cache, dict):
            cache = {"as_of": "", "symbols": {}}
        symbols = cache.get("symbols")
        if not isinstance(symbols, dict):
            symbols = {}
            cache["symbols"] = symbols

        today_iso = date.today().isoformat()
        cached_raw = symbols.get(symbol_key, {})
        if (
            cache.get("as_of") == today_iso
            and isinstance(cached_raw, dict)
            and cached_raw.get("ex_dividend_date")
        ):
            return _parse_date(cached_raw.get("ex_dividend_date"))

        ex_div = self._fetch_from_yahoo(symbol_key)
        symbols[symbol_key] = {"ex_dividend_date": ex_div.isoformat() if ex_div else ""}
        cache["as_of"] = today_iso
        dump_json(self.cache_path, cache)
        return ex_div

    def assess_trade_risk(
        self,
        *,
        symbol: str,
        strategy: str,
        expiration: str,
        short_strike: float,
        underlying_price: float,
        is_call_side: bool,
    ) -> dict:
        """Return score adjustments and warning flags for dividend-related risk."""
        exp_date = _parse_date(expiration)
        ex_div = self.get_ex_dividend_date(symbol)
        if exp_date is None or ex_div is None or ex_div > exp_date:
            return {"score_adjustment": 0.0, "warning": None, "pin_risk": False}

        strategy_key = str(strategy).lower()
        is_itm_call = bool(is_call_side and underlying_price > short_strike)

        if strategy_key == "covered_call" and is_itm_call:
            warning = (
                f"Ex-dividend {ex_div.isoformat()} before expiration {exp_date.isoformat()} "
                "and call is ITM; elevated early-assignment risk."
            )
            return {
                "score_adjustment": -20.0,
                "warning": warning,
                "pin_risk": False,
                "ex_dividend_date": ex_div.isoformat(),
            }

        proximity = (
            abs(float(underlying_price) - float(short_strike))
            / max(1.0, float(underlying_price))
        )
        pin_risk = proximity <= 0.01
        if "spread" in strategy_key and pin_risk:
            warning = (
                f"Ex-dividend {ex_div.isoformat()} may increase pin risk near short strike "
                f"{short_strike} before expiration {exp_date.isoformat()}."
            )
            return {
                "score_adjustment": -5.0,
                "warning": warning,
                "pin_risk": True,
                "ex_dividend_date": ex_div.isoformat(),
            }

        return {"score_adjustment": 0.0, "warning": None, "pin_risk": False}

    def _fetch_from_yahoo(self, symbol: str) -> Optional[date]:
        url = YAHOO_CALENDAR_URL.format(symbol=symbol)
        params = {"modules": "calendarEvents,summaryDetail"}
        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.request_timeout_seconds,
                headers={"User-Agent": "TradingBot-DividendCalendar/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Failed to fetch dividend data for %s: %s", symbol, exc)
            return None

        result = payload.get("quoteSummary", {}).get("result", [{}])[0]
        summary = result.get("summaryDetail", {})
        ex_div = summary.get("exDividendDate", {})

        if isinstance(ex_div, dict):
            raw = ex_div.get("fmt") or ex_div.get("raw")
        else:
            raw = ex_div
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
