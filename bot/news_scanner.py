"""News intelligence scanner for symbol-specific and macro market headlines."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus

import requests
from defusedxml import ElementTree as DET

from bot.config import NewsConfig
from bot.openai_compat import request_openai_json

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_SEARCH_URL = "https://news.google.com/rss/search"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
MAX_RSS_PAYLOAD_CHARS = 1_000_000
SENTIMENT_EVENT_BLOCKLIST = {"fda", "merger", "acquisition", "lawsuit", "bankruptcy"}

POSITIVE_KEYWORDS = {
    "beats", "beat", "upgrade", "upgraded", "surge", "rally", "bullish",
    "growth", "record", "strong", "guidance raised", "profit jumps",
}

NEGATIVE_KEYWORDS = {
    "miss", "downgrade", "downgraded", "plunge", "lawsuit", "investigation",
    "recall", "layoff", "bankruptcy", "weak", "warning", "guidance cut",
    "fraud", "default",
}

TRACKED_TOPICS = {
    "inflation", "federal reserve", "interest rates", "earnings", "ai",
    "tariffs", "recession", "unemployment", "geopolitics", "volatility",
}
SECRET_PLACEHOLDER_MARKERS = ("your_", "_here", "changeme")


@dataclass
class NewsItem:
    """Structured representation of a single headline."""

    title: str
    link: str
    published: str
    source: str
    sentiment: float
    topics: list[str]

    def to_context(self) -> dict:
        data = asdict(self)
        data["title"] = data["title"][:180]
        return data


class NewsScanner:
    """Fetches and summarizes relevant market and symbol news."""

    def __init__(self, config: NewsConfig):
        self.config = config
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}
        self._sentiment_cache: dict[str, tuple[float, dict]] = {}

    def build_context(self, symbol: str, *, macro_events: Optional[dict] = None) -> dict:
        """Return a structured news context payload for the requested symbol."""
        symbol_news = self.get_symbol_news(symbol) if self.config.include_symbol_news else []
        market_news = self.get_market_news() if self.config.include_market_news else []
        sentiment = self.get_symbol_sentiment(symbol, symbol_news)
        policy = self.trade_direction_policy(symbol, sentiment=sentiment)

        return {
            "symbol": symbol.upper(),
            "symbol_headlines": [item.to_context() for item in symbol_news],
            "market_headlines": [item.to_context() for item in market_news],
            "symbol_sentiment": round(_average_sentiment(symbol_news), 3),
            "market_sentiment": round(_average_sentiment(market_news), 3),
            "dominant_market_topics": _top_topics(market_news, limit=5),
            "llm_sentiment": sentiment,
            "trade_policy": policy,
            "economic_events": macro_events or {},
        }

    def get_symbol_news(self, symbol: str) -> list[NewsItem]:
        symbol_norm = symbol.strip().upper()
        cache_key = f"symbol:{symbol_norm}"
        return self._with_cache(cache_key, lambda: self._fetch_symbol_news(symbol_norm))

    def get_market_news(self) -> list[NewsItem]:
        return self._with_cache("market", self._fetch_market_news)

    def get_symbol_sentiment(
        self,
        symbol: str,
        headlines: list[NewsItem] | None = None,
        *,
        model: Optional[str] = None,
        cache_seconds: Optional[int] = None,
    ) -> dict:
        """Return LLM-scored symbol sentiment with per-symbol caching."""
        symbol_key = symbol.upper().strip()
        model_key = str(model or self.config.llm_model or "").strip() or self.config.llm_model
        ttl = max(0, int(self.config.llm_sentiment_cache_seconds if cache_seconds is None else cache_seconds))
        cache_key = f"{symbol_key}:{model_key}"
        cached = self._sentiment_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1]

        items = headlines if headlines is not None else self.get_symbol_news(symbol_key)
        top_titles = [item.title for item in items[:5]]
        if not top_titles:
            sentiment = {
                "sentiment": "neutral",
                "score": 0.0,
                "catalyst": None,
                "confidence": 0.50,
                "key_event": None,
            }
        elif self.config.llm_sentiment_enabled:
            sentiment = self._score_sentiment_with_llm(symbol_key, top_titles, model=model_key)
        else:
            sentiment = self._fallback_sentiment(top_titles)

        self._sentiment_cache[cache_key] = (time.time(), sentiment)
        return sentiment

    def trade_direction_policy(self, symbol: str, *, sentiment: Optional[dict] = None) -> dict:
        """Map sentiment signal into strategy-side blocks."""
        sentiment = sentiment or self.get_symbol_sentiment(symbol)
        direction = str(sentiment.get("sentiment", "neutral")).lower()
        confidence = float(sentiment.get("confidence", 0.0) or 0.0)
        if confidence > 1.0:
            confidence /= 100.0
        key_event = str(sentiment.get("catalyst", sentiment.get("key_event", "")) or "").lower()

        block_all = any(keyword in key_event for keyword in SENTIMENT_EVENT_BLOCKLIST)
        allow_bull_put = True
        allow_bear_call = True
        reason = ""

        if block_all:
            allow_bull_put = False
            allow_bear_call = False
            reason = f"Binary event risk: {sentiment.get('key_event')}"
        elif direction == "bearish" and confidence > 0.70:
            allow_bull_put = False
            reason = "Bearish high-confidence sentiment blocks bull puts"
        elif direction == "bullish" and confidence > 0.70:
            allow_bear_call = False
            reason = "Bullish high-confidence sentiment blocks bear calls"

        return {
            "allow_bull_put": allow_bull_put,
            "allow_bear_call": allow_bear_call,
            "block_all": block_all,
            "reason": reason,
            "sentiment": direction,
            "confidence": confidence,
            "key_event": sentiment.get("catalyst", sentiment.get("key_event")),
            "score": float(sentiment.get("score", 0.0) or 0.0),
        }

    def _with_cache(self, key: str, loader) -> list[NewsItem]:
        if self.config.cache_seconds > 0:
            cached = self._cache.get(key)
            if cached:
                ts, items = cached
                if (time.time() - ts) < self.config.cache_seconds:
                    return items

        items = loader()
        self._cache[key] = (time.time(), items)
        return items

    def _fetch_symbol_news(self, symbol: str) -> list[NewsItem]:
        queries = [f"{symbol} stock", f"{symbol} earnings", f"{symbol} options"]
        per_query_limit = max(2, self.config.max_symbol_headlines // max(len(queries), 1))
        items: list[NewsItem] = []
        for query in queries:
            items.extend(self._fetch_google_news(query, per_query_limit))
            if len(items) >= self.config.max_symbol_headlines * 2:
                break

        if self.config.finnhub_api_key:
            items.extend(self._fetch_finnhub_news(symbol, self.config.max_symbol_headlines))

        return _dedupe_news(items)[: self.config.max_symbol_headlines]

    def _fetch_market_news(self) -> list[NewsItem]:
        if not self.config.market_queries:
            return []
        per_query_limit = max(2, self.config.max_market_headlines // max(len(self.config.market_queries), 1))
        items: list[NewsItem] = []
        for query in self.config.market_queries:
            items.extend(self._fetch_google_news(query, per_query_limit))
            if len(items) >= self.config.max_market_headlines * 2:
                break
        return _dedupe_news(items)[: self.config.max_market_headlines]

    def _fetch_google_news(self, query: str, limit: int) -> list[NewsItem]:
        url = (
            f"{GOOGLE_NEWS_RSS_SEARCH_URL}"
            f"?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            response = requests.get(
                url,
                timeout=self.config.request_timeout_seconds,
                headers={"User-Agent": "TradingBot-NewsScanner/1.0"},
            )
            response.raise_for_status()
            payload = response.text
            if len(payload) > MAX_RSS_PAYLOAD_CHARS:
                logger.warning("Skipping oversized RSS response for query %r (%d chars).", query, len(payload))
                return []
            return _parse_rss_items(payload, limit)
        except Exception as exc:
            logger.debug("Failed to fetch news for query %r: %s", query, exc)
            return []

    def _fetch_finnhub_news(self, symbol: str, limit: int) -> list[NewsItem]:
        to_date = date_str(datetime.utcnow())
        from_date = date_str(datetime.utcnow() - timedelta(days=5))
        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.config.finnhub_api_key,
        }
        try:
            response = requests.get(
                FINNHUB_COMPANY_NEWS_URL,
                params=params,
                timeout=self.config.request_timeout_seconds,
                headers={"User-Agent": "TradingBot-NewsScanner/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Finnhub news fetch failed for %s: %s", symbol, exc)
            return []

        items: list[NewsItem] = []
        if not isinstance(payload, list):
            return items
        for row in payload[:limit]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("headline", "")).strip()
            if not title:
                continue
            published = ""
            if row.get("datetime"):
                try:
                    published = datetime.utcfromtimestamp(int(row.get("datetime"))).isoformat()
                except Exception:
                    published = ""
            items.append(
                NewsItem(
                    title=title,
                    link=str(row.get("url", "")),
                    published=published,
                    source=str(row.get("source", "Finnhub")),
                    sentiment=_score_headline_sentiment(title),
                    topics=_extract_topics(title),
                )
            )
        return items

    def _score_sentiment_with_llm(self, symbol: str, headlines: list[str], *, model: Optional[str] = None) -> dict:
        api_key = os.getenv("OPENAI_API_KEY")
        if not _is_configured_secret(api_key):
            return self._fallback_sentiment(headlines)

        prompt = {
            "symbol": symbol,
            "headlines": headlines[:5],
            "task": "Return JSON sentiment summary for these headlines.",
            "schema": {
                "sentiment": "bullish|bearish|neutral",
                "score": "-1.0 to 1.0",
                "catalyst": "string|null",
                "confidence": "0.0 to 1.0",
            },
        }
        try:
            raw = request_openai_json(
                api_key=api_key,
                model=str(model or self.config.llm_model),
                system_prompt="You are a financial news sentiment classifier. Respond ONLY with valid JSON.",
                user_prompt=json.dumps(prompt, separators=(",", ":")),
                timeout_seconds=self.config.request_timeout_seconds,
                temperature=0.0,
                reasoning_effort=self.config.llm_reasoning_effort,
                text_verbosity=self.config.llm_text_verbosity,
                max_output_tokens=self.config.llm_max_output_tokens,
                chat_fallback_model=self.config.llm_chat_fallback_model,
                schema_name="news_sentiment",
                schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                        "score": {"type": "number", "minimum": -1, "maximum": 1},
                        "catalyst": {"type": ["string", "null"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["sentiment", "score", "catalyst", "confidence"],
                    "additionalProperties": False,
                },
            )
            parsed = json.loads(raw) if raw else {}
            score = float(parsed.get("score", 0.0) or 0.0)
            confidence = float(parsed.get("confidence", 0.5) or 0.5)
            if confidence > 1.0:
                confidence = confidence / 100.0
            catalyst = parsed.get("catalyst", parsed.get("key_event"))
            return {
                "sentiment": str(parsed.get("sentiment", "neutral")).lower(),
                "score": max(-1.0, min(1.0, score)),
                "confidence": max(0.0, min(1.0, confidence)),
                "catalyst": str(catalyst).strip() if catalyst is not None else None,
                "key_event": str(catalyst).strip() if catalyst is not None else None,
            }
        except Exception as exc:
            logger.debug("LLM sentiment failed for %s: %s", symbol, exc)
            return self._fallback_sentiment(headlines)

    @staticmethod
    def _fallback_sentiment(headlines: list[str]) -> dict:
        score = 0.0
        key_event = None
        for title in headlines:
            title_lower = title.lower()
            score += _score_headline_sentiment(title)
            for keyword in SENTIMENT_EVENT_BLOCKLIST:
                if keyword in title_lower:
                    key_event = keyword
                    break
            if key_event:
                break

        if score > 0.6:
            sentiment = "bullish"
        elif score < -0.6:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        confidence = min(0.95, max(0.40, abs(score) * 0.30 + 0.40))
        return {
            "sentiment": sentiment,
            "score": round(max(-1.0, min(1.0, score)), 4),
            "confidence": round(confidence, 4),
            "catalyst": key_event,
            "key_event": key_event,
        }


def _parse_rss_items(xml_text: str, limit: int) -> list[NewsItem]:
    """Parse RSS XML payload into structured news items."""
    try:
        root = DET.fromstring(xml_text)
    except DET.ParseError:
        return []

    items: list[NewsItem] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        published = (item.findtext("pubDate") or "").strip()
        source = (item.findtext("source") or "").strip()
        if not source and " - " in title:
            source = title.rsplit(" - ", 1)[-1].strip()
        if not title:
            continue

        items.append(
            NewsItem(
                title=title,
                link=link,
                published=published,
                source=source,
                sentiment=_score_headline_sentiment(title),
                topics=_extract_topics(title),
            )
        )
        if len(items) >= limit:
            break
    return items


def _extract_topics(text: str) -> list[str]:
    lowered = text.lower()
    return [topic for topic in TRACKED_TOPICS if topic in lowered]


def _score_headline_sentiment(title: str) -> float:
    lowered = title.lower()
    score = 0.0
    for keyword in POSITIVE_KEYWORDS:
        if keyword in lowered:
            score += 1.0
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in lowered:
            score -= 1.0
    if score > 3:
        return 1.0
    if score < -3:
        return -1.0
    return round(score / 3.0, 3)


def _average_sentiment(items: list[NewsItem]) -> float:
    if not items:
        return 0.0
    return sum(item.sentiment for item in items) / len(items)


def _top_topics(items: list[NewsItem], limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for item in items:
        for topic in item.topics:
            counts[topic] = counts.get(topic, 0) + 1
    sorted_topics = sorted(counts.items(), key=lambda t: t[1], reverse=True)
    return [topic for topic, _ in sorted_topics[:limit]]


def _dedupe_news(items: list[NewsItem]) -> list[NewsItem]:
    deduped: list[NewsItem] = []
    seen: set[str] = set()
    for item in items:
        key = item.link or item.title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _is_configured_secret(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in SECRET_PLACEHOLDER_MARKERS)
