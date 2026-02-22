"""News intelligence scanner for symbol-specific and macro market headlines."""

import logging
import time
from dataclasses import asdict, dataclass
from urllib.parse import quote_plus

import requests
from defusedxml import ElementTree as DET

from bot.config import NewsConfig

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS_SEARCH_URL = "https://news.google.com/rss/search"
MAX_RSS_PAYLOAD_CHARS = 1_000_000

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
        """Convert to compact dict for LLM context payloads."""
        data = asdict(self)
        data["title"] = data["title"][:180]
        return data


class NewsScanner:
    """Fetches and summarizes relevant market and symbol news."""

    def __init__(self, config: NewsConfig):
        self.config = config
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}

    def build_context(self, symbol: str) -> dict:
        """Return a structured news context payload for the requested symbol."""
        symbol_news = self.get_symbol_news(symbol) if self.config.include_symbol_news else []
        market_news = self.get_market_news() if self.config.include_market_news else []

        return {
            "symbol": symbol.upper(),
            "symbol_headlines": [item.to_context() for item in symbol_news],
            "market_headlines": [item.to_context() for item in market_news],
            "symbol_sentiment": round(_average_sentiment(symbol_news), 3),
            "market_sentiment": round(_average_sentiment(market_news), 3),
            "dominant_market_topics": _top_topics(market_news, limit=5),
        }

    def get_symbol_news(self, symbol: str) -> list[NewsItem]:
        """Fetch ticker-relevant headlines with cache support."""
        symbol_norm = symbol.strip().upper()
        cache_key = f"symbol:{symbol_norm}"
        return self._with_cache(
            cache_key,
            lambda: self._fetch_symbol_news(symbol_norm),
        )

    def get_market_news(self) -> list[NewsItem]:
        """Fetch broad market headlines with cache support."""
        return self._with_cache("market", self._fetch_market_news)

    def _with_cache(self, key: str, loader) -> list[NewsItem]:
        """Return cached results when fresh; otherwise refresh."""
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
        queries = [
            f"{symbol} stock",
            f"{symbol} earnings",
            f"{symbol} options",
        ]
        per_query_limit = max(2, self.config.max_symbol_headlines // max(len(queries), 1))
        items: list[NewsItem] = []
        for query in queries:
            items.extend(self._fetch_google_news(query, per_query_limit))
            if len(items) >= self.config.max_symbol_headlines * 2:
                break
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
        """Fetch headlines from Google News RSS search endpoint."""
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
                logger.warning(
                    "Skipping oversized RSS response for query %r (%d chars).",
                    query,
                    len(payload),
                )
                return []
            return _parse_rss_items(payload, limit)
        except Exception as exc:
            logger.debug("Failed to fetch news for query %r: %s", query, exc)
            return []


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

        headline_sentiment = _score_headline_sentiment(title)
        items.append(
            NewsItem(
                title=title,
                link=link,
                published=published,
                source=source,
                sentiment=headline_sentiment,
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
    """Return a lightweight sentiment score in [-1.0, 1.0] from headline keywords."""
    lowered = title.lower()
    score = 0.0
    for keyword in POSITIVE_KEYWORDS:
        if keyword in lowered:
            score += 1.0
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in lowered:
            score -= 1.0

    # Normalize to a bounded range while still reflecting intensity.
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
    """De-duplicate headlines by link/title while preserving order."""
    deduped: list[NewsItem] = []
    seen: set[str] = set()
    for item in items:
        key = item.link or item.title.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
