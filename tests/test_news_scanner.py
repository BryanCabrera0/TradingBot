import unittest
from unittest import mock

from bot.config import NewsConfig
from bot.news_scanner import NewsScanner, _parse_rss_items


RSS_SAMPLE = """\
<rss version="2.0">
  <channel>
    <item>
      <title>Apple beats earnings estimates, shares rally</title>
      <link>https://example.com/aapl-earnings</link>
      <pubDate>Fri, 20 Feb 2026 14:30:00 GMT</pubDate>
      <source>Example News</source>
    </item>
    <item>
      <title>Fed warns inflation remains sticky</title>
      <link>https://example.com/fed-inflation</link>
      <pubDate>Fri, 20 Feb 2026 15:00:00 GMT</pubDate>
      <source>Example Macro</source>
    </item>
  </channel>
</rss>
"""


class NewsScannerTests(unittest.TestCase):
    def test_parse_rss_items_extracts_structure_and_sentiment(self) -> None:
        items = _parse_rss_items(RSS_SAMPLE, limit=5)

        self.assertEqual(len(items), 2)
        self.assertIn("Apple beats earnings", items[0].title)
        self.assertGreater(items[0].sentiment, 0)
        self.assertIn("inflation", items[1].topics)

    def test_news_scanner_builds_context_and_uses_cache(self) -> None:
        config = NewsConfig(
            enabled=True,
            cache_seconds=3600,
            max_symbol_headlines=2,
            max_market_headlines=2,
            market_queries=["stock market"],
        )
        scanner = NewsScanner(config)

        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.text = RSS_SAMPLE

        with mock.patch("bot.news_scanner.requests.get", return_value=response) as get:
            first = scanner.build_context("AAPL")
            second = scanner.build_context("AAPL")

        self.assertEqual(first["symbol"], "AAPL")
        self.assertGreaterEqual(len(first["symbol_headlines"]), 1)
        self.assertIn("market_sentiment", first)
        self.assertEqual(first, second)
        # First call: 2 symbol queries (early stop) + 1 market query.
        # Second call should be fully cached.
        self.assertEqual(get.call_count, 3)

    def test_trade_direction_policy_blocks_bull_put_on_bearish_signal(self) -> None:
        scanner = NewsScanner(NewsConfig(enabled=True))
        policy = scanner.trade_direction_policy(
            "AAPL",
            sentiment={"sentiment": "bearish", "confidence": 88, "key_event": None},
        )

        self.assertFalse(policy["allow_bull_put"])
        self.assertTrue(policy["allow_bear_call"])


if __name__ == "__main__":
    unittest.main()
