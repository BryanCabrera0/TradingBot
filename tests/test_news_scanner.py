import os
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

    def test_bearish_sentiment_blocks_bull_puts(self) -> None:
        scanner = NewsScanner(NewsConfig(enabled=True))
        policy = scanner.trade_direction_policy(
            "AAPL",
            sentiment={"sentiment": "bearish", "confidence": 80, "key_event": None},
        )
        self.assertFalse(policy["allow_bull_put"])
        self.assertTrue(policy["allow_bear_call"])

    def test_bullish_sentiment_blocks_bear_calls(self) -> None:
        scanner = NewsScanner(NewsConfig(enabled=True))
        policy = scanner.trade_direction_policy(
            "AAPL",
            sentiment={"sentiment": "bullish", "confidence": 80, "key_event": None},
        )
        self.assertTrue(policy["allow_bull_put"])
        self.assertFalse(policy["allow_bear_call"])

    def test_binary_event_blocks_all_entries(self) -> None:
        scanner = NewsScanner(NewsConfig(enabled=True))
        policy = scanner.trade_direction_policy(
            "AAPL",
            sentiment={
                "sentiment": "bullish",
                "confidence": 75,
                "key_event": "FDA approval",
            },
        )
        self.assertTrue(policy["block_all"])
        self.assertFalse(policy["allow_bull_put"])
        self.assertFalse(policy["allow_bear_call"])

    def test_finnhub_news_parsing(self) -> None:
        scanner = NewsScanner(
            NewsConfig(
                enabled=True, finnhub_api_key="test-key", request_timeout_seconds=5
            )
        )
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [
            {
                "headline": "AAPL wins major contract",
                "url": "https://example.com/1",
                "source": "Finnhub",
                "datetime": 1760000000,
            },
            {
                "headline": "AAPL faces lawsuit risk",
                "url": "https://example.com/2",
                "source": "Finnhub",
                "datetime": 1760000500,
            },
            {
                "headline": "AAPL guidance unchanged",
                "url": "https://example.com/3",
                "source": "Finnhub",
                "datetime": 1760001000,
            },
        ]

        with mock.patch("bot.news_scanner.requests.get", return_value=response):
            items = scanner._fetch_finnhub_news("AAPL", limit=3)

        self.assertEqual(len(items), 3)
        self.assertEqual(items[0].source, "Finnhub")
        self.assertTrue(items[0].link.startswith("https://example.com/"))

    def test_llm_sentiment_uses_gemini_generate_content(self) -> None:
        scanner = NewsScanner(
            NewsConfig(
                enabled=True,
                llm_sentiment_enabled=True,
                llm_sentiment_cache_seconds=0,
            )
        )
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"sentiment":"bullish","confidence":78,"key_event":null}'
                            }
                        ]
                    }
                }
            ]
        }

        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with mock.patch("bot.news_scanner.requests.post", return_value=response):
                sentiment = scanner.get_symbol_sentiment(
                    "AAPL",
                    headlines=[mock.Mock(title="AAPL jumps on guidance raise")],
                )

        self.assertEqual(sentiment["sentiment"], "bullish")

    def test_llm_sentiment_uses_gemini_payload(self) -> None:
        scanner = NewsScanner(
            NewsConfig(
                enabled=True,
                llm_sentiment_enabled=True,
                llm_sentiment_cache_seconds=0,
                llm_model="gemini-3.1-pro-thinking-preview",
                llm_reasoning_effort="medium",
                llm_text_verbosity="low",
                llm_max_output_tokens=256,
                llm_chat_fallback_model="gemini-3.1-flash-thinking-preview",
            )
        )
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"sentiment":"neutral","confidence":61,"key_event":null}'
                            }
                        ]
                    }
                }
            ]
        }

        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with mock.patch(
                "bot.news_scanner.requests.post", return_value=response
            ) as post:
                sentiment = scanner.get_symbol_sentiment(
                    "AAPL",
                    headlines=[mock.Mock(title="AAPL announces new AI roadmap")],
                )

        self.assertEqual(sentiment["sentiment"], "neutral")
        args, kwargs = post.call_args
        self.assertIn(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-thinking-preview:generateContent",
            args[0],
        )
        self.assertEqual(kwargs["params"]["key"], "test-key")
        payload = kwargs["json"]
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 256)
        self.assertEqual(
            payload["generationConfig"]["responseMimeType"], "application/json"
        )


if __name__ == "__main__":
    unittest.main()
