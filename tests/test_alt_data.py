import json
import unittest
from unittest import mock

from bot.alt_data import AltDataEngine
from bot.analysis import SpreadAnalysis
from bot.config import AltDataConfig, LLMConfig
from bot.llm_advisor import LLMAdvisor
from bot.strategies.base import TradeSignal


def make_signal(symbol: str = "SPY") -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol=symbol,
        analysis=SpreadAnalysis(
            symbol=symbol,
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=95,
            long_strike=90,
            credit=1.25,
            max_loss=3.75,
            probability_of_profit=0.68,
            score=64,
        ),
    )


class AltDataTests(unittest.TestCase):
    def test_gex_flip_point_calculation(self) -> None:
        chain = {
            "underlying_price": 103.0,
            "calls": {
                "2026-03-20": [
                    {"strike": 100.0, "gamma": 0.02, "open_interest": 10},
                    {"strike": 105.0, "gamma": 0.02, "open_interest": 50},
                ]
            },
            "puts": {
                "2026-03-20": [
                    {"strike": 100.0, "gamma": 0.02, "open_interest": 40},
                    {"strike": 105.0, "gamma": 0.02, "open_interest": 10},
                ]
            },
        }

        signal = AltDataEngine.estimate_gex(chain, underlying_price=103.0)

        self.assertGreater(signal["gex_flip"], 100.0)
        self.assertLess(signal["gex_flip"], 105.0)
        self.assertEqual(signal["dealer_gamma_bias"], "long")
        self.assertGreaterEqual(signal["magnitude"], 0.0)

    def test_dark_pool_proxy_scoring(self) -> None:
        curr = {
            "calls": {"2026-03-20": [{"open_interest": 220, "volume": 120}]},
            "puts": {"2026-03-20": [{"open_interest": 80, "volume": 30}]},
        }
        prev = {
            "calls": {"2026-03-20": [{"open_interest": 150, "volume": 60}]},
            "puts": {"2026-03-20": [{"open_interest": 100, "volume": 45}]},
        }
        signal = AltDataEngine.estimate_dark_pool_proxy(
            curr,
            previous_chain_data=prev,
            flow_context={
                "directional_bias": "bullish",
                "institutional_flow_direction": "buying",
            },
        )

        self.assertGreater(signal["dark_pool_proxy_score"], 0.0)
        self.assertLessEqual(signal["dark_pool_proxy_score"], 1.0)

    def test_social_sentiment_caching(self) -> None:
        scanner = mock.Mock()
        scanner.get_symbol_sentiment = mock.Mock(
            return_value={
                "sentiment": "bearish",
                "score": -0.7,
                "catalyst": "FDA rejection risk",
                "confidence": 0.82,
            }
        )
        engine = AltDataEngine(
            AltDataConfig(
                social_sentiment_enabled=True,
                social_sentiment_cache_minutes=30,
                social_sentiment_model="gpt-5.2",
            ),
            news_scanner=scanner,
        )

        first = engine.get_social_sentiment("AAPL")
        second = engine.get_social_sentiment("AAPL")

        self.assertEqual(first, second)
        self.assertEqual(scanner.get_symbol_sentiment.call_count, 1)

    def test_alt_data_fields_present_in_debate_prompt_payload(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="openai"))
        payload = json.loads(
            advisor._build_prompt(
                make_signal(),
                {
                    "alt_data": {
                        "gex": {"gex_flip": 445.5},
                        "dark_pool_proxy": {"dark_pool_proxy_score": -0.2},
                        "social_sentiment": {"sentiment": "bearish", "score": -0.7},
                    },
                    "gex": {"gex_flip": 445.5},
                    "dark_pool_proxy": {"dark_pool_proxy_score": -0.2},
                    "social_sentiment": {"sentiment": "bearish", "score": -0.7},
                },
            )
        )

        sections = payload["sections"]
        self.assertIn("alt_data", sections)
        self.assertIn("gex", sections)
        self.assertIn("dark_pool_proxy", sections)
        self.assertIn("social_sentiment", sections)


if __name__ == "__main__":
    unittest.main()
