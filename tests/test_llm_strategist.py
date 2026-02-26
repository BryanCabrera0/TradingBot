import os
import json
import unittest
from unittest import mock

from bot.config import LLMStrategistConfig
from bot.llm_strategist import LLMStrategist


class LLMStrategistTests(unittest.TestCase):
    def test_review_portfolio_parses_directives(self) -> None:
        strategist = LLMStrategist(LLMStrategistConfig(enabled=True, provider="ollama", max_directives=2))
        strategist._query = mock.Mock(
            return_value='{"directives":[{"action":"scale_size","confidence":80,"reason":"vol rising","payload":{"scalar":0.8}},{"action":"skip_sector","confidence":75,"reason":"crowded","payload":{"sector":"Information Technology"}},{"action":"none","confidence":50,"reason":"n/a","payload":{}}]}'
        )

        directives = strategist.review_portfolio({"portfolio_greeks": {}})

        self.assertEqual(len(directives), 2)
        self.assertEqual(directives[0].action, "scale_size")

    def test_invalid_json_returns_empty(self) -> None:
        strategist = LLMStrategist(LLMStrategistConfig(enabled=True, provider="ollama"))
        strategist._query = mock.Mock(return_value="not json")
        directives = strategist.review_portfolio({"portfolio_greeks": {}})
        self.assertEqual(directives, [])

    def test_openai_provider_calls_request_helper(self) -> None:
        strategist = LLMStrategist(LLMStrategistConfig(enabled=True, provider="openai", model="gpt-4.1"))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch("bot.llm_strategist.request_openai_json", return_value='{"directives":[]}') as req:
                directives = strategist.review_portfolio({"portfolio_greeks": {}})
        self.assertEqual(directives, [])
        self.assertTrue(req.called)

    def test_openai_prompt_uses_enhanced_directives_and_temperature(self) -> None:
        strategist = LLMStrategist(LLMStrategistConfig(enabled=True, provider="openai", model="gpt-4.1"))
        context = {
            "portfolio_greeks": {"delta": 35},
            "recent_regime_states": [{"regime": "BULL_TREND"}],
            "strategy_streaks": {"credit_spreads": {"streak": 3, "type": "losses"}},
            "top_risk_contributors": [{"symbol": "SPY", "risk_dollars": 1200}],
        }
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            with mock.patch("bot.llm_strategist.request_openai_json", return_value='{"directives":[]}') as req:
                strategist.review_portfolio(context)

        kwargs = req.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.2)
        user_prompt = json.loads(kwargs["user_prompt"])
        self.assertIn("directive_categories", user_prompt)
        self.assertIn("examples", user_prompt)
        self.assertIn("recent_regime_states", user_prompt["context"])

    def test_openai_missing_key_raises(self) -> None:
        strategist = LLMStrategist(LLMStrategistConfig(enabled=True, provider="openai"))
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                strategist.review_portfolio({"portfolio_greeks": {}})

    def test_google_provider_calls_generate_content(self) -> None:
        strategist = LLMStrategist(
            LLMStrategistConfig(enabled=True, provider="google", model="gemini-2.5-pro")
        )
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": '{"directives":[]}'}],
                    }
                }
            ]
        }
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with mock.patch("bot.llm_strategist.requests.post", return_value=response) as post:
                directives = strategist.review_portfolio({"portfolio_greeks": {}})

        self.assertEqual(directives, [])
        args, kwargs = post.call_args
        self.assertIn(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            args[0],
        )
        self.assertEqual(kwargs["params"]["key"], "test-key")


if __name__ == "__main__":
    unittest.main()
