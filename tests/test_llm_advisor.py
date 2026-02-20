import json
import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import LLMConfig
from bot.llm_advisor import LLMAdvisor
from bot.strategies.base import TradeSignal


def make_signal() -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        quantity=2,
        analysis=SpreadAnalysis(
            symbol="SPY",
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


class LLMAdvisorTests(unittest.TestCase):
    def test_review_trade_parses_structured_json(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(
            return_value=(
                '{"approve":false,"confidence":0.82,'
                '"risk_adjustment":0.6,"reason":"volatility spike"}'
            )
        )

        decision = advisor.review_trade(make_signal(), {"account_balance": 100000})

        self.assertFalse(decision.approve)
        self.assertAlmostEqual(decision.confidence, 0.82, places=2)
        self.assertAlmostEqual(decision.risk_adjustment, 0.6, places=2)
        self.assertEqual(decision.reason, "volatility spike")

    def test_review_trade_handles_invalid_response(self) -> None:
        advisor = LLMAdvisor(LLMConfig(enabled=True, provider="ollama"))
        advisor._query_model = mock.Mock(return_value="not-json")

        decision = advisor.review_trade(make_signal(), None)

        self.assertTrue(decision.approve)
        self.assertEqual(decision.confidence, 0.0)
        self.assertEqual(decision.risk_adjustment, 1.0)
        self.assertIn("missing reason", decision.reason.lower())

    def test_prompt_includes_conservative_policy_thresholds(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="ollama", risk_style="conservative")
        )
        prompt = advisor._build_prompt(make_signal(), {"account_balance": 100000})
        payload = json.loads(prompt)

        self.assertEqual(payload["risk_style"], "conservative")
        self.assertEqual(payload["risk_policy"]["min_probability_of_profit"], 0.65)
        self.assertEqual(payload["risk_policy"]["min_score"], 60.0)

    def test_invalid_risk_style_falls_back_to_moderate(self) -> None:
        advisor = LLMAdvisor(
            LLMConfig(enabled=True, provider="ollama", risk_style="unknown-style")
        )
        prompt = advisor._build_prompt(make_signal(), None)
        payload = json.loads(prompt)

        self.assertEqual(payload["risk_style"], "moderate")


if __name__ == "__main__":
    unittest.main()
