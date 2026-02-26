import unittest
from unittest import mock

from bot.config import BotConfig
from bot.orchestrator import TradingBot


def make_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.llm.enabled = True
    cfg.news.enabled = False
    cfg.llm.adversarial_review_enabled = True
    cfg.llm.adversarial_loss_threshold_pct = 0.50
    return cfg


class AdversarialReviewTests(unittest.TestCase):
    def test_orchestrator_adversarial_review_generates_exit_signal(self) -> None:
        bot = TradingBot(make_config())
        bot.llm_advisor = mock.Mock()
        bot.llm_advisor.adversarial_review_position.return_value = {
            "should_exit": True,
            "close_conviction": 82,
            "hold_conviction": 54,
        }
        positions = [
            {
                "position_id": "p1",
                "status": "open",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "entry_credit": 1.0,
                "current_value": 3.0,
                "max_loss": 4.0,
                "quantity": 1,
            }
        ]

        signals = bot._apply_adversarial_llm_reviews(positions)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertEqual(signals[0].position_id, "p1")

    def test_orchestrator_adversarial_review_skips_small_loss(self) -> None:
        bot = TradingBot(make_config())
        bot.llm_advisor = mock.Mock()
        positions = [
            {
                "position_id": "p1",
                "status": "open",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "entry_credit": 1.0,
                "current_value": 1.2,
                "max_loss": 4.0,
                "quantity": 1,
            }
        ]

        signals = bot._apply_adversarial_llm_reviews(positions)

        self.assertEqual(signals, [])
        bot.llm_advisor.adversarial_review_position.assert_not_called()


if __name__ == "__main__":
    unittest.main()
