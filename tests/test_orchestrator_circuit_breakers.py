import unittest
from unittest import mock

from bot.config import BotConfig
from bot.orchestrator import TradingBot


def _config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.news.enabled = False
    cfg.llm.enabled = False
    cfg.iron_condors.enabled = False
    cfg.covered_calls.enabled = False
    cfg.credit_spreads.enabled = True
    return cfg


class OrchestratorCircuitBreakerTests(unittest.TestCase):
    def test_vix_crisis_halts_entries(self) -> None:
        bot = TradingBot(_config())
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 40.0}})

        bot._update_market_regime()

        self.assertEqual(bot.circuit_state["regime"], "crisis")
        self.assertTrue(bot.circuit_state["halt_entries"])

    def test_vix_elevated_reduces_open_positions(self) -> None:
        bot = TradingBot(_config())
        starting_limit = bot.risk_manager.config.max_open_positions
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 28.0}})

        bot._update_market_regime()

        self.assertEqual(bot.circuit_state["regime"], "elevated")
        self.assertLess(bot.risk_manager.config.max_open_positions, starting_limit)


if __name__ == "__main__":
    unittest.main()
