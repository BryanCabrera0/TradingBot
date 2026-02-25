import signal
import unittest
from unittest import mock

from bot.config import BotConfig
from bot.orchestrator import TradingBot


def _live_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "live"
    cfg.scanner.enabled = False
    cfg.news.enabled = False
    cfg.llm.enabled = False
    cfg.credit_spreads.enabled = True
    cfg.iron_condors.enabled = False
    cfg.covered_calls.enabled = False
    cfg.naked_puts.enabled = False
    cfg.calendar_spreads.enabled = False
    return cfg


class OrchestratorShutdownTests(unittest.TestCase):
    def test_shutdown_sets_running_false(self) -> None:
        bot = TradingBot(_live_config())
        bot._running = True
        bot.live_ledger = mock.Mock(
            pending_entry_order_ids=mock.Mock(return_value=[]),
            pending_exit_order_ids=mock.Mock(return_value=[]),
            save=mock.Mock(),
        )

        bot._handle_shutdown(signal.SIGTERM, None)

        self.assertFalse(bot._running)

    def test_shutdown_cancels_pending_orders(self) -> None:
        bot = TradingBot(_live_config())
        bot._running = True
        bot.schwab.cancel_order = mock.Mock()
        bot.live_ledger = mock.Mock(
            pending_entry_order_ids=mock.Mock(return_value=["e1", "e2"]),
            pending_exit_order_ids=mock.Mock(return_value=["x1"]),
            save=mock.Mock(),
        )

        bot._handle_shutdown(signal.SIGTERM, None)

        self.assertEqual(bot.schwab.cancel_order.call_count, 3)
        bot.schwab.cancel_order.assert_any_call("e1")
        bot.schwab.cancel_order.assert_any_call("e2")
        bot.schwab.cancel_order.assert_any_call("x1")


if __name__ == "__main__":
    unittest.main()
