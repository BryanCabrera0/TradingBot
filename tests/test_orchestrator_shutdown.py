import json
import signal
import tempfile
import unittest
from pathlib import Path
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
        bot.generate_dashboard = mock.Mock(return_value="logs/dashboard.html")
        bot.schwab.get_orders = mock.Mock(return_value=[])
        bot.schwab.get_order = mock.Mock(return_value={"status": "CANCELED"})
        bot.live_ledger = mock.Mock(
            pending_entry_order_ids=mock.Mock(return_value=[]),
            pending_exit_order_ids=mock.Mock(return_value=[]),
            save=mock.Mock(),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            bot._runtime_state_path = Path(tmp_dir) / "runtime_state.json"

            bot._handle_shutdown(signal.SIGTERM, None)

        self.assertFalse(bot._running)

    def test_shutdown_cancels_pending_orders(self) -> None:
        bot = TradingBot(_live_config())
        bot._running = True
        bot.schwab.cancel_order = mock.Mock()
        bot.schwab.get_orders = mock.Mock(return_value=[])
        bot.schwab.get_order = mock.Mock(return_value={"status": "CANCELED"})
        bot.generate_dashboard = mock.Mock(return_value="logs/dashboard.html")
        bot.live_ledger = mock.Mock(
            pending_entry_order_ids=mock.Mock(return_value=["e1", "e2"]),
            pending_exit_order_ids=mock.Mock(return_value=["x1"]),
            save=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            bot._runtime_state_path = Path(tmp_dir) / "runtime_state.json"
            bot._handle_shutdown(signal.SIGTERM, None)

        self.assertEqual(bot.schwab.cancel_order.call_count, 2)
        bot.schwab.cancel_order.assert_any_call("e1")
        bot.schwab.cancel_order.assert_any_call("e2")

    def test_shutdown_cancels_broker_working_orders_and_sets_clean_flag(self) -> None:
        bot = TradingBot(_live_config())
        bot._running = True
        bot.schwab.cancel_order = mock.Mock()
        bot.schwab.get_orders = mock.Mock(
            return_value=[
                {"orderId": "w1", "status": "WORKING", "orderLegCollection": [{"instruction": "BUY_TO_OPEN"}]},
                {"orderId": "f1", "status": "FILLED"},
                {"orderId": "w2", "status": "PENDING_CANCEL", "orderLegCollection": [{"instruction": "SELL_TO_OPEN"}]},
            ]
        )
        bot.schwab.get_order = mock.Mock(return_value={"status": "CANCELED"})
        bot.generate_dashboard = mock.Mock(return_value="logs/dashboard.html")
        bot.live_ledger = mock.Mock(
            pending_entry_order_ids=mock.Mock(return_value=[]),
            pending_exit_order_ids=mock.Mock(return_value=[]),
            save=mock.Mock(),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "runtime_state.json"
            bot._runtime_state_path = state_path

            bot._handle_shutdown(signal.SIGTERM, None)

            bot.schwab.cancel_order.assert_any_call("w1")
            bot.schwab.cancel_order.assert_any_call("w2")
            self.assertNotIn(mock.call("f1"), bot.schwab.cancel_order.mock_calls)
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(payload.get("clean_shutdown"))

    def test_no_warning_when_runtime_state_file_missing(self) -> None:
        bot = TradingBot(_live_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            bot._runtime_state_path = Path(tmp_dir) / "runtime_state.json"
            with mock.patch("bot.orchestrator.logger.warning") as warning_mock:
                bot._warn_if_unclean_previous_shutdown()
            warning_mock.assert_not_called()

    def test_warns_when_clean_shutdown_flag_is_false(self) -> None:
        bot = TradingBot(_live_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "runtime_state.json"
            state_path.write_text(
                json.dumps({"clean_shutdown": False}),
                encoding="utf-8",
            )
            bot._runtime_state_path = state_path
            with self.assertLogs("bot.orchestrator", level="WARNING") as captured:
                bot._warn_if_unclean_previous_shutdown()
            self.assertTrue(
                any(
                    "did not shut down cleanly" in line.lower()
                    for line in captured.output
                )
            )


if __name__ == "__main__":
    unittest.main()
