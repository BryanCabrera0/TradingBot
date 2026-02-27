import unittest
from unittest import mock

from rich.layout import Layout

from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.terminal_ui import TerminalUI


def make_config(*, max_events: int = 50) -> BotConfig:
    cfg = BotConfig()
    cfg.terminal_ui.enabled = True
    cfg.terminal_ui.refresh_rate = 0.5
    cfg.terminal_ui.max_activity_events = max_events
    cfg.terminal_ui.show_rejected_trades = True
    cfg.terminal_ui.compact_mode = False
    cfg.scanner.enabled = False
    cfg.llm.enabled = False
    cfg.news.enabled = False
    return cfg


class TerminalUITests(unittest.TestCase):
    def test_instantiation_with_config(self) -> None:
        ui = TerminalUI(make_config())
        self.assertEqual(ui.refresh_rate, 0.5)
        self.assertEqual(ui.max_activity_events, 50)

    def test_update_portfolio_stores_state(self) -> None:
        ui = TerminalUI(make_config())
        ui.update_portfolio(
            balance=123456.78,
            buying_power=90000.0,
            open_count=3,
            max_positions=10,
            daily_pnl=111.22,
            daily_risk_pct=1.5,
            max_daily_risk_pct=3.0,
            greeks={"delta": 12.3, "theta": -8.5, "vega": -100.0, "gamma": 0.4},
        )
        self.assertEqual(ui._portfolio["balance"], 123456.78)
        self.assertEqual(ui._portfolio["open_count"], 3)
        self.assertEqual(ui._portfolio["greeks"]["delta"], 12.3)

    def test_update_metrics_stores_state(self) -> None:
        ui = TerminalUI(make_config())
        ui.update_metrics(
            sharpe=1.2,
            sortino=1.8,
            calmar=0.9,
            win_rate=0.6,
            wins=6,
            total=10,
            profit_factor=1.7,
            max_drawdown=0.03,
            expectancy=22.5,
            today_pnl=100.0,
            today_pnl_pct=0.1,
            week_pnl=300.0,
        )
        self.assertEqual(ui._metrics["wins"], 6)
        self.assertEqual(ui._metrics["total"], 10)
        self.assertEqual(ui._metrics["profit_factor"], 1.7)

    def test_update_positions_handles_empty_single_and_many(self) -> None:
        ui = TerminalUI(make_config())
        ui.update_positions([])
        self.assertEqual(len(ui._positions), 0)

        ui.update_positions(
            [
                {
                    "symbol": "SPY",
                    "strategy": "credit_spreads",
                    "qty": 1,
                    "dte": 20,
                    "entry_price": 1.2,
                    "current_price": 0.8,
                    "pnl": 40,
                    "pct_of_max_profit": 33,
                    "delta": -0.05,
                }
            ]
        )
        self.assertEqual(len(ui._positions), 1)
        self.assertEqual(ui._positions[0]["symbol"], "SPY")

        positions = []
        for idx in range(10):
            positions.append(
                {
                    "symbol": f"T{idx}",
                    "strategy": "credit_spreads",
                    "qty": 1,
                    "dte": 10 - idx,
                    "entry_price": 1.0,
                    "current_price": 0.9,
                    "pnl": 10,
                    "pct_of_max_profit": 10,
                    "delta": 0.01,
                }
            )
        ui.update_positions(positions)
        self.assertEqual(len(ui._positions), 10)
        self.assertLessEqual(ui._positions[0]["dte"], ui._positions[-1]["dte"])

    def test_add_event_respects_maxlen(self) -> None:
        ui = TerminalUI(make_config(max_events=3))
        for i in range(6):
            ui.add_event("warning", f"event-{i}")
        self.assertEqual(len(ui._events), 3)
        self.assertTrue(any(row["message"] == "event-5" for row in ui._events))

    def test_build_layout_returns_layout(self) -> None:
        ui = TerminalUI(make_config())
        ui.add_event("opened", "OPENED credit spread SPY")
        layout = ui._build_layout()
        self.assertIsInstance(layout, Layout)

    def test_event_type_mapping_contains_all_supported_types(self) -> None:
        mapping = TerminalUI.event_mapping()
        expected = {
            "opened",
            "closed_profit",
            "closed_loss",
            "rejected",
            "rolled",
            "adjusted",
            "hedged",
            "llm",
            "regime",
            "warning",
            "circuit_breaker",
            "paused",
            "resumed",
        }
        self.assertTrue(expected.issubset(set(mapping.keys())))

    def test_color_logic_for_pnl_and_dte_thresholds(self) -> None:
        ui = TerminalUI(make_config())
        self.assertEqual(ui._pnl_color(10.0), "bright_green")
        self.assertEqual(ui._pnl_color(-1.0), "#ff6b6b")
        self.assertEqual(ui._dte_color(5), "#ff6b6b")
        self.assertEqual(ui._dte_color(10), "yellow")
        self.assertEqual(ui._dte_color(25), "white")

    def test_orchestrator_ui_update_noop_when_ui_missing(self) -> None:
        cfg = BotConfig()
        cfg.scanner.enabled = False
        cfg.llm.enabled = False
        cfg.news.enabled = False
        cfg.terminal_ui.enabled = False
        bot = TradingBot(cfg)
        bot.ui = None
        bot._ui_update("update_system_status", scanner="ok")
        self.assertIsNone(bot.ui)

    def test_start_and_stop_do_not_raise(self) -> None:
        ui = TerminalUI(make_config())

        def fake_loop() -> None:
            ui._stop_event.wait(timeout=0.05)

        with mock.patch.object(ui, "_install_logging_bridge", return_value=None), mock.patch.object(
            ui,
            "_remove_logging_bridge",
            return_value=None,
        ), mock.patch.object(ui, "_run_live_loop", side_effect=fake_loop):
            ui.start()
            ui.stop()


if __name__ == "__main__":
    unittest.main()
