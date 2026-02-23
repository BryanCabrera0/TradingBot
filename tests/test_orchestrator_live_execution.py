import unittest
from datetime import datetime
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_live_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "live"
    cfg.scanner.enabled = False
    cfg.news.enabled = False
    cfg.llm.enabled = False
    cfg.credit_spreads.enabled = True
    cfg.covered_calls.enabled = False
    cfg.iron_condors.enabled = False
    return cfg


class OrchestratorLiveExecutionTests(unittest.TestCase):
    def test_live_entry_registers_pending_ledger_position(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.schwab.build_bull_put_spread = mock.Mock(return_value={"order": "spec"})
        bot.schwab.place_order = mock.Mock(
            return_value={"order_id": "abc123", "status": "PLACED"}
        )

        signal = TradeSignal(
            action="open",
            strategy="bull_put_spread",
            symbol="SPY",
            quantity=2,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="bull_put_spread",
                expiration="2026-03-20",
                dte=25,
                short_strike=100,
                long_strike=95,
                credit=1.4,
                max_loss=3.6,
                probability_of_profit=0.63,
                score=58,
            ),
        )

        opened = bot._execute_live_entry(
            signal,
            details={
                "expiration": "2026-03-20",
                "short_strike": 100,
                "long_strike": 95,
            },
        )

        self.assertTrue(opened)
        bot.live_ledger.register_entry_order.assert_called_once()

    def test_live_exit_places_close_order_and_marks_closing(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.get_position.return_value = {
            "position_id": "live_1",
            "strategy": "bull_put_spread",
            "symbol": "SPY",
            "status": "open",
            "quantity": 1,
            "entry_credit": 1.25,
            "current_value": 0.55,
            "details": {
                "expiration": "2026-03-20",
                "short_strike": 100,
                "long_strike": 95,
            },
        }
        bot.schwab.build_bull_put_spread_close = mock.Mock(return_value={"order": "close"})
        bot.schwab.place_order = mock.Mock(
            return_value={"order_id": "close123", "status": "PLACED"}
        )

        signal = TradeSignal(
            action="close",
            strategy="bull_put_spread",
            symbol="SPY",
            position_id="live_1",
            reason="profit target",
        )

        closed = bot._execute_live_exit(signal)

        self.assertTrue(closed)
        bot.schwab.build_bull_put_spread_close.assert_called_once()
        bot.live_ledger.register_exit_order.assert_called_once_with(
            position_id="live_1",
            exit_order_id="close123",
            reason="profit target",
        )

    def test_live_iron_condor_entry_uses_condor_builder(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.schwab.build_iron_condor = mock.Mock(return_value={"order": "condor"})
        bot.schwab.place_order = mock.Mock(
            return_value={"order_id": "condor1", "status": "PLACED"}
        )

        signal = TradeSignal(
            action="open",
            strategy="iron_condor",
            symbol="SPY",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="SPY",
                strategy="iron_condor",
                expiration="2026-03-20",
                dte=30,
                short_strike=0,
                long_strike=0,
                put_short_strike=95,
                put_long_strike=90,
                call_short_strike=110,
                call_long_strike=115,
                credit=1.1,
                max_loss=3.9,
                probability_of_profit=0.61,
                score=57,
            ),
        )

        opened = bot._execute_live_entry(signal, details={"expiration": "2026-03-20"})

        self.assertTrue(opened)
        bot.schwab.build_iron_condor.assert_called_once()

    def test_reconcile_cancels_stale_entry_orders(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.pending_entry_order_ids.return_value = ["stale-order"]
        bot.live_ledger.pending_exit_order_ids.return_value = []
        bot.live_ledger.get_position_by_order_id.return_value = {
            "entry_order_time": "2026-02-23T09:00:00-05:00"
        }
        bot.schwab.get_order = mock.Mock(
            return_value={"status": "WORKING", "enteredTime": "2026-02-23T09:00:00-05:00"}
        )
        bot.schwab.cancel_order = mock.Mock()
        bot._now_eastern = mock.Mock(
            return_value=datetime.fromisoformat("2026-02-23T09:30:00-05:00")
        )
        bot.config.execution.stale_order_minutes = 5
        bot.config.execution.cancel_stale_orders = True

        bot._reconcile_live_orders()

        bot.schwab.cancel_order.assert_called_once_with("stale-order")
        bot.live_ledger.reconcile_entry_order.assert_called_with(
            "stale-order",
            status="CANCELED",
        )

    def test_bootstrap_imports_existing_broker_spread_position(self) -> None:
        bot = TradingBot(make_live_config())
        bot.live_ledger = mock.Mock()
        bot.live_ledger.list_positions.return_value = []
        bot.schwab.get_positions = mock.Mock(
            return_value=[
                {
                    "instrument": {"assetType": "OPTION", "symbol": "SPY  260320P00100000"},
                    "longQuantity": 0.0,
                    "shortQuantity": 1.0,
                },
                {
                    "instrument": {"assetType": "OPTION", "symbol": "SPY  260320P00095000"},
                    "longQuantity": 1.0,
                    "shortQuantity": 0.0,
                },
            ]
        )

        imported = bot._bootstrap_live_ledger_from_broker()

        self.assertEqual(imported, 1)
        bot.live_ledger.register_entry_order.assert_called_once()


if __name__ == "__main__":
    unittest.main()
