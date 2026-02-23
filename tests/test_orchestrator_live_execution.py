import unittest
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


if __name__ == "__main__":
    unittest.main()
