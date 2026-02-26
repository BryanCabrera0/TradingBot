import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
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


def make_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.llm.enabled = False
    cfg.news.enabled = False
    cfg.multi_timeframe.enabled = True
    cfg.multi_timeframe.min_agreement = 2
    cfg.credit_spreads.enabled = True
    cfg.covered_calls.enabled = False
    cfg.iron_condors.enabled = False
    return cfg


class MultiTimeframeTests(unittest.TestCase):
    def test_passes_multi_timeframe_confirmation_threshold(self) -> None:
        bot = TradingBot(make_config())
        bot._compute_multi_timeframe_votes = mock.Mock(
            return_value={"daily": True, "weekly": False, "hourly": True}
        )

        ok, agreement, votes = bot._passes_multi_timeframe_confirmation(make_signal())

        self.assertTrue(ok)
        self.assertEqual(agreement, 2)
        self.assertEqual(votes["daily"], True)

    def test_try_execute_entry_rejects_on_low_timeframe_agreement(self) -> None:
        bot = TradingBot(make_config())
        bot._passes_multi_timeframe_confirmation = mock.Mock(
            return_value=(False, 1, {"daily": True, "weekly": False, "hourly": False})
        )
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))

        executed = bot._try_execute_entry(make_signal())

        self.assertFalse(executed)
        bot.risk_manager.approve_trade.assert_not_called()

    def test_try_execute_entry_allows_when_timeframes_agree(self) -> None:
        bot = TradingBot(make_config())
        bot._passes_multi_timeframe_confirmation = mock.Mock(
            return_value=(True, 2, {"daily": True, "weekly": False, "hourly": True})
        )
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=1)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot.risk_manager.register_open_position = mock.Mock()

        executed = bot._try_execute_entry(make_signal())

        self.assertTrue(executed)
        bot.risk_manager.approve_trade.assert_called_once()


if __name__ == "__main__":
    unittest.main()
