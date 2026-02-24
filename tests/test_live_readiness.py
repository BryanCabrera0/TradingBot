import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_live_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "live"
    cfg.watchlist = ["SPY"]
    cfg.scanner.enabled = False
    cfg.alerts.require_in_live = False
    return cfg


class LiveReadinessTests(unittest.TestCase):
    def test_market_hours_convert_to_eastern_time(self) -> None:
        pacific_time = datetime(2026, 1, 7, 7, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        self.assertTrue(TradingBot.is_market_open(pacific_time))

        pacific_after_close = datetime(
            2026, 1, 7, 14, 30, tzinfo=ZoneInfo("America/Los_Angeles")
        )
        self.assertFalse(TradingBot.is_market_open(pacific_after_close))

    def test_preflight_rejects_when_no_enabled_strategies(self) -> None:
        cfg = make_live_config()
        cfg.credit_spreads.enabled = False
        cfg.covered_calls.enabled = False
        cfg.iron_condors.enabled = False

        bot = TradingBot(cfg)

        with self.assertRaises(RuntimeError):
            bot.validate_live_readiness()

    def test_preflight_passes_with_supported_strategies_and_market_data(self) -> None:
        cfg = make_live_config()
        cfg.credit_spreads.enabled = True
        cfg.covered_calls.enabled = False
        cfg.iron_condors.enabled = False

        bot = TradingBot(cfg)
        bot.connect = mock.Mock()
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 100.0}})
        bot._get_chain_data = mock.Mock(
            return_value=({"underlying_price": 100.0, "calls": {"2026-03-20": []}}, 100.0)
        )

        bot.validate_live_readiness()

        bot.connect.assert_called_once()

    def test_live_covered_call_requires_held_shares(self) -> None:
        cfg = make_live_config()
        cfg.credit_spreads.enabled = False
        cfg.covered_calls.enabled = True
        cfg.iron_condors.enabled = False

        bot = TradingBot(cfg)
        bot._get_live_equity_shares = mock.Mock(return_value=0)
        bot.schwab.build_covered_call_open = mock.Mock()
        bot.schwab.place_order = mock.Mock()

        signal = TradeSignal(
            action="open",
            strategy="covered_call",
            symbol="AAPL",
            quantity=1,
            analysis=SpreadAnalysis(
                symbol="AAPL",
                strategy="covered_call",
                expiration="2026-03-20",
                dte=30,
                short_strike=210,
                long_strike=0,
                credit=1.2,
                max_loss=0,
                probability_of_profit=0.7,
                score=60,
            ),
        )

        executed = bot._execute_live_entry(signal)

        self.assertFalse(executed)
        bot.schwab.build_covered_call_open.assert_not_called()

    def test_live_preflight_requires_alert_destination_when_enforced(self) -> None:
        cfg = make_live_config()
        cfg.alerts.require_in_live = True
        cfg.alerts.enabled = False
        cfg.credit_spreads.enabled = True
        cfg.covered_calls.enabled = False
        cfg.iron_condors.enabled = False

        bot = TradingBot(cfg)

        with self.assertRaises(RuntimeError):
            bot.validate_live_readiness()


if __name__ == "__main__":
    unittest.main()
