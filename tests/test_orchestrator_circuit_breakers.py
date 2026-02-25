import unittest
from datetime import datetime, timedelta
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
    def test_vix_above_35_halts_entries(self) -> None:
        bot = TradingBot(_config())
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 38.0}})

        bot._update_market_regime()

        self.assertEqual(bot.circuit_state["regime"], "crisis")
        self.assertTrue(bot.circuit_state["halt_entries"])
        self.assertFalse(bot._entries_allowed())

    def test_vix_25_to_35_reduces_positions(self) -> None:
        bot = TradingBot(_config())
        starting_limit = bot.risk_manager.config.max_open_positions
        start_min_credit = bot.strategies[0].config.get("min_credit_pct")
        bot.schwab.get_quote = mock.Mock(return_value={"quote": {"lastPrice": 28.0}})

        bot._update_market_regime()

        self.assertEqual(bot.circuit_state["regime"], "elevated")
        self.assertEqual(bot.risk_manager.config.max_open_positions, int(round(starting_limit * 0.7)))
        self.assertAlmostEqual(bot.strategies[0].config.get("min_credit_pct"), round(start_min_credit * 1.2, 4))

    def test_three_consecutive_max_losses_pauses_24h(self) -> None:
        bot = TradingBot(_config())
        now = bot._now_eastern()
        today = now.date().isoformat()
        bot.paper_trader.closed_trades = [
            {"pnl": -200.0, "max_loss": 2.0, "quantity": 1, "close_date": f"{today}T10:00:00"},
            {"pnl": -200.0, "max_loss": 2.0, "quantity": 1, "close_date": f"{today}T11:00:00"},
            {"pnl": -200.0, "max_loss": 2.0, "quantity": 1, "close_date": f"{today}T12:00:00"},
        ]
        bot.risk_manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)

        bot._update_loss_breakers()

        pause_raw = bot.circuit_state.get("consecutive_loss_pause_until")
        self.assertIsNotNone(pause_raw)
        pause_dt = datetime.fromisoformat(str(pause_raw))
        self.assertGreaterEqual(pause_dt, now + timedelta(hours=23))
        self.assertLessEqual(pause_dt, now + timedelta(hours=25))

    def test_weekly_loss_exceeds_5pct_pauses_to_monday(self) -> None:
        bot = TradingBot(_config())
        now = bot._now_eastern()
        today = now.date().isoformat()
        bot.paper_trader.closed_trades = [
            {"pnl": -3000.0, "max_loss": 30.0, "quantity": 1, "close_date": f"{today}T10:00:00"},
            {"pnl": -2600.0, "max_loss": 26.0, "quantity": 1, "close_date": f"{today}T11:00:00"},
        ]
        bot.risk_manager.update_portfolio(account_balance=100_000, open_positions=[], daily_pnl=0.0)

        bot._update_loss_breakers()

        pause_raw = bot.circuit_state.get("weekly_loss_pause_until")
        self.assertIsNotNone(pause_raw)
        pause_dt = datetime.fromisoformat(str(pause_raw))
        self.assertEqual(pause_dt.weekday(), 0)
        self.assertGreater(pause_dt, now)


if __name__ == "__main__":
    unittest.main()
