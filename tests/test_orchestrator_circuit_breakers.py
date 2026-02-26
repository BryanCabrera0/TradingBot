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

    def test_correlated_loss_breaker_closes_two_worst_and_pauses(self) -> None:
        bot = TradingBot(_config())
        positions = [
            {"position_id": "p1", "status": "open", "symbol": "AAPL", "strategy": "bull_put_spread", "quantity": 1, "entry_credit": 1.0, "current_value": 1.8},
            {"position_id": "p2", "status": "open", "symbol": "MSFT", "strategy": "bull_put_spread", "quantity": 1, "entry_credit": 1.0, "current_value": 1.7},
            {"position_id": "p3", "status": "open", "symbol": "NVDA", "strategy": "bull_put_spread", "quantity": 1, "entry_credit": 1.0, "current_value": 1.6},
            {"position_id": "p4", "status": "open", "symbol": "JPM", "strategy": "bull_put_spread", "quantity": 1, "entry_credit": 1.0, "current_value": 1.1},
        ]
        bot.alerts = mock.Mock()

        signals = bot._apply_correlated_loss_protection(positions)

        self.assertEqual(len(signals), 2)
        self.assertIn("correlated_loss_pause_until", bot.circuit_state)
        self.assertLessEqual(bot._cycle_size_scalar, 0.7)
        self.assertTrue(all(sig.action == "close" for sig in signals))

    def test_gamma_risk_flagging_and_expiration_day_force_close(self) -> None:
        bot = TradingBot(_config())
        positions = [
            {
                "position_id": "g1",
                "status": "open",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "quantity": 1,
                "dte_remaining": 4,
                "underlying_price": 99.0,
                "details": {"short_strike": 100.0},
            },
            {
                "position_id": "g2",
                "status": "open",
                "symbol": "QQQ",
                "strategy": "bear_call_spread",
                "quantity": 1,
                "dte_remaining": 0,
                "underlying_price": 103.0,
                "details": {"short_strike": 105.0},
            },
        ]

        exits = bot._apply_gamma_risk_controls(positions)

        self.assertTrue(positions[0]["gamma_risk_flag"])
        self.assertEqual(
            float(positions[0]["details"].get("stop_loss_override_multiple", 0.0)),
            bot.config.risk.gamma_week_tight_stop,
        )
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].position_id, "g2")

    def test_correlation_crisis_sets_pause_and_size_reduction(self) -> None:
        bot = TradingBot(_config())
        bot.correlation_monitor = mock.Mock(
            get_correlation_state=mock.Mock(
                return_value={
                    "correlation_regime": "crisis",
                    "correlations": {"SPY_QQQ": 0.99},
                    "flags": {"spy_vix_positive": True},
                }
            )
        )

        bot._update_correlation_state()

        self.assertEqual(bot.circuit_state.get("correlation_regime"), "crisis")
        self.assertAlmostEqual(float(bot.circuit_state.get("correlation_size_scalar")), 0.5, places=4)
        self.assertAlmostEqual(float(bot.circuit_state.get("correlation_stop_widen_scalar")), 1.25, places=4)
        self.assertFalse(bot._entries_allowed())

    def test_correlation_stressed_reduces_entry_size_without_halt(self) -> None:
        bot = TradingBot(_config())
        bot.correlation_monitor = mock.Mock(
            get_correlation_state=mock.Mock(
                return_value={
                    "correlation_regime": "stressed",
                    "correlations": {"SPY_QQQ": 0.90},
                    "flags": {"equity_corr_spike": True},
                }
            )
        )

        bot._update_correlation_state()

        self.assertEqual(bot.circuit_state.get("correlation_regime"), "stressed")
        self.assertAlmostEqual(float(bot.circuit_state.get("correlation_size_scalar")), 0.75, places=4)
        self.assertAlmostEqual(float(bot.circuit_state.get("correlation_stop_widen_scalar")), 1.0, places=4)
        self.assertTrue(bot._entries_allowed())


if __name__ == "__main__":
    unittest.main()
