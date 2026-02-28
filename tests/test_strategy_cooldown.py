import unittest
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def make_config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.llm.enabled = False
    cfg.news.enabled = False
    cfg.cooldown.graduated = True
    cfg.cooldown.level_1_losses = 2
    cfg.cooldown.level_1_reduction = 0.25
    cfg.cooldown.level_2_losses = 3
    cfg.cooldown.level_2_reduction = 0.50
    cfg.credit_spreads.enabled = True
    cfg.covered_calls.enabled = False
    cfg.iron_condors.enabled = False
    return cfg


def make_signal() -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        analysis=SpreadAnalysis(
            symbol="SPY",
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=95,
            long_strike=90,
            credit=1.2,
            max_loss=3.8,
            probability_of_profit=0.62,
            score=60,
        ),
    )


class StrategyCooldownTests(unittest.TestCase):
    def test_loss_streak_level1_applies_25pct_reduction(self) -> None:
        bot = TradingBot(make_config())
        bot._recent_closed_trades = mock.Mock(
            return_value=[
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -50.0,
                    "close_date": "2026-02-20",
                },
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -25.0,
                    "close_date": "2026-02-21",
                },
            ]
        )

        bot._refresh_strategy_and_symbol_breakers()

        state = bot._strategy_cooldown_state.get("credit_spreads", {})
        self.assertAlmostEqual(float(state.get("reduction", 0.0)), 0.25, places=4)
        self.assertEqual(int(state.get("remaining_trades", 0)), 3)

    def test_loss_streak_level2_applies_50pct_reduction(self) -> None:
        bot = TradingBot(make_config())
        bot._recent_closed_trades = mock.Mock(
            return_value=[
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -50.0,
                    "close_date": "2026-02-20",
                },
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -25.0,
                    "close_date": "2026-02-21",
                },
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -35.0,
                    "close_date": "2026-02-22",
                },
            ]
        )

        bot._refresh_strategy_and_symbol_breakers()

        state = bot._strategy_cooldown_state.get("credit_spreads", {})
        self.assertAlmostEqual(float(state.get("reduction", 0.0)), 0.50, places=4)
        self.assertEqual(int(state.get("remaining_trades", 0)), 5)

    def test_profitable_close_resets_cooldown_state(self) -> None:
        bot = TradingBot(make_config())
        bot._strategy_cooldown_state = {
            "credit_spreads": {"reduction": 0.5, "remaining_trades": 5, "level": 2}
        }
        bot._recent_closed_trades = mock.Mock(
            return_value=[
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": -50.0,
                    "close_date": "2026-02-20",
                },
                {
                    "strategy": "bull_put_spread",
                    "symbol": "SPY",
                    "pnl": 75.0,
                    "close_date": "2026-02-21",
                },
            ]
        )

        bot._refresh_strategy_and_symbol_breakers()

        self.assertNotIn("credit_spreads", bot._strategy_cooldown_state)

    def test_entry_consumes_cooldown_remaining_trades(self) -> None:
        bot = TradingBot(make_config())
        bot._strategy_cooldown_state = {
            "credit_spreads": {"reduction": 0.25, "remaining_trades": 1, "level": 1}
        }
        bot._passes_multi_timeframe_confirmation = mock.Mock(
            return_value=(True, 2, {"daily": True})
        )
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=4)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(
            return_value={"status": "FILLED", "position_id": "p1"}
        )
        bot.risk_manager.register_open_position = mock.Mock()
        bot._refresh_monte_carlo_risk = mock.Mock()

        signal = make_signal()
        opened = bot._try_execute_entry(signal)

        self.assertTrue(opened)
        self.assertEqual(bot.paper_trader.execute_open.call_args.kwargs["quantity"], 3)
        self.assertNotIn("credit_spreads", bot._strategy_cooldown_state)


if __name__ == "__main__":
    unittest.main()
