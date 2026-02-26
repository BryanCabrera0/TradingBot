import unittest
from datetime import datetime, timedelta
from unittest import mock

from bot.analysis import SpreadAnalysis
from bot.config import BotConfig
from bot.orchestrator import TradingBot
from bot.strategies.base import TradeSignal


def _config() -> BotConfig:
    cfg = BotConfig()
    cfg.trading_mode = "paper"
    cfg.scanner.enabled = False
    cfg.news.enabled = False
    cfg.llm.enabled = False
    cfg.credit_spreads.enabled = True
    cfg.iron_condors.enabled = False
    cfg.covered_calls.enabled = False
    cfg.scaling.enabled = True
    cfg.scaling.scale_in_delay_minutes = 60
    cfg.scaling.scale_in_max_adds = 2
    cfg.scaling.partial_exit_pct = 0.40
    cfg.scaling.partial_exit_size = 0.50
    return cfg


def _signal() -> TradeSignal:
    return TradeSignal(
        action="open",
        strategy="bull_put_spread",
        symbol="SPY",
        analysis=SpreadAnalysis(
            symbol="SPY",
            strategy="bull_put_spread",
            expiration="2026-03-20",
            dte=30,
            short_strike=100.0,
            long_strike=95.0,
            credit=1.2,
            max_loss=3.8,
            probability_of_profit=0.62,
            score=58.0,
        ),
        metadata={
            "composite_score": 90.0,
            "ml_score": 0.8,
            "llm_verdict": "approve",
            "llm_confidence_pct": 85.0,
        },
    )


class PositionScalingTests(unittest.TestCase):
    def test_high_conviction_entry_starts_with_one_and_queues_scale_plan(self) -> None:
        bot = TradingBot(_config())
        bot.risk_manager.calculate_position_size = mock.Mock(return_value=3)
        bot.risk_manager.approve_trade = mock.Mock(return_value=(True, "ok"))
        bot.paper_trader.execute_open = mock.Mock(return_value={"status": "FILLED", "position_id": "p1"})
        bot.risk_manager.register_open_position = mock.Mock()

        executed = bot._try_execute_entry(_signal())

        self.assertTrue(executed)
        kwargs = bot.paper_trader.execute_open.call_args.kwargs
        self.assertEqual(kwargs["quantity"], 1)
        self.assertEqual(len(bot._pending_scale_ins), 1)
        self.assertEqual(bot._pending_scale_ins[0]["max_adds"], 2)

    def test_process_scale_ins_adds_when_profitable_after_delay(self) -> None:
        bot = TradingBot(_config())
        now = datetime.fromisoformat("2026-02-23T12:30:00-05:00")
        bot._now_eastern = mock.Mock(return_value=now)
        bot._pending_scale_ins = [
            {
                "position_id": "p1",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "adds_done": 0,
                "max_adds": 1,
                "last_action_at": (now - timedelta(minutes=61)).isoformat(),
            }
        ]
        bot._get_tracked_positions = mock.Mock(
            return_value=[
                {
                    "position_id": "p1",
                    "status": "open",
                    "symbol": "SPY",
                    "strategy": "bull_put_spread",
                    "quantity": 1,
                    "entry_credit": 1.2,
                    "current_value": 0.8,
                    "max_loss": 3.8,
                    "details": {
                        "expiration": "2026-03-20",
                        "dte": 30,
                        "short_strike": 100.0,
                        "long_strike": 95.0,
                        "score": 58.0,
                        "probability_of_profit": 0.62,
                    },
                }
            ]
        )
        bot._try_execute_entry = mock.Mock(return_value=True)

        bot._process_scale_ins()

        bot._try_execute_entry.assert_called_once()
        signal = bot._try_execute_entry.call_args.args[0]
        self.assertEqual(signal.quantity, 1)
        self.assertTrue(signal.metadata.get("scale_in_add"))
        self.assertEqual(bot._pending_scale_ins, [])

    def test_partial_exit_signal_closes_fraction_at_target(self) -> None:
        bot = TradingBot(_config())
        positions = [
            {
                "position_id": "p2",
                "status": "open",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "quantity": 4,
                "entry_credit": 1.0,
                "current_value": 0.55,
                "partial_closed": False,
            }
        ]

        signals = bot._apply_scaling_partial_exits(positions)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertEqual(signals[0].quantity, 2)


if __name__ == "__main__":
    unittest.main()

