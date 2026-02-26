import unittest

from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.earnings_vol_crush import EarningsVolCrushStrategy
from bot.strategies.iron_condors import IronCondorStrategy
from bot.strategies.broken_wing_butterfly import BrokenWingButterflyStrategy
from bot.strategies.strangles import StranglesStrategy


class StrategyExitTests(unittest.TestCase):
    def test_credit_spreads_skip_non_open_positions(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p0",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "status": "closing",
                "entry_credit": 1.0,
                "current_value": 0.1,
                "dte_remaining": 2,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(signals, [])

    def test_credit_spreads_no_duplicate_close_signals(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p1",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.2,
                "dte_remaining": 3,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].position_id, "p1")

    def test_credit_spreads_dte_exit_works_without_entry_credit(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p1dte",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 0.0,
                "current_value": 0.2,
                "dte_remaining": 2,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].position_id, "p1dte")

    def test_iron_condors_no_duplicate_close_signals(self) -> None:
        strategy = IronCondorStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p2",
                "strategy": "iron_condor",
                "symbol": "QQQ",
                "entry_credit": 1.0,
                "current_value": 0.2,
                "dte_remaining": 2,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].position_id, "p2")

    def test_covered_calls_no_duplicate_close_signals(self) -> None:
        strategy = CoveredCallStrategy({"profit_target_pct": 0.5})
        positions = [
            {
                "position_id": "p3",
                "strategy": "covered_call",
                "symbol": "AAPL",
                "entry_credit": 1.0,
                "current_value": 0.2,
                "dte_remaining": 1,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].position_id, "p3")

    def test_adaptive_profit_target_under_14_dte(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p_under14",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.79,
                "dte_remaining": 9,
                "status": "open",
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")

    def test_adaptive_profit_target_over_30_dte(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p_over30",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.5,
                "dte_remaining": 35,
                "status": "open",
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")

    def test_partial_close_at_40pct(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p_partial",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.6,
                "dte_remaining": 25,
                "status": "open",
                "quantity": 4,
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].quantity, 2)
        self.assertEqual(signals[0].action, "close")

    def test_roll_signal_at_21_dte(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p_roll",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 0.7,
                "dte_remaining": 21,
                "status": "open",
                "quantity": 1,
                "details": {"short_strike": 95.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "roll")

    def test_defend_mode_tightens_stop(self) -> None:
        strategy = CreditSpreadStrategy({"profit_target_pct": 0.5, "stop_loss_pct": 2.0})
        positions = [
            {
                "position_id": "p_defend",
                "strategy": "bull_put_spread",
                "symbol": "SPY",
                "entry_credit": 1.0,
                "current_value": 2.6,
                "dte_remaining": 30,
                "status": "open",
                "quantity": 1,
                "underlying_price": 99.5,
                "details": {"short_strike": 100.0},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertIn("Stop loss", signals[0].reason)

    def test_strangles_trailing_stop_activation_and_trigger(self) -> None:
        strategy = StranglesStrategy(
            {
                "adaptive_targets": True,
                "trailing_stop_enabled": True,
                "trailing_stop_activation_pct": 0.25,
                "trailing_stop_floor_pct": 0.10,
            }
        )
        position = {
            "position_id": "str1",
            "strategy": "short_strangle",
            "symbol": "SPY",
            "status": "open",
            "entry_credit": 1.0,
            "current_value": 0.6,  # 40% profit
            "dte_remaining": 35,
            "details": {},
        }

        first = strategy.check_exits([position], market_client=None)
        self.assertEqual(first, [])
        self.assertGreater(position.get("trailing_stop_high", 0.0), 0.0)

        position["current_value"] = 0.72  # 28% profit: below high-water minus floor (30%)
        second = strategy.check_exits([position], market_client=None)
        self.assertEqual(len(second), 1)
        self.assertIn("Trailing stop", second[0].reason)

    def test_bwb_adaptive_target_respects_long_dte_threshold(self) -> None:
        strategy = BrokenWingButterflyStrategy({"adaptive_targets": True})
        positions = [
            {
                "position_id": "bwb1",
                "strategy": "broken_wing_butterfly",
                "symbol": "QQQ",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.55,
                "dte_remaining": 35,  # target 50%
                "details": {},
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 0)

    def test_earnings_vol_crush_adaptive_target_short_dte(self) -> None:
        strategy = EarningsVolCrushStrategy({"adaptive_targets": True})
        positions = [
            {
                "position_id": "evc1",
                "strategy": "earnings_vol_crush",
                "symbol": "AAPL",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.79,  # 21% profit
                "dte_remaining": 5,     # target 20%
                "details": {},
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")


if __name__ == "__main__":
    unittest.main()
