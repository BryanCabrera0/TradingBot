import unittest

from bot.strategies.covered_calls import CoveredCallStrategy
from bot.strategies.credit_spreads import CreditSpreadStrategy
from bot.strategies.iron_condors import IronCondorStrategy


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


if __name__ == "__main__":
    unittest.main()
