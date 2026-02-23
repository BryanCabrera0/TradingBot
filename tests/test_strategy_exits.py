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


if __name__ == "__main__":
    unittest.main()
