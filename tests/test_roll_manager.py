import unittest

from bot.roll_manager import RollManager
from bot.strategies.base import TradeSignal


class RollManagerTests(unittest.TestCase):
    def test_profit_roll_trigger(self) -> None:
        manager = RollManager(
            {
                "enabled": True,
                "min_dte_trigger": 7,
                "min_credit_for_roll": 0.2,
                "max_rolls_per_position": 2,
            }
        )
        position = {
            "dte_remaining": 6,
            "entry_credit": 1.0,
            "current_value": 0.5,
            "details": {"roll_count": 0},
        }
        decision = manager.evaluate(position, regime="normal")
        self.assertTrue(decision.should_roll)
        self.assertEqual(decision.roll_type, "profit")

    def test_defensive_roll_trigger(self) -> None:
        manager = RollManager(
            {
                "enabled": True,
                "min_dte_trigger": 7,
                "min_credit_for_roll": 0.1,
                "max_rolls_per_position": 2,
            }
        )
        position = {
            "dte_remaining": 9,
            "entry_credit": 1.0,
            "current_value": 1.5,
            "details": {"roll_count": 0},
        }
        decision = manager.evaluate(position, regime="CRASH/CRISIS")
        self.assertTrue(decision.should_roll)
        self.assertEqual(decision.roll_type, "defensive")

    def test_max_rolls_blocks_roll(self) -> None:
        manager = RollManager({"enabled": True, "max_rolls_per_position": 1})
        position = {
            "dte_remaining": 5,
            "entry_credit": 1.0,
            "current_value": 0.8,
            "details": {"roll_count": 1},
        }
        decision = manager.evaluate(position, regime="normal")
        self.assertFalse(decision.should_roll)

    def test_disabled_returns_false(self) -> None:
        manager = RollManager({"enabled": False})
        decision = manager.evaluate(
            {
                "dte_remaining": 5,
                "entry_credit": 1.0,
                "current_value": 0.7,
                "details": {},
            },
            regime="normal",
        )
        self.assertFalse(decision.should_roll)

    def test_annotate_roll_metadata(self) -> None:
        source = {"position_id": "pos-1", "details": {"roll_count": 1}}
        signal = TradeSignal(action="open", strategy="bull_put_spread", symbol="SPY")
        RollManager.annotate_roll_metadata(source, signal)
        self.assertEqual(signal.metadata["position_details"]["rolled_from"], "pos-1")
        self.assertEqual(signal.metadata["position_details"]["roll_count"], 2)


if __name__ == "__main__":
    unittest.main()
