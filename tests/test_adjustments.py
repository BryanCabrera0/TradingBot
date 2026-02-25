import unittest

from bot.adjustments import AdjustmentEngine


def _position(**overrides):
    base = {
        "position_id": "p1",
        "status": "open",
        "strategy": "bull_put_spread",
        "symbol": "SPY",
        "entry_credit": 1.0,
        "current_value": 0.9,
        "dte_remaining": 20,
        "underlying_price": 100.0,
        "quantity": 2,
        "details": {"short_strike": 100.5, "adjustment_count": 0},
    }
    base.update(overrides)
    return base


class AdjustmentEngineTests(unittest.TestCase):
    def test_disabled_engine_returns_none_plan(self) -> None:
        engine = AdjustmentEngine({"enabled": False})
        plan = engine.evaluate(position=_position(), regime="normal", iv_change_since_entry=0.0)
        self.assertEqual(plan.action, "none")

    def test_add_wing_when_high_vol_and_tested(self) -> None:
        engine = AdjustmentEngine({"enabled": True, "delta_test_threshold": 0.50, "min_dte_remaining": 7, "max_adjustments_per_position": 2})
        plan = engine.evaluate(position=_position(), regime="HIGH_VOL_CHOP", iv_change_since_entry=0.20)
        self.assertEqual(plan.action, "add_wing")

    def test_roll_tested_side_when_losing(self) -> None:
        engine = AdjustmentEngine({"enabled": True, "min_dte_remaining": 7, "max_adjustments_per_position": 2})
        plan = engine.evaluate(position=_position(current_value=1.4), regime="normal", iv_change_since_entry=0.0)
        self.assertEqual(plan.action, "roll_tested_side")

    def test_to_signal_generates_roll_signal(self) -> None:
        engine = AdjustmentEngine({"enabled": True})
        plan = engine.evaluate(position=_position(current_value=1.4), regime="normal", iv_change_since_entry=0.0)
        signal = engine.to_signal(position=_position(current_value=1.4), plan=plan)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.action, "roll")

    def test_max_adjustments_block(self) -> None:
        engine = AdjustmentEngine({"enabled": True, "max_adjustments_per_position": 1})
        plan = engine.evaluate(position=_position(details={"short_strike": 100.5, "adjustment_count": 1}), regime="normal", iv_change_since_entry=0.0)
        self.assertEqual(plan.action, "none")


if __name__ == "__main__":
    unittest.main()

