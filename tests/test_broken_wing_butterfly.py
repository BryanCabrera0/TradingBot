import unittest

from bot.strategies.broken_wing_butterfly import BrokenWingButterflyStrategy


def _chain() -> dict:
    return {
        "calls": {
            "2026-03-20": [
                {
                    "dte": 30,
                    "strike": 100.0,
                    "mid": 4.0,
                    "delta": 0.30,
                    "theta": -0.04,
                    "vega": 0.08,
                },
                {
                    "dte": 30,
                    "strike": 95.0,
                    "mid": 6.0,
                    "delta": 0.45,
                    "theta": -0.05,
                    "vega": 0.09,
                },
                {
                    "dte": 30,
                    "strike": 90.0,
                    "mid": 8.0,
                    "delta": 0.55,
                    "theta": -0.06,
                    "vega": 0.10,
                },
                {
                    "dte": 30,
                    "strike": 105.0,
                    "mid": 2.5,
                    "delta": 0.20,
                    "theta": -0.03,
                    "vega": 0.07,
                },
                {
                    "dte": 30,
                    "strike": 110.0,
                    "mid": 1.5,
                    "delta": 0.10,
                    "theta": -0.02,
                    "vega": 0.06,
                },
            ]
        },
        "puts": {
            "2026-03-20": [
                {
                    "dte": 30,
                    "strike": 100.0,
                    "mid": 4.2,
                    "delta": -0.30,
                    "theta": -0.04,
                    "vega": 0.08,
                },
                {
                    "dte": 30,
                    "strike": 95.0,
                    "mid": 2.6,
                    "delta": -0.20,
                    "theta": -0.03,
                    "vega": 0.07,
                },
                {
                    "dte": 30,
                    "strike": 90.0,
                    "mid": 1.6,
                    "delta": -0.10,
                    "theta": -0.02,
                    "vega": 0.06,
                },
                {
                    "dte": 30,
                    "strike": 105.0,
                    "mid": 6.2,
                    "delta": -0.45,
                    "theta": -0.05,
                    "vega": 0.09,
                },
                {
                    "dte": 30,
                    "strike": 110.0,
                    "mid": 8.1,
                    "delta": -0.55,
                    "theta": -0.06,
                    "vega": 0.10,
                },
            ]
        },
    }


class BrokenWingButterflyTests(unittest.TestCase):
    def test_generates_bullish_signal_in_bull_regime(self) -> None:
        strategy = BrokenWingButterflyStrategy(
            {
                "min_dte": 20,
                "max_dte": 45,
                "short_delta": 0.30,
                "near_wing_width": 5.0,
                "far_wing_width": 10.0,
                "min_credit": 0.1,
            }
        )
        signals = strategy.scan_for_entries(
            "SPY", _chain(), 100.0, market_context={"regime": "BULL_TREND"}
        )
        self.assertTrue(signals)
        self.assertEqual(
            signals[0].metadata["position_details"]["direction"], "bullish"
        )

    def test_generates_bearish_signal_in_bear_regime(self) -> None:
        strategy = BrokenWingButterflyStrategy(
            {
                "min_dte": 20,
                "max_dte": 45,
                "short_delta": 0.30,
                "near_wing_width": 5.0,
                "far_wing_width": 10.0,
                "min_credit": 0.1,
            }
        )
        signals = strategy.scan_for_entries(
            "SPY", _chain(), 100.0, market_context={"regime": "BEAR_TREND"}
        )
        self.assertTrue(signals)
        self.assertEqual(
            signals[0].metadata["position_details"]["direction"], "bearish"
        )

    def test_min_credit_filter(self) -> None:
        strategy = BrokenWingButterflyStrategy(
            {
                "min_dte": 20,
                "max_dte": 45,
                "short_delta": 0.30,
                "near_wing_width": 5.0,
                "far_wing_width": 10.0,
                "min_credit": 5.0,
            }
        )
        signals = strategy.scan_for_entries(
            "SPY", _chain(), 100.0, market_context={"regime": "BULL_TREND"}
        )
        self.assertEqual(signals, [])

    def test_exit_on_profit_target(self) -> None:
        strategy = BrokenWingButterflyStrategy({})
        positions = [
            {
                "position_id": "b1",
                "strategy": "broken_wing_butterfly",
                "symbol": "SPY",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.5,
                "quantity": 1,
                "dte_remaining": 20,
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")

    def test_exit_on_dte_threshold(self) -> None:
        strategy = BrokenWingButterflyStrategy({})
        positions = [
            {
                "position_id": "b2",
                "strategy": "broken_wing_butterfly",
                "symbol": "SPY",
                "status": "open",
                "entry_credit": 1.0,
                "current_value": 0.95,
                "quantity": 1,
                "dte_remaining": 6,
            }
        ]
        signals = strategy.check_exits(positions, market_client=None)
        self.assertEqual(len(signals), 1)


if __name__ == "__main__":
    unittest.main()
