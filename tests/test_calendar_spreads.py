import unittest

from bot.strategies.calendar_spreads import CalendarSpreadStrategy


def _base_chain(
    front_iv: float = 32.0, back_iv: float = 25.0, back_mid: float = 2.8
) -> dict:
    return {
        "underlying_price": 100.0,
        "calls": {
            "2026-03-20": [
                {
                    "strike": 100.0,
                    "mid": 2.0,
                    "bid": 1.9,
                    "ask": 2.1,
                    "delta": 0.50,
                    "gamma": 0.02,
                    "theta": -0.05,
                    "vega": 0.12,
                    "iv": front_iv,
                    "open_interest": 800,
                    "volume": 120,
                    "dte": 25,
                }
            ],
            "2026-04-17": [
                {
                    "strike": 100.0,
                    "mid": back_mid,
                    "bid": back_mid - 0.1,
                    "ask": back_mid + 0.1,
                    "delta": 0.52,
                    "gamma": 0.015,
                    "theta": -0.03,
                    "vega": 0.16,
                    "iv": back_iv,
                    "open_interest": 700,
                    "volume": 90,
                    "dte": 55,
                }
            ],
        },
        "puts": {},
    }


class CalendarSpreadStrategyTests(unittest.TestCase):
    def test_entry_signal_generation(self) -> None:
        strategy = CalendarSpreadStrategy(
            {
                "front_min_dte": 20,
                "front_max_dte": 30,
                "back_min_dte": 50,
                "back_max_dte": 60,
                "profit_target_pct": 0.25,
                "exit_dte": 7,
            }
        )

        signals = strategy.scan_for_entries("AAPL", _base_chain(), 100.0)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].strategy, "calendar_spread")
        self.assertLess(signals[0].analysis.credit, 0.0)

    def test_iv_term_structure_filter(self) -> None:
        strategy = CalendarSpreadStrategy({})

        signals = strategy.scan_for_entries(
            "AAPL",
            _base_chain(front_iv=22.0, back_iv=25.0),
            100.0,
        )

        self.assertEqual(signals, [])

    def test_exit_on_dte(self) -> None:
        strategy = CalendarSpreadStrategy({"profit_target_pct": 0.25, "exit_dte": 7})
        positions = [
            {
                "position_id": "cal1",
                "strategy": "calendar_spread",
                "symbol": "AAPL",
                "entry_credit": -0.8,
                "current_value": -0.9,
                "dte_remaining": 7,
                "status": "open",
                "quantity": 1,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, "close")
        self.assertIn("DTE", signals[0].reason)

    def test_exit_on_profit(self) -> None:
        strategy = CalendarSpreadStrategy({"profit_target_pct": 0.25, "exit_dte": 7})
        positions = [
            {
                "position_id": "cal2",
                "strategy": "calendar_spread",
                "symbol": "AAPL",
                "entry_credit": -0.8,
                "current_value": -1.1,
                "dte_remaining": 20,
                "status": "open",
                "quantity": 1,
            }
        ]

        signals = strategy.check_exits(positions, market_client=None)

        self.assertEqual(len(signals), 1)
        self.assertIn("Profit target", signals[0].reason)

    def test_debit_cost_validation(self) -> None:
        strategy = CalendarSpreadStrategy({})
        # back_mid=3.2 -> debit=1.2, which is >= 50% of front_mid(2.0), so reject.
        signals = strategy.scan_for_entries("AAPL", _base_chain(back_mid=3.2), 100.0)

        self.assertEqual(signals, [])


if __name__ == "__main__":
    unittest.main()
