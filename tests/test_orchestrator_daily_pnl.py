import unittest
from datetime import date

from bot.orchestrator import TradingBot


class OrchestratorDailyPnlTests(unittest.TestCase):
    def test_compute_daily_pnl_for_cross_day_short_option_close(self) -> None:
        target_date = date(2026, 2, 22)
        orders = [
            {
                "status": "FILLED",
                "orderLegCollection": [
                    {
                        "legId": 1,
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "SPY   260320P00095000",
                            "assetType": "OPTION",
                        },
                    }
                ],
                "orderActivityCollection": [
                    {
                        "executionLegs": [
                            {
                                "legId": 1,
                                "quantity": 1,
                                "price": 1.50,
                                "time": "2026-02-21T15:00:00+00:00",
                            }
                        ]
                    }
                ],
            },
            {
                "status": "FILLED",
                "orderLegCollection": [
                    {
                        "legId": 1,
                        "instruction": "BUY_TO_CLOSE",
                        "instrument": {
                            "symbol": "SPY   260320P00095000",
                            "assetType": "OPTION",
                        },
                    }
                ],
                "orderActivityCollection": [
                    {
                        "executionLegs": [
                            {
                                "legId": 1,
                                "quantity": 1,
                                "price": 0.50,
                                "time": "2026-02-22T15:30:00+00:00",
                            }
                        ]
                    }
                ],
            },
        ]

        pnl = TradingBot._compute_daily_pnl_from_orders(orders, target_date)

        self.assertEqual(round(pnl, 2), 100.0)

    def test_compute_daily_pnl_for_multi_leg_credit_spread_close(self) -> None:
        target_date = date(2026, 2, 22)
        orders = [
            {
                "status": "FILLED",
                "orderLegCollection": [
                    {
                        "legId": 1,
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "SPY   260320P00100000",
                            "assetType": "OPTION",
                        },
                    },
                    {
                        "legId": 2,
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "SPY   260320P00095000",
                            "assetType": "OPTION",
                        },
                    },
                ],
                "orderActivityCollection": [
                    {
                        "executionLegs": [
                            {
                                "legId": 1,
                                "quantity": 1,
                                "price": 2.00,
                                "time": "2026-02-21T16:00:00+00:00",
                            },
                            {
                                "legId": 2,
                                "quantity": 1,
                                "price": 1.20,
                                "time": "2026-02-21T16:00:00+00:00",
                            },
                        ]
                    }
                ],
            },
            {
                "status": "FILLED",
                "orderLegCollection": [
                    {
                        "legId": 1,
                        "instruction": "BUY_TO_CLOSE",
                        "instrument": {
                            "symbol": "SPY   260320P00100000",
                            "assetType": "OPTION",
                        },
                    },
                    {
                        "legId": 2,
                        "instruction": "SELL_TO_CLOSE",
                        "instrument": {
                            "symbol": "SPY   260320P00095000",
                            "assetType": "OPTION",
                        },
                    },
                ],
                "orderActivityCollection": [
                    {
                        "executionLegs": [
                            {
                                "legId": 1,
                                "quantity": 1,
                                "price": 0.80,
                                "time": "2026-02-22T16:10:00+00:00",
                            },
                            {
                                "legId": 2,
                                "quantity": 1,
                                "price": 0.20,
                                "time": "2026-02-22T16:10:00+00:00",
                            },
                        ]
                    }
                ],
            },
        ]

        pnl = TradingBot._compute_daily_pnl_from_orders(orders, target_date)

        self.assertEqual(round(pnl, 2), 20.0)

    def test_parse_order_timestamp_accepts_z_and_compact_offsets(self) -> None:
        parsed_z = TradingBot._parse_order_timestamp("2026-02-22T12:34:56Z")
        parsed_compact = TradingBot._parse_order_timestamp("2026-02-22T12:34:56+0000")

        self.assertIsNotNone(parsed_z)
        self.assertIsNotNone(parsed_compact)
        self.assertEqual(parsed_z.isoformat(), "2026-02-22T12:34:56+00:00")
        self.assertEqual(parsed_compact.isoformat(), "2026-02-22T12:34:56+00:00")

    def test_option_symbol_key_normalizes_occ_and_underscore_formats(self) -> None:
        occ = TradingBot._option_symbol_key("SPY  260320C00100000")
        underscore = TradingBot._option_symbol_key("SPY_032026C100")

        self.assertEqual(occ, underscore)


if __name__ == "__main__":
    unittest.main()
