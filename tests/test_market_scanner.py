import unittest
from datetime import datetime, timedelta
from unittest import mock

from bot.config import ScannerConfig
from bot.market_scanner import MarketScanner, TickerScore


class MarketScannerTests(unittest.TestCase):
    def test_build_universe_respects_include_movers_flag(self) -> None:
        scanner = MarketScanner(
            schwab_client=object(),
            config=ScannerConfig(include_movers=False),
        )

        with mock.patch.object(scanner, "_fetch_movers", side_effect=AssertionError("should not be called")):
            universe = scanner._build_universe()

        self.assertIn("SPY", universe)

    def test_cache_expiration_uses_total_elapsed_time(self) -> None:
        scanner = MarketScanner(schwab_client=None, config=ScannerConfig(cache_seconds=1800))
        scanner._last_scan_results = [TickerScore(symbol="OLD", score=75.0)]
        scanner._last_scan_time = datetime.now() - timedelta(days=1, minutes=5)

        scanner.scan = mock.Mock(return_value=["NEW"])

        results = scanner.get_cached_results()

        self.assertEqual(results, ["NEW"])
        scanner.scan.assert_called_once()

    def test_score_ticker_tolerates_null_numeric_contract_fields(self) -> None:
        fake_schwab = mock.Mock()
        fake_schwab.get_quote.return_value = {
            "quote": {"lastPrice": 100.0, "totalVolume": 2_000_000}
        }
        fake_schwab.get_option_chain.return_value = {
            "callExpDateMap": {
                "2026-03-20:30": {
                    "100.0": [
                        {
                            "totalVolume": None,
                            "openInterest": None,
                            "bid": None,
                            "ask": "1.50",
                            "volatility": None,
                        }
                    ]
                }
            },
            "putExpDateMap": {},
        }

        scanner = MarketScanner(
            schwab_client=fake_schwab,
            config=ScannerConfig(min_underlying_volume=1_000),
        )

        score = scanner._score_ticker("SPY")

        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.score, 0.0)

    def test_scan_stops_after_max_consecutive_errors(self) -> None:
        scanner = MarketScanner(
            schwab_client=object(),
            config=ScannerConfig(
                request_pause_seconds=0.0,
                max_consecutive_errors=2,
            ),
        )
        scanner._build_universe = mock.Mock(return_value=["A", "B", "C", "D"])
        scanner._score_ticker = mock.Mock(side_effect=RuntimeError("boom"))

        results = scanner.scan()

        self.assertEqual(results, [])
        self.assertEqual(scanner._score_ticker.call_count, 2)


if __name__ == "__main__":
    unittest.main()
