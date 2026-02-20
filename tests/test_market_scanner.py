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


if __name__ == "__main__":
    unittest.main()
