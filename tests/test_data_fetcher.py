import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

from bot.data_fetcher import HistoricalDataFetcher, _flatten_chain_rows


class _FakeSchwab:
    def get_option_chain(self, symbol, **kwargs):
        return {
            "underlyingPrice": 100.0,
            "callExpDateMap": {
                "2026-03-20:30": {
                    "105.0": [
                        {
                            "bid": 1.0,
                            "ask": 1.2,
                            "delta": 0.25,
                            "gamma": 0.01,
                            "theta": -0.03,
                            "vega": 0.05,
                            "volatility": 22.0,
                            "totalVolume": 100,
                            "openInterest": 500,
                            "daysToExpiration": 30,
                        }
                    ]
                }
            },
            "putExpDateMap": {
                "2026-03-20:30": {
                    "95.0": [
                        {
                            "bid": 1.1,
                            "ask": 1.3,
                            "delta": -0.25,
                            "gamma": 0.01,
                            "theta": -0.03,
                            "vega": 0.05,
                            "volatility": 23.0,
                            "totalVolume": 90,
                            "openInterest": 450,
                            "daysToExpiration": 30,
                        }
                    ]
                }
            },
        }


class DataFetcherTests(unittest.TestCase):
    def test_fetch_day_writes_snapshot_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fetcher = HistoricalDataFetcher(_FakeSchwab(), data_dir=tmp_dir)

            result = fetcher.fetch_day(symbol="SPY", trading_day=date(2026, 2, 20))

            self.assertFalse(result.skipped)
            self.assertGreaterEqual(result.rows, 1)
            parquet_path = Path(tmp_dir) / "SPY_2026-02-20.parquet.gz"
            csv_fallback = Path(tmp_dir) / "SPY_2026-02-20.parquet.csv.gz"
            self.assertTrue(parquet_path.exists() or csv_fallback.exists())

    def test_flatten_chain_rows_schema(self) -> None:
        rows = _flatten_chain_rows(
            "SPY",
            date(2026, 2, 20),
            _FakeSchwab().get_option_chain("SPY"),
        )
        self.assertGreaterEqual(len(rows), 2)
        required = {
            "symbol",
            "snapshot_date",
            "expiration",
            "side",
            "strike",
            "bid",
            "ask",
            "mid",
            "delta",
            "gamma",
            "theta",
            "vega",
            "iv",
            "volume",
            "open_interest",
            "dte",
            "underlying_price",
        }
        self.assertTrue(required.issubset(rows[0].keys()))

    def test_cache_hit_skips_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "SPY_2026-02-20.parquet.gz"
            path.write_bytes(b"cached")
            fake = mock.Mock()
            fetcher = HistoricalDataFetcher(fake, data_dir=tmp_dir)

            result = fetcher.fetch_day(symbol="SPY", trading_day=date(2026, 2, 20))

            self.assertTrue(result.skipped)
            fake.get_option_chain.assert_not_called()

    def test_retry_on_failure(self) -> None:
        client = mock.Mock()
        client.get_option_chain.side_effect = [
            RuntimeError("try-1"),
            RuntimeError("try-2"),
            _FakeSchwab().get_option_chain("SPY"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            fetcher = HistoricalDataFetcher(client, data_dir=tmp_dir, max_attempts=5, backoff_seconds=0.01)
            with mock.patch("bot.data_fetcher.time.sleep", return_value=None):
                result = fetcher.fetch_day(symbol="SPY", trading_day=date(2026, 2, 20))

            self.assertFalse(result.skipped)
            self.assertGreater(result.rows, 0)
            self.assertEqual(client.get_option_chain.call_count, 3)


if __name__ == "__main__":
    unittest.main()
