import tempfile
import unittest
from datetime import date
from pathlib import Path

from bot.data_fetcher import HistoricalDataFetcher


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


if __name__ == "__main__":
    unittest.main()
