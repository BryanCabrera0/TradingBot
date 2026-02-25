import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from bot.backtester import Backtester
from bot.config import BotConfig


def _write_snapshot(path: Path, snapshot_date: str, short_mid: float, long_mid: float) -> None:
    frame = pd.DataFrame(
        [
            {
                "symbol": "SPY",
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "PUT",
                "strike": 95.0,
                "bid": short_mid - 0.05,
                "ask": short_mid + 0.05,
                "mid": short_mid,
                "delta": -0.30,
                "gamma": 0.01,
                "theta": -0.04,
                "vega": 0.05,
                "iv": 25.0,
                "volume": 200,
                "open_interest": 1000,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": "SPY",
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "PUT",
                "strike": 90.0,
                "bid": long_mid - 0.05,
                "ask": long_mid + 0.05,
                "mid": long_mid,
                "delta": -0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.03,
                "iv": 24.0,
                "volume": 150,
                "open_interest": 900,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": "SPY",
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "CALL",
                "strike": 105.0,
                "bid": 0.9,
                "ask": 1.1,
                "mid": 1.0,
                "delta": 0.30,
                "gamma": 0.01,
                "theta": -0.03,
                "vega": 0.05,
                "iv": 26.0,
                "volume": 150,
                "open_interest": 1000,
                "dte": 30,
                "underlying_price": 100.0,
            },
            {
                "symbol": "SPY",
                "snapshot_date": snapshot_date,
                "expiration": "2026-03-20",
                "side": "CALL",
                "strike": 110.0,
                "bid": 0.2,
                "ask": 0.3,
                "mid": 0.25,
                "delta": 0.10,
                "gamma": 0.01,
                "theta": -0.02,
                "vega": 0.03,
                "iv": 24.0,
                "volume": 110,
                "open_interest": 800,
                "dte": 30,
                "underlying_price": 100.0,
            },
        ]
    )
    frame.to_csv(path, index=False, compression="gzip")


class BacktesterTests(unittest.TestCase):
    def test_backtester_generates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            _write_snapshot(data_dir / "SPY_2026-02-20.parquet.csv.gz", "2026-02-20", short_mid=1.2, long_mid=0.2)
            _write_snapshot(data_dir / "SPY_2026-02-23.parquet.csv.gz", "2026-02-23", short_mid=0.4, long_mid=0.1)

            cfg = BotConfig()
            cfg.scanner.enabled = False
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.credit_spreads.enabled = True
            cfg.credit_spreads.min_dte = 1
            cfg.credit_spreads.max_dte = 45

            backtester = Backtester(cfg, data_dir=data_dir, initial_balance=100_000.0)
            backtester.risk_manager.earnings_calendar = mock.Mock(
                earnings_within_window=mock.Mock(return_value=(False, None))
            )

            result = backtester.run(start="2026-02-20", end="2026-02-23")

            self.assertTrue(Path(result.report_path).exists())
            self.assertIn("total_return", result.report)
            self.assertIn("monthly_returns", result.report)


if __name__ == "__main__":
    unittest.main()
