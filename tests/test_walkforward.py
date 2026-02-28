import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bot.backtester import Backtester
from bot.config import BotConfig


class WalkForwardTests(unittest.TestCase):
    def test_build_walkforward_windows_split_logic(self) -> None:
        days = [date(2026, 1, 1) + timedelta(days=i) for i in range(120)]

        windows = Backtester._build_walkforward_windows(
            days,
            train_days=60,
            test_days=20,
            step_days=20,
        )

        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].train_start, "2026-01-01")
        self.assertEqual(windows[0].test_start, "2026-03-02")

    def test_parameter_grid_contains_expected_variations(self) -> None:
        grid = Backtester._walkforward_parameter_grid()

        self.assertTrue(grid)
        short_deltas = sorted({row["short_delta"] for row in grid})
        min_dte = sorted({row["min_dte"] for row in grid})
        self.assertEqual(short_deltas, [0.1, 0.15, 0.2, 0.25])
        self.assertEqual(min_dte, [15, 20, 25, 30])

    def test_run_walkforward_persists_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            start = date(2026, 1, 1)
            for idx in range(120):
                day = start + timedelta(days=idx)
                (data_dir / f"SPY_{day.isoformat()}.parquet.csv.gz").touch()

            cfg = BotConfig()
            cfg.credit_spreads.enabled = True
            cfg.iron_condors.enabled = False
            cfg.covered_calls.enabled = False
            cfg.naked_puts.enabled = False
            cfg.calendar_spreads.enabled = False
            backtester = Backtester(cfg, data_dir=data_dir)

            backtester._build_strategies = mock.Mock(
                return_value=[SimpleNamespace(name="credit_spreads")]
            )

            def eval_side_effect(
                *, strategy_name: str, params: dict, start: str, end: str
            ) -> dict:
                bias = 0.02 if float(params.get("short_delta", 0.0)) == 0.25 else 0.01
                return {
                    "total_return": bias,
                    "sharpe_ratio": 1.0 + bias,
                    "win_rate": 0.6 + bias,
                    "max_drawdown": 0.1 - bias,
                    "closed_trades": 12,
                }

            backtester._walkforward_eval_window = mock.Mock(
                side_effect=eval_side_effect
            )

            result = backtester.run_walkforward(
                train_days=60, test_days=20, step_days=20
            )

            self.assertIn("strategies", result)
            self.assertIn("credit_spreads", result["strategies"])
            self.assertGreater(len(result["windows"]), 0)
            output = Path(result["output_path"])
            self.assertTrue(output.exists())

    def test_run_walkforward_handles_empty_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = BotConfig()
            backtester = Backtester(cfg, data_dir=tmp_dir)

            result = backtester.run_walkforward()

            self.assertEqual(result["windows"], [])
            self.assertEqual(result["oos_summary"]["windows"], 0)


if __name__ == "__main__":
    unittest.main()
