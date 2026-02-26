import unittest

from bot.analytics import compute


class AnalyticsTests(unittest.TestCase):
    def _trades(self):
        return [
            {
                "strategy": "bull_put_spread",
                "regime": "BULL_TREND",
                "close_date": "2026-02-10",
                "pnl": 100.0,
                "max_loss": 2.0,
                "quantity": 1,
            },
            {
                "strategy": "bull_put_spread",
                "regime": "BULL_TREND",
                "close_date": "2026-02-11",
                "pnl": -50.0,
                "max_loss": 2.5,
                "quantity": 1,
            },
            {
                "strategy": "iron_condor",
                "regime": "HIGH_VOL_CHOP",
                "close_date": "2026-03-01",
                "pnl": 150.0,
                "max_loss": 3.0,
                "quantity": 1,
            },
            {
                "strategy": "iron_condor",
                "regime": "HIGH_VOL_CHOP",
                "close_date": "2026-03-02",
                "pnl": -100.0,
                "max_loss": 3.0,
                "quantity": 1,
            },
        ]

    def test_compute_core_metrics(self) -> None:
        report = compute(self._trades(), initial_equity=100_000)
        core = report.core_metrics

        self.assertEqual(core["trades"], 4)
        self.assertAlmostEqual(core["total_pnl"], 100.0, places=4)
        self.assertAlmostEqual(core["win_rate"], 0.5, places=4)
        self.assertAlmostEqual(core["profit_factor"], 1.666667, places=5)
        self.assertAlmostEqual(core["expectancy_per_trade"], 25.0, places=4)
        self.assertEqual(core["max_consecutive_wins"], 1)
        self.assertEqual(core["max_consecutive_losses"], 1)
        self.assertEqual(core["current_consecutive_losses"], 1)

    def test_compute_breakdowns(self) -> None:
        report = compute(self._trades(), initial_equity=100_000)

        self.assertIn("bull_put_spread", report.strategy_metrics)
        self.assertIn("iron_condor", report.strategy_metrics)
        self.assertAlmostEqual(report.strategy_metrics["bull_put_spread"]["total_pnl"], 50.0, places=4)
        self.assertAlmostEqual(report.strategy_metrics["iron_condor"]["total_pnl"], 50.0, places=4)

        self.assertIn("BULL_TREND", report.regime_metrics)
        self.assertIn("HIGH_VOL_CHOP", report.regime_metrics)
        self.assertIn("2026-02", report.monthly_metrics)
        self.assertIn("2026-03", report.monthly_metrics)

    def test_risk_adjusted_return_uses_risk_deployed(self) -> None:
        report = compute(self._trades(), initial_equity=100_000)
        core = report.core_metrics
        # Total deployed risk = (2.0 + 2.5 + 3.0 + 3.0) * 100 = 1050
        self.assertAlmostEqual(core["risk_adjusted_return"], 100.0 / 1050.0, places=6)

    def test_compute_handles_empty(self) -> None:
        report = compute([], initial_equity=100_000)
        core = report.core_metrics

        self.assertEqual(core["trades"], 0)
        self.assertEqual(core["total_pnl"], 0.0)
        self.assertEqual(core["sharpe"], 0.0)
        self.assertEqual(report.strategy_metrics, {})

    def test_drawdown_and_duration_are_computed(self) -> None:
        trades = [
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-01", "pnl": 500.0, "max_loss": 1.0, "quantity": 1},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-02", "pnl": -1000.0, "max_loss": 1.0, "quantity": 1},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-03", "pnl": 200.0, "max_loss": 1.0, "quantity": 1},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-04", "pnl": 400.0, "max_loss": 1.0, "quantity": 1},
        ]
        report = compute(trades, initial_equity=100_000)
        core = report.core_metrics

        self.assertGreater(core["max_drawdown"], 0.0)
        self.assertGreaterEqual(core["max_drawdown_duration"], 1)

    def test_core_streak_metrics_track_current_and_max(self) -> None:
        trades = [
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-01", "pnl": 100.0},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-02", "pnl": 120.0},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-03", "pnl": -50.0},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-04", "pnl": -30.0},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-05", "pnl": -20.0},
        ]
        report = compute(trades, initial_equity=100_000)
        core = report.core_metrics

        self.assertEqual(core["max_consecutive_wins"], 2)
        self.assertEqual(core["max_consecutive_losses"], 3)
        self.assertEqual(core["current_consecutive_losses"], 3)
        self.assertEqual(core["current_consecutive_wins"], 0)

    def test_bucket_metrics_include_risk_and_streak_fields(self) -> None:
        report = compute(self._trades(), initial_equity=100_000)
        strategy_row = report.strategy_metrics["bull_put_spread"]

        self.assertIn("sharpe", strategy_row)
        self.assertIn("sortino", strategy_row)
        self.assertIn("calmar", strategy_row)
        self.assertIn("max_drawdown", strategy_row)
        self.assertIn("max_drawdown_duration", strategy_row)
        self.assertIn("avg_win_loss_ratio", strategy_row)
        self.assertIn("max_consecutive_wins", strategy_row)
        self.assertIn("max_consecutive_losses", strategy_row)

    def test_profit_factor_infinite_when_no_losses(self) -> None:
        trades = [
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-01", "pnl": 50.0, "max_loss": 1.0},
            {"strategy": "s1", "regime": "R1", "close_date": "2026-01-02", "pnl": 25.0, "max_loss": 1.0},
        ]
        report = compute(trades, initial_equity=100_000)
        self.assertEqual(report.core_metrics["profit_factor"], float("inf"))
        self.assertEqual(report.strategy_metrics["s1"]["profit_factor"], float("inf"))

    def test_to_dict_includes_all_sections(self) -> None:
        report = compute(self._trades(), initial_equity=100_000)
        payload = report.to_dict()
        self.assertIn("core_metrics", payload)
        self.assertIn("strategy_metrics", payload)
        self.assertIn("regime_metrics", payload)
        self.assertIn("monthly_metrics", payload)
        self.assertIn("daily_pnl", payload)


if __name__ == "__main__":
    unittest.main()
