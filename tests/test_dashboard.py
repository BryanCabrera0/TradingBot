import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bot.dashboard import enrich_dashboard_payload, generate_dashboard


class DashboardTests(unittest.TestCase):
    def test_generate_dashboard_writes_html(self) -> None:
        payload = {
            "equity_curve": [
                {"date": "2026-02-20", "equity": 100000},
                {"date": "2026-02-21", "equity": 100500},
            ],
            "monthly_pnl": {"2026-02": 500.0},
            "strategy_breakdown": {
                "bull_put_spread": {
                    "win_rate": 60.0,
                    "avg_pnl": 50.0,
                    "total_pnl": 500.0,
                }
            },
            "top_winners": [{"symbol": "SPY", "pnl": 200.0}],
            "top_losers": [{"symbol": "QQQ", "pnl": -100.0}],
            "risk_metrics": {
                "sharpe": 1.1,
                "sortino": 1.4,
                "max_drawdown": 0.05,
                "current_drawdown": 0.01,
            },
            "portfolio_greeks": {
                "delta": 10.0,
                "theta": 2.0,
                "gamma": 0.5,
                "vega": 15.0,
            },
            "sector_exposure": {"Technology": 40.0, "Financials": 20.0},
            "circuit_breakers": {"regime": "normal", "halt_entries": False},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dashboard.html"
            output = generate_dashboard(payload, output_path=path)

            html = Path(output).read_text(encoding="utf-8")
            self.assertIn("TradingBot Dashboard", html)
            self.assertIn("Equity Curve", html)
            self.assertIn("Technology", html)

    def test_llm_accuracy_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            llm_path = Path(tmp_dir) / "llm_track.json"
            trades = []
            for idx in range(55):
                verdict = "approve" if idx % 2 == 0 else "reject"
                outcome = 10.0 if verdict == "approve" else -5.0
                trades.append({"verdict": verdict, "outcome": outcome})
            llm_path.write_text(json.dumps({"trades": trades}), encoding="utf-8")

            with mock.patch("bot.dashboard.LLM_TRACK_RECORD_PATH", llm_path):
                enriched = enrich_dashboard_payload({})

            self.assertIn("llm_accuracy", enriched)
            self.assertEqual(enriched["llm_accuracy"]["trades"], 55)
            self.assertAlmostEqual(enriched["llm_accuracy"]["hit_rate"], 1.0, places=4)

    def test_execution_quality_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            execution_path = Path(tmp_dir) / "execution_quality.json"
            execution_path.write_text(
                json.dumps(
                    {
                        "fills": [
                            {"slippage": 0.10, "fill_improvement_vs_mid": -0.10},
                            {"slippage": -0.20, "fill_improvement_vs_mid": 0.20},
                            {"slippage": 0.30, "fill_improvement_vs_mid": -0.30},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("bot.dashboard.EXECUTION_QUALITY_PATH", execution_path):
                enriched = enrich_dashboard_payload({})

            self.assertIn("execution_quality", enriched)
            self.assertAlmostEqual(
                enriched["execution_quality"]["avg_slippage"], 0.0667, places=4
            )
            self.assertAlmostEqual(
                enriched["execution_quality"]["avg_adverse_slippage"], 0.1333, places=4
            )
            self.assertAlmostEqual(
                enriched["execution_quality"]["avg_fill_improvement"], -0.0667, places=4
            )
            self.assertEqual(enriched["execution_quality"]["samples"], 3)
            self.assertIn("by_strategy", enriched["execution_quality"])

    def test_execution_quality_by_strategy_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            execution_path = Path(tmp_dir) / "execution_quality.json"
            execution_path.write_text(
                json.dumps(
                    {
                        "fills": [
                            {
                                "strategy": "bull_put_spread",
                                "slippage": 0.12,
                                "adverse_slippage": 0.12,
                            },
                            {
                                "strategy": "bull_put_spread",
                                "slippage": 0.08,
                                "adverse_slippage": 0.08,
                            },
                            {
                                "strategy": "iron_condor",
                                "slippage": -0.05,
                                "adverse_slippage": 0.00,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("bot.dashboard.EXECUTION_QUALITY_PATH", execution_path):
                enriched = enrich_dashboard_payload({})

            by_strategy = enriched["execution_quality"]["by_strategy"]
            self.assertAlmostEqual(
                by_strategy["bull_put_spread"]["avg_slippage"], 0.10, places=4
            )
            self.assertAlmostEqual(
                by_strategy["bull_put_spread"]["avg_adverse_slippage"], 0.10, places=4
            )
            self.assertEqual(by_strategy["bull_put_spread"]["samples"], 2)
            self.assertAlmostEqual(
                by_strategy["iron_condor"]["avg_adverse_slippage"], 0.0, places=4
            )

    def test_llm_calibration_enrichment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            llm_path = Path(tmp_dir) / "llm_track.json"
            trades = [
                {"verdict": "approve", "outcome": 10.0, "confidence": 85.0}
                for _ in range(55)
            ]
            llm_path.write_text(
                json.dumps(
                    {
                        "trades": trades,
                        "meta": {
                            "calibration": {
                                "80-90": {
                                    "actual_win_rate": 0.56,
                                    "expected_confidence": 0.85,
                                    "trades": 55,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("bot.dashboard.LLM_TRACK_RECORD_PATH", llm_path):
                enriched = enrich_dashboard_payload({})

            self.assertIn("llm_calibration", enriched)
            self.assertIn("80-90", enriched["llm_calibration"])

    def test_html_contains_sections(self) -> None:
        payload = {
            "equity_curve": [
                {"date": "2026-02-20", "equity": 100000},
                {"date": "2026-02-21", "equity": 100500},
            ],
            "monthly_pnl": {"2026-02": 500.0},
            "daily_pnl_calendar": {"2026-02-20": 150.0, "2026-02-21": -20.0},
            "strategy_breakdown": {
                "bull_put_spread": {
                    "win_rate": 60.0,
                    "avg_pnl": 50.0,
                    "total_pnl": 500.0,
                    "avg_profit": 80.0,
                    "avg_loss": -30.0,
                }
            },
            "regime_performance": {
                "BULL_TREND": {"trades": 10, "total_pnl": 300.0, "avg_pnl": 30.0}
            },
            "top_winners": [{"symbol": "SPY", "pnl": 200.0}],
            "top_losers": [{"symbol": "QQQ", "pnl": -100.0}],
            "risk_metrics": {
                "sharpe": 1.1,
                "sortino": 1.4,
                "max_drawdown": 0.05,
                "current_drawdown": 0.01,
            },
            "portfolio_greeks": {
                "delta": 10.0,
                "theta": 2.0,
                "gamma": 0.5,
                "vega": 15.0,
            },
            "sector_exposure": {"Information Technology": 40.0, "Financials": 20.0},
            "circuit_breakers": {"regime": "normal", "halt_entries": False},
            "llm_accuracy": {
                "hit_rate": 0.6,
                "approve_accuracy": 0.7,
                "reject_accuracy": 0.5,
                "reduce_size_accuracy": 0.5,
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = generate_dashboard(
                payload, output_path=Path(tmp_dir) / "dashboard.html"
            )
            html = Path(output).read_text(encoding="utf-8")
            self.assertIn("Equity Curve", html)
            self.assertIn("Monthly P&amp;L", html)
            self.assertIn("Strategy Breakdown", html)
            self.assertIn("Regime Performance", html)
            self.assertIn("Risk Metrics", html)
            self.assertIn("Circuit Breakers", html)
            self.assertIn("Monthly P&amp;L Calendar", html)


if __name__ == "__main__":
    unittest.main()
