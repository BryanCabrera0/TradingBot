import tempfile
import unittest
from pathlib import Path

from bot.data_store import load_json
from bot.pnl_attribution import PnLAttributionEngine


class PnLAttributionTests(unittest.TestCase):
    def test_compute_attribution_decomposes_components(self) -> None:
        engine = PnLAttributionEngine()
        positions = [
            {
                "position_id": "p1",
                "symbol": "SPY",
                "strategy": "bull_put_spread",
                "quantity": 2,
                "entry_credit": 1.5,
                "current_value": 1.0,
                "details": {
                    "net_delta": 3.0,
                    "net_gamma": 0.2,
                    "net_theta": 0.1,
                    "net_vega": 1.5,
                    "net_rho": 0.5,
                    "days_elapsed": 1.0,
                    "rate_change": 0.02,
                },
            }
        ]
        result = engine.compute_attribution(
            positions=positions,
            price_changes={"SPY": 2.0},
            iv_changes={"SPY": 0.5},
        )
        row = result["positions"][0]
        self.assertAlmostEqual(row["delta_pnl"], 12.0, places=6)
        self.assertAlmostEqual(row["gamma_pnl"], 0.8, places=6)
        self.assertAlmostEqual(row["theta_pnl"], 0.2, places=6)
        self.assertAlmostEqual(row["vega_pnl"], 1.5, places=6)
        self.assertAlmostEqual(row["rho_pnl"], 0.02, places=6)

    def test_zero_inputs_produce_zero_components(self) -> None:
        engine = PnLAttributionEngine()
        positions = [
            {
                "position_id": "p2",
                "symbol": "QQQ",
                "quantity": 1,
                "entry_credit": 1.0,
                "current_value": 1.0,
                "details": {
                    "net_delta": 0.0,
                    "net_gamma": 0.0,
                    "net_theta": 0.0,
                    "net_vega": 0.0,
                },
            }
        ]
        result = engine.compute_attribution(positions, {"QQQ": 0.0}, {"QQQ": 0.0})
        row = result["positions"][0]
        self.assertEqual(row["delta_pnl"], 0.0)
        self.assertEqual(row["gamma_pnl"], 0.0)
        self.assertEqual(row["theta_pnl"], 0.0)
        self.assertEqual(row["vega_pnl"], 0.0)

    def test_record_daily_snapshot_appends_by_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pnl_attribution.json"
            engine = PnLAttributionEngine(path=str(path))
            entry = {"portfolio": {"delta_pnl": 10.0}}
            engine.record_daily_snapshot("2026-02-26", entry)
            engine.record_daily_snapshot("2026-02-27", entry)
            payload = load_json(path, {})
            history = payload.get("history", [])
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["date"], "2026-02-26")
            self.assertEqual(history[1]["date"], "2026-02-27")


if __name__ == "__main__":
    unittest.main()
