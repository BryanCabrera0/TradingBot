import tempfile
import unittest
from pathlib import Path

from bot.iv_history import IVHistory
from bot.vol_surface import VolSurfaceAnalyzer


def _chain(front_iv: float = 30.0, back_iv: float = 24.0) -> dict:
    return {
        "calls": {
            "2026-03-20": [
                {"dte": 25, "strike": 105, "iv": front_iv, "delta": 0.25, "mid": 1.2},
                {"dte": 25, "strike": 100, "iv": front_iv - 1.0, "delta": 0.50, "mid": 2.4},
            ],
            "2026-05-15": [
                {"dte": 81, "strike": 105, "iv": back_iv, "delta": 0.25, "mid": 1.6},
                {"dte": 81, "strike": 100, "iv": back_iv - 1.0, "delta": 0.50, "mid": 3.0},
            ],
        },
        "puts": {
            "2026-03-20": [
                {"dte": 25, "strike": 95, "iv": front_iv + 2.0, "delta": -0.25, "mid": 1.1},
                {"dte": 25, "strike": 100, "iv": front_iv - 1.0, "delta": -0.50, "mid": 2.3},
            ],
            "2026-05-15": [
                {"dte": 81, "strike": 95, "iv": back_iv + 1.0, "delta": -0.25, "mid": 1.5},
                {"dte": 81, "strike": 100, "iv": back_iv - 1.0, "delta": -0.50, "mid": 2.9},
            ],
        },
    }


class VolSurfaceTests(unittest.TestCase):
    def test_analyzer_flags_positive_vol_risk_premium(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = IVHistory(Path(tmp_dir) / "iv_history.json")
            analyzer = VolSurfaceAnalyzer(history)
            price_history = [{"close": v} for v in [100, 101, 99, 102, 100, 101, 100, 99, 101, 100, 102, 101, 100]]

            context = analyzer.analyze(symbol="SPY", chain_data=_chain(32.0, 25.0), price_history=price_history)

            self.assertTrue(context.flags["positive_vol_risk_premium"])
            self.assertEqual(context.term_structure_regime, "backwardation")

    def test_skew_regime_detects_put_skew(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = IVHistory(Path(tmp_dir) / "iv_history.json")
            analyzer = VolSurfaceAnalyzer(history)

            context = analyzer.analyze(symbol="SPY", chain_data=_chain(35.0, 28.0), price_history=[])

            self.assertGreater(context.put_call_skew, 0.0)
            self.assertIn(context.skew_regime, {"put_skew_fear", "flat"})

    def test_term_structure_flat_when_insufficient_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = IVHistory(Path(tmp_dir) / "iv_history.json")
            analyzer = VolSurfaceAnalyzer(history)
            chain_data = {"calls": {"2026-03-20": [{"dte": 25, "iv": 22.0, "delta": 0.2}]}, "puts": {}}

            context = analyzer.analyze(symbol="QQQ", chain_data=chain_data, price_history=[])

            self.assertEqual(context.term_structure_regime, "flat")
            self.assertAlmostEqual(context.term_structure_front_back, 0.0, places=4)

    def test_iv_rank_and_percentile_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = IVHistory(Path(tmp_dir) / "iv_history.json")
            analyzer = VolSurfaceAnalyzer(history)
            context = analyzer.analyze(symbol="AAPL", chain_data=_chain(40.0, 30.0), price_history=[])

            self.assertGreaterEqual(context.iv_rank, 0.0)
            self.assertLessEqual(context.iv_rank, 100.0)
            self.assertGreaterEqual(context.iv_percentile, 0.0)
            self.assertLessEqual(context.iv_percentile, 100.0)


if __name__ == "__main__":
    unittest.main()

