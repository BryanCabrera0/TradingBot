import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from bot.data_store import dump_json, load_json
from bot.regime_detector import (
    BULL_TREND,
    CRASH_CRISIS,
    HIGH_VOL_CHOP,
    MarketRegimeDetector,
)


class RegimeDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = MarketRegimeDetector()

    def test_detects_crash_regime(self) -> None:
        state = self.detector.detect_from_inputs(
            {
                "vix_level": 39.0,
                "vix_term_ratio": 1.10,
                "spy_trend_score": -0.7,
                "breadth_above_50ma": 0.25,
                "put_call_ratio": 1.4,
                "vol_of_vol": 0.30,
            }
        )
        self.assertEqual(state.regime, CRASH_CRISIS)
        self.assertTrue(state.recommended_strategy_weights["iron_condors"] <= 0.1)
        self.assertLessEqual(state.recommended_position_size_scalar, 0.7)

    def test_detects_bull_trend(self) -> None:
        state = self.detector.detect_from_inputs(
            {
                "vix_level": 16.0,
                "vix_term_ratio": 0.92,
                "spy_trend_score": 0.65,
                "breadth_above_50ma": 0.75,
                "put_call_ratio": 0.82,
                "vol_of_vol": 0.08,
            }
        )
        self.assertEqual(state.regime, BULL_TREND)
        self.assertGreaterEqual(
            state.recommended_strategy_weights["credit_spreads"], 1.3
        )

    def test_detects_high_vol_chop(self) -> None:
        state = self.detector.detect_from_inputs(
            {
                "vix_level": 28.0,
                "vix_term_ratio": 0.98,
                "spy_trend_score": 0.05,
                "breadth_above_50ma": 0.50,
                "put_call_ratio": 1.0,
                "vol_of_vol": 0.30,
            }
        )
        self.assertEqual(state.regime, HIGH_VOL_CHOP)
        self.assertGreaterEqual(state.recommended_strategy_weights["iron_condors"], 1.4)

    def test_context_payload_contains_subsignals(self) -> None:
        state = self.detector.detect_from_inputs(
            {
                "vix_level": 20.0,
                "vix_term_ratio": 1.0,
                "spy_trend_score": 0.0,
                "breadth_above_50ma": 0.55,
                "put_call_ratio": 1.0,
                "vol_of_vol": 0.12,
            }
        )
        payload = state.as_context()
        self.assertIn("sub_signals", payload)
        self.assertIn("vix_level", payload["sub_signals"])
        self.assertIn("term_structure_steepness", payload["sub_signals"])

    def test_term_structure_inversion_can_force_crisis(self) -> None:
        state = self.detector.detect_from_inputs(
            {
                "vix_level": 22.0,
                "vix_term_ratio": 0.95,
                "vix_3m": 18.0,
                "vix_6m": 17.0,
                "vix_1y": 16.0,
                "term_structure_steepness": -0.08,
                "term_structure_momentum": -0.04,
                "spy_trend_score": -0.2,
                "breadth_above_50ma": 0.45,
                "put_call_ratio": 1.1,
                "vol_of_vol": 0.12,
            }
        )

        self.assertEqual(state.regime, CRASH_CRISIS)
        self.assertTrue(state.sub_signals.get("term_structure_flattening_warning"))

    def test_collect_inputs_estimates_missing_vix_curve_from_chain(self) -> None:
        get_quote = mock.Mock(
            side_effect=lambda symbol: (
                {"quote": {"lastPrice": 18.0}}
                if symbol in {"$VIX", "^VIX", "VIX"}
                else {}
            )
        )
        option_chain = {
            "callExpDateMap": {
                "2026-05-15:80": {"500.0": [{"volatility": 0.22}]},
                "2026-08-21:180": {"500.0": [{"volatility": 0.24}]},
                "2027-02-19:360": {"500.0": [{"volatility": 0.26}]},
            },
            "putExpDateMap": {
                "2026-05-15:80": {"480.0": [{"volatility": 0.23}]},
                "2026-08-21:180": {"480.0": [{"volatility": 0.25}]},
                "2027-02-19:360": {"480.0": [{"volatility": 0.27}]},
            },
        }
        detector = MarketRegimeDetector(
            get_price_history=mock.Mock(
                return_value=[{"close": 400.0 + i} for i in range(260)]
            ),
            get_quote=get_quote,
            get_option_chain=mock.Mock(return_value=option_chain),
        )

        inputs = detector._collect_inputs()

        self.assertGreater(inputs.get("vix_3m", 0.0), 0.0)
        self.assertGreater(inputs.get("vix_6m", 0.0), 0.0)
        self.assertGreater(inputs.get("vix_1y", 0.0), 0.0)
        self.assertIn("term_structure_steepness", inputs)
        self.assertIn("term_structure_momentum", inputs)

    def test_detect_uses_cache_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = Path(tmp_dir) / "regime_history.json"
            get_price_history = mock.Mock(
                return_value=[{"close": 400.0 + i * 0.2} for i in range(300)]
            )
            get_quote = mock.Mock(return_value={"quote": {"lastPrice": 18.0}})
            get_chain = mock.Mock(return_value={"calls": {}, "puts": {}})
            detector = MarketRegimeDetector(
                get_price_history=get_price_history,
                get_quote=get_quote,
                get_option_chain=get_chain,
                config={"cache_seconds": 3600, "history_file": str(history)},
            )

            first = detector.detect()
            calls_after_first = get_price_history.call_count
            second = detector.detect()

            self.assertEqual(first.regime, second.regime)
            self.assertEqual(get_price_history.call_count, calls_after_first)

    def test_persist_history_and_momentum_boost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = Path(tmp_dir) / "regime_history.json"
            base_ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
            dump_json(
                history,
                {
                    "entries": [
                        {
                            "timestamp": (base_ts).isoformat().replace("+00:00", "Z"),
                            "regime": "BULL_TREND",
                            "confidence": 0.6,
                            "sub_signals": {},
                        },
                        {
                            "timestamp": (base_ts).isoformat().replace("+00:00", "Z"),
                            "regime": "BULL_TREND",
                            "confidence": 0.6,
                            "sub_signals": {},
                        },
                        {
                            "timestamp": (base_ts).isoformat().replace("+00:00", "Z"),
                            "regime": "BULL_TREND",
                            "confidence": 0.6,
                            "sub_signals": {},
                        },
                    ]
                },
            )
            detector = MarketRegimeDetector(
                config={"cache_seconds": 0, "history_file": str(history)}
            )
            state = detector.detect_from_inputs(
                {
                    "vix_level": 16.0,
                    "vix_term_ratio": 0.92,
                    "spy_trend_score": 0.65,
                    "breadth_above_50ma": 0.75,
                    "put_call_ratio": 0.82,
                    "vol_of_vol": 0.08,
                }
            )
            boosted = detector._apply_regime_momentum(state)
            detector._persist_history(boosted)

            self.assertGreaterEqual(boosted.confidence, state.confidence)
            self.assertIn("regime_momentum", boosted.sub_signals)
            payload = load_json(history, {"entries": []})
            self.assertGreaterEqual(len(payload.get("entries", [])), 1)

    def test_transition_marker_added_on_regime_change(self) -> None:
        detector = MarketRegimeDetector(config={"cache_seconds": 0})
        first = detector.detect_from_inputs(
            {
                "vix_level": 16.0,
                "vix_term_ratio": 0.92,
                "spy_trend_score": 0.65,
                "breadth_above_50ma": 0.75,
                "put_call_ratio": 0.82,
                "vol_of_vol": 0.08,
            }
        )
        detector._last_regime = first.regime
        second = detector.detect_from_inputs(
            {
                "vix_level": 39.0,
                "vix_term_ratio": 1.10,
                "spy_trend_score": -0.7,
                "breadth_above_50ma": 0.25,
                "put_call_ratio": 1.4,
                "vol_of_vol": 0.30,
            }
        )
        second = detector._apply_regime_momentum(second)
        if detector._last_regime and detector._last_regime != second.regime:
            second.sub_signals["transition_from"] = detector._last_regime

        self.assertEqual(second.regime, CRASH_CRISIS)
        self.assertEqual(second.sub_signals.get("transition_from"), BULL_TREND)


if __name__ == "__main__":
    unittest.main()
