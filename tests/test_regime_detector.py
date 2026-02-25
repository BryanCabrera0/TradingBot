import unittest

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
        self.assertGreaterEqual(state.recommended_strategy_weights["credit_spreads"], 1.3)

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


if __name__ == "__main__":
    unittest.main()

