import tempfile
import unittest
from pathlib import Path

from bot.data_store import dump_json, load_json
from bot.ml_scorer import MLSignalScorer


def _sample_trade(idx: int, *, win: bool) -> dict:
    pnl = 120.0 if win else -90.0
    return {
        "strategy": "bull_put_spread" if idx % 2 == 0 else "iron_condor",
        "symbol": "SPY" if idx % 3 == 0 else "QQQ",
        "status": "closed",
        "open_date": f"2026-02-{(idx % 20) + 1:02d}T{10 + (idx % 6):02d}:15:00-05:00",
        "close_date": f"2026-02-{(idx % 20) + 2:02d}",
        "pnl": pnl,
        "max_loss": 2.0,
        "quantity": 1,
        "entry_credit": 1.2,
        "details": {
            "regime": "bull_trend" if idx % 2 == 0 else "high_vol_chop",
            "iv_rank": 55 + (idx % 15),
            "score": 50 + (idx % 20),
            "dte": 20 + (idx % 15),
            "short_delta": 0.20 + ((idx % 4) * 0.02),
            "spread_width": 5.0,
            "sector": "information technology" if idx % 2 == 0 else "financials",
            "news_sentiment_score": 0.6 if win else -0.4,
            "econ_calendar_proximity_days": 5 + (idx % 10),
            "vix_level": 16.0 + (idx % 8),
            "put_call_ratio": 0.95 + ((idx % 5) * 0.05),
            "vol_surface": {
                "term_structure_regime": "backwardation" if win else "contango"
            },
            "options_flow": {"directional_bias": "bullish" if win else "bearish"},
        },
    }


class MLScorerTests(unittest.TestCase):
    def test_predict_returns_neutral_when_not_trained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scorer = MLSignalScorer(
                {
                    "enabled": True,
                    "min_training_trades": 30,
                    "closed_trades_file": str(Path(tmp_dir) / "closed_trades.json"),
                    "model_file": str(Path(tmp_dir) / "ml_model.json"),
                    "feature_importance_file": str(
                        Path(tmp_dir) / "ml_feature_importance.json"
                    ),
                }
            )
            self.assertEqual(
                scorer.predict_score({"strategy_type": "bull_put_spread"}), 0.5
            )

    def test_train_and_predict_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scorer = MLSignalScorer(
                {
                    "enabled": True,
                    "min_training_trades": 30,
                    "closed_trades_file": str(Path(tmp_dir) / "closed_trades.json"),
                    "model_file": str(Path(tmp_dir) / "ml_model.json"),
                    "feature_importance_file": str(
                        Path(tmp_dir) / "ml_feature_importance.json"
                    ),
                }
            )
            trades = [_sample_trade(idx, win=(idx % 3 != 0)) for idx in range(45)]
            trained = scorer.train(trades)
            self.assertTrue(trained)
            self.assertTrue(Path(tmp_dir, "ml_model.json").exists())
            prob = scorer.predict_score(
                {
                    "strategy_type": "bull_put_spread",
                    "regime": "bull_trend",
                    "iv_rank": 68,
                    "term_structure_regime": "backwardation",
                    "flow_bias": "bullish",
                    "spread_score": 72,
                    "dte": 25,
                    "delta": 0.22,
                    "credit_to_width_ratio": 0.28,
                    "day_of_week": "wednesday",
                    "time_of_day_bucket": "midday",
                    "sector": "information technology",
                    "news_sentiment_score": 0.8,
                    "econ_calendar_proximity_days": 8,
                    "vix_level": 18,
                    "put_call_ratio": 0.95,
                }
            )
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertNotEqual(prob, 0.5)

    def test_retrain_from_file_persists_feature_importance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            closed_path = Path(tmp_dir) / "closed_trades.json"
            model_path = Path(tmp_dir) / "ml_model.json"
            importance_path = Path(tmp_dir) / "ml_feature_importance.json"
            trades = [_sample_trade(idx, win=(idx % 2 == 0)) for idx in range(40)]
            dump_json(closed_path, {"trades": trades})

            scorer = MLSignalScorer(
                {
                    "enabled": True,
                    "min_training_trades": 30,
                    "closed_trades_file": str(closed_path),
                    "model_file": str(model_path),
                    "feature_importance_file": str(importance_path),
                    "feature_importance_log": True,
                }
            )
            self.assertTrue(scorer.retrain_from_file())
            self.assertTrue(model_path.exists())
            self.assertTrue(importance_path.exists())
            payload = load_json(importance_path, {})
            self.assertIn("latest", payload)
            self.assertTrue(len(payload.get("latest", [])) > 0)

    def test_extract_trade_features_maps_timestamp_to_buckets(self) -> None:
        scorer = MLSignalScorer({"enabled": True})
        features = scorer.extract_trade_features(_sample_trade(1, win=True))
        self.assertIn(
            features.get("day_of_week"),
            {"monday", "tuesday", "wednesday", "thursday", "friday", "unknown"},
        )
        self.assertIn(
            features.get("time_of_day_bucket"),
            {
                "premarket",
                "open_hour",
                "morning",
                "midday",
                "afternoon",
                "close_hour",
                "unknown",
            },
        )


if __name__ == "__main__":
    unittest.main()
