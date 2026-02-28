"""Lightweight gradient-boosted signal scorer trained from closed trade history."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from math import exp, log
from pathlib import Path
from typing import Optional

import numpy as np

from bot.data_store import dump_json, ensure_data_dir, load_json
from bot.number_utils import safe_float, safe_int

logger = logging.getLogger(__name__)

DEFAULT_CLOSED_TRADES_PATH = Path("bot/data/closed_trades.json")
DEFAULT_MODEL_PATH = Path("bot/data/ml_model.json")
DEFAULT_IMPORTANCE_PATH = Path("bot/data/ml_feature_importance.json")

CATEGORICAL_FEATURES = [
    "strategy_type",
    "regime",
    "term_structure_regime",
    "flow_bias",
    "day_of_week",
    "time_of_day_bucket",
    "sector",
]
NUMERIC_FEATURES = [
    "iv_rank",
    "spread_score",
    "dte",
    "delta",
    "credit_to_width_ratio",
    "news_sentiment_score",
    "econ_calendar_proximity_days",
    "vix_level",
    "put_call_ratio",
]


@dataclass
class MLScorerConfig:
    enabled: bool = True
    min_training_trades: int = 30
    retrain_day: str = "sunday"
    retrain_time: str = "18:00"
    feature_importance_log: bool = True
    closed_trades_file: str = str(DEFAULT_CLOSED_TRADES_PATH)
    model_file: str = str(DEFAULT_MODEL_PATH)
    feature_importance_file: str = str(DEFAULT_IMPORTANCE_PATH)


class MLSignalScorer:
    """Gradient-boosted stump ensemble for trade outcome prediction."""

    def __init__(self, config: Optional[dict] = None):
        raw = config or {}
        self.config = MLScorerConfig(
            enabled=bool(raw.get("enabled", True)),
            min_training_trades=max(1, safe_int(raw.get("min_training_trades"), 30)),
            retrain_day=str(raw.get("retrain_day", "sunday")).strip().lower()
            or "sunday",
            retrain_time=str(raw.get("retrain_time", "18:00")).strip() or "18:00",
            feature_importance_log=bool(raw.get("feature_importance_log", True)),
            closed_trades_file=str(
                raw.get("closed_trades_file", DEFAULT_CLOSED_TRADES_PATH)
            ),
            model_file=str(raw.get("model_file", DEFAULT_MODEL_PATH)),
            feature_importance_file=str(
                raw.get("feature_importance_file", DEFAULT_IMPORTANCE_PATH)
            ),
        )
        ensure_data_dir(Path(self.config.model_file).parent)
        ensure_data_dir(Path(self.config.feature_importance_file).parent)
        ensure_data_dir(Path(self.config.closed_trades_file).parent)
        payload = load_json(Path(self.config.model_file), {})
        self._model: dict = payload if isinstance(payload, dict) else {}

    def is_active(self) -> bool:
        """Return True when model is enabled and trained with enough trades."""
        if not self.config.enabled:
            return False
        sample_size = safe_int(self._model.get("sample_size"), 0)
        return sample_size >= int(self.config.min_training_trades) and bool(
            self._model.get("stumps")
        )

    def load_closed_trades(self) -> list[dict]:
        """Load persisted closed trades from disk."""
        payload = load_json(Path(self.config.closed_trades_file), {"trades": []})
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            raw_rows = payload.get(
                "trades", payload.get("closed_trades", payload.get("positions", []))
            )
            rows = raw_rows if isinstance(raw_rows, list) else []
        else:
            rows = []
        trades = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "")).lower()
            if status and status not in {"closed", "rolled", "closed_external"}:
                continue
            trades.append(row)
        return trades

    def retrain_from_file(self) -> bool:
        """Train model from persisted closed-trade file."""
        trades = self.load_closed_trades()
        return self.train(trades)

    def train(self, trades: list[dict]) -> bool:
        """Fit a boosted-stump model from closed trades."""
        if not self.config.enabled:
            return False
        if not isinstance(trades, list) or len(trades) < int(
            self.config.min_training_trades
        ):
            logger.info(
                "ML scorer training skipped: %d trades < minimum %d",
                len(trades) if isinstance(trades, list) else 0,
                self.config.min_training_trades,
            )
            return False

        feature_rows: list[dict] = []
        y_binary: list[float] = []
        y_continuous: list[float] = []
        for trade in trades:
            row = self.extract_trade_features(trade)
            pnl = safe_float(trade.get("pnl", trade.get("realized_pnl", 0.0)), 0.0)
            max_loss = (
                abs(safe_float(trade.get("max_loss"), 0.0))
                * max(1.0, safe_float(trade.get("quantity"), 1.0))
                * 100.0
            )
            pnl_pct = safe_float(trade.get("pnl_pct"), 0.0)
            if pnl_pct == 0.0 and max_loss > 0:
                pnl_pct = pnl / max_loss
            row["pnl_pct"] = pnl_pct
            feature_rows.append(row)
            y_binary.append(1.0 if pnl > 0 else 0.0)
            y_continuous.append(pnl_pct)

        X, feature_names, category_maps = self._matrix_from_rows(feature_rows)
        y = np.asarray(y_binary, dtype=float)
        y_reg = np.asarray(y_continuous, dtype=float)
        if X.shape[0] < int(self.config.min_training_trades):
            return False

        model, importances = _fit_gradient_boosted_stumps(
            X=X,
            y=y,
            n_estimators=50,
            learning_rate=0.1,
        )
        reg_model, _ = _fit_gradient_boosted_stumps_regression(
            X=X,
            y=y_reg,
            n_estimators=50,
            learning_rate=0.05,
        )
        payload = {
            "version": 1,
            "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "sample_size": int(X.shape[0]),
            "feature_names": feature_names,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "category_maps": category_maps,
            "base_logit": model["base"],
            "learning_rate": model["learning_rate"],
            "stumps": model["stumps"],
            "pnl_regression": reg_model,
            "min_training_trades": int(self.config.min_training_trades),
        }
        dump_json(Path(self.config.model_file), payload)
        self._model = payload
        if self.config.feature_importance_log:
            self._log_feature_importance(feature_names, importances)
        logger.info(
            "ML scorer trained with %d samples and %d features.", X.shape[0], X.shape[1]
        )
        return True

    def predict_score(self, features: dict) -> float:
        """Return predicted trade profitability probability in [0, 1]."""
        if not self.is_active():
            return 0.5
        if not isinstance(features, dict):
            return 0.5
        feature_names = self._model.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            return 0.5

        row = self.normalize_feature_input(features)
        x = self._vectorize_row(
            row=row,
            feature_names=feature_names,
            category_maps=self._model.get("category_maps", {}),
        )
        if x is None:
            return 0.5
        logit = safe_float(self._model.get("base_logit"), 0.0)
        learning_rate = safe_float(self._model.get("learning_rate"), 0.1)
        stumps = self._model.get("stumps", [])
        if not isinstance(stumps, list):
            return 0.5
        for stump in stumps:
            if not isinstance(stump, dict):
                continue
            idx = safe_int(stump.get("feature"), -1)
            if idx < 0 or idx >= len(x):
                continue
            threshold = safe_float(stump.get("threshold"), 0.0)
            left = safe_float(stump.get("left"), 0.0)
            right = safe_float(stump.get("right"), 0.0)
            logit += learning_rate * (left if x[idx] <= threshold else right)
        prob = 1.0 / (1.0 + exp(-max(-20.0, min(20.0, logit))))
        return float(max(0.0, min(1.0, prob)))

    def extract_trade_features(self, trade: dict) -> dict:
        """Extract normalized model features from a closed-trade record."""
        details = (
            trade.get("details", {}) if isinstance(trade.get("details"), dict) else {}
        )
        vol_surface = (
            details.get("vol_surface", {})
            if isinstance(details.get("vol_surface"), dict)
            else {}
        )
        flow = (
            details.get("options_flow", {})
            if isinstance(details.get("options_flow"), dict)
            else {}
        )
        timestamp = str(
            trade.get("open_date", trade.get("entry_time", trade.get("created_at", "")))
        )
        day_of_week, time_bucket = _time_buckets(timestamp)

        score = safe_float(
            details.get(
                "score", trade.get("score", details.get("analysis_score", 0.0))
            ),
            0.0,
        )
        width = safe_float(
            details.get(
                "spread_width", details.get("width", trade.get("spread_width", 0.0))
            ),
            0.0,
        )
        credit = safe_float(trade.get("entry_credit", details.get("credit", 0.0)), 0.0)
        credit_to_width = safe_float(details.get("credit_pct_of_width"), 0.0)
        if credit_to_width <= 0 and width > 0:
            credit_to_width = credit / width

        news_score = safe_float(details.get("news_sentiment_score"), 0.0)
        if news_score == 0.0:
            sentiment = str(
                details.get("news_sentiment", details.get("sentiment", "neutral"))
            ).lower()
            news_score = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}.get(
                sentiment, 0.0
            )

        return {
            "strategy_type": str(
                trade.get("strategy", details.get("strategy", "unknown"))
            ).lower(),
            "regime": str(
                trade.get("regime", details.get("regime", "unknown"))
            ).lower(),
            "iv_rank": safe_float(
                details.get("iv_rank", trade.get("iv_rank", 0.0)), 0.0
            ),
            "term_structure_regime": str(
                vol_surface.get(
                    "term_structure_regime",
                    details.get("term_structure_regime", "flat"),
                )
            ).lower(),
            "flow_bias": str(
                flow.get("directional_bias", details.get("directional_bias", "neutral"))
            ).lower(),
            "spread_score": score,
            "dte": safe_float(
                details.get("dte", trade.get("dte_remaining", trade.get("dte", 0.0))),
                0.0,
            ),
            "delta": safe_float(
                details.get(
                    "short_delta", details.get("net_delta", trade.get("delta", 0.0))
                ),
                0.0,
            ),
            "credit_to_width_ratio": credit_to_width,
            "day_of_week": day_of_week,
            "time_of_day_bucket": time_bucket,
            "sector": str(
                details.get("sector", trade.get("sector", "unknown"))
            ).lower(),
            "news_sentiment_score": news_score,
            "econ_calendar_proximity_days": safe_float(
                details.get(
                    "econ_calendar_proximity_days",
                    details.get("econ_days_to_event", 99.0),
                ),
                99.0,
            ),
            "vix_level": safe_float(
                details.get("vix_level", trade.get("vix_level", 20.0)), 20.0
            ),
            "put_call_ratio": safe_float(
                details.get("put_call_ratio", trade.get("put_call_ratio", 1.0)), 1.0
            ),
        }

    def normalize_feature_input(self, features: dict) -> dict:
        """Normalize arbitrary feature dict into the model input schema."""
        normalized = dict(features)
        if "strategy" in normalized and "strategy_type" not in normalized:
            normalized["strategy_type"] = normalized.get("strategy")
        if "flow_directional_bias" in normalized and "flow_bias" not in normalized:
            normalized["flow_bias"] = normalized.get("flow_directional_bias")
        if (
            "credit_pct_of_width" in normalized
            and "credit_to_width_ratio" not in normalized
        ):
            normalized["credit_to_width_ratio"] = normalized.get("credit_pct_of_width")
        if "time_bucket" in normalized and "time_of_day_bucket" not in normalized:
            normalized["time_of_day_bucket"] = normalized.get("time_bucket")
        if "news_sentiment" in normalized and "news_sentiment_score" not in normalized:
            sentiment = str(normalized.get("news_sentiment", "neutral")).lower()
            normalized["news_sentiment_score"] = {
                "bullish": 1.0,
                "neutral": 0.0,
                "bearish": -1.0,
            }.get(sentiment, 0.0)
        if "entry_timestamp" in normalized:
            day_of_week, time_bucket = _time_buckets(
                str(normalized.get("entry_timestamp"))
            )
            normalized.setdefault("day_of_week", day_of_week)
            normalized.setdefault("time_of_day_bucket", time_bucket)

        for key in CATEGORICAL_FEATURES:
            normalized[key] = (
                str(normalized.get(key, "unknown")).strip().lower() or "unknown"
            )
        for key in NUMERIC_FEATURES:
            normalized[key] = safe_float(normalized.get(key), 0.0)
        return normalized

    def _matrix_from_rows(self, rows: list[dict]) -> tuple[np.ndarray, list[str], dict]:
        category_maps: dict[str, list[str]] = {}
        for feature in CATEGORICAL_FEATURES:
            values = sorted(
                {
                    str((row or {}).get(feature, "unknown")).strip().lower()
                    or "unknown"
                    for row in rows
                }
            )
            category_maps[feature] = values or ["unknown"]

        feature_names: list[str] = []
        for feature in NUMERIC_FEATURES:
            feature_names.append(feature)
        for feature in CATEGORICAL_FEATURES:
            for category in category_maps.get(feature, []):
                feature_names.append(f"{feature}={category}")

        matrix = []
        for row in rows:
            vec = self._vectorize_row(
                row=self.normalize_feature_input(row),
                feature_names=feature_names,
                category_maps=category_maps,
            )
            matrix.append(vec if vec is not None else [0.0] * len(feature_names))
        return np.asarray(matrix, dtype=float), feature_names, category_maps

    def _vectorize_row(
        self, *, row: dict, feature_names: list[str], category_maps: dict
    ) -> Optional[list[float]]:
        if not isinstance(row, dict):
            return None
        vec: list[float] = []
        for feature in NUMERIC_FEATURES:
            vec.append(safe_float(row.get(feature), 0.0))
        for feature in CATEGORICAL_FEATURES:
            value = str(row.get(feature, "unknown")).strip().lower() or "unknown"
            for category in category_maps.get(feature, []):
                vec.append(1.0 if value == category else 0.0)
        if len(vec) != len(feature_names):
            return None
        return vec

    def _log_feature_importance(
        self, feature_names: list[str], importances: np.ndarray
    ) -> None:
        total = float(np.sum(np.abs(importances)))
        rows = []
        for idx, name in enumerate(feature_names):
            value = safe_float(importances[idx] if idx < len(importances) else 0.0, 0.0)
            normalized = (value / total) if total > 0 else 0.0
            rows.append({"feature": name, "importance": normalized})
        rows.sort(key=lambda row: safe_float(row.get("importance"), 0.0), reverse=True)

        path = Path(self.config.feature_importance_file)
        payload = load_json(path, {"history": []})
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            history = []
        history.append(
            {
                "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "sample_size": safe_int(self._model.get("sample_size"), 0),
                "features": rows[:50],
            }
        )
        payload = {"history": history[-200:], "latest": rows[:50]}
        dump_json(path, payload)


def _fit_gradient_boosted_stumps(
    *,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    learning_rate: float,
) -> tuple[dict, np.ndarray]:
    n_rows, n_cols = X.shape
    pos_rate = float(np.mean(y))
    pos_rate = min(max(pos_rate, 1e-6), 1 - 1e-6)
    base = log(pos_rate / (1.0 - pos_rate))
    scores = np.full(n_rows, base, dtype=float)
    stumps: list[dict] = []
    importances = np.zeros(n_cols, dtype=float)

    for _ in range(max(1, n_estimators)):
        prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
        residual = y - prob
        baseline_loss = float(np.mean(residual**2))
        best = _best_stump(X, residual, baseline_loss=baseline_loss)
        if best is None:
            continue
        feature_idx, threshold, left_val, right_val, gain = best
        pred = np.where(X[:, feature_idx] <= threshold, left_val, right_val)
        scores += learning_rate * pred
        stumps.append(
            {
                "feature": int(feature_idx),
                "threshold": float(threshold),
                "left": float(left_val),
                "right": float(right_val),
                "gain": float(gain),
            }
        )
        importances[feature_idx] += max(0.0, float(gain))

    model = {"base": base, "learning_rate": float(learning_rate), "stumps": stumps}
    return model, importances


def _fit_gradient_boosted_stumps_regression(
    *,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    learning_rate: float,
) -> tuple[dict, np.ndarray]:
    n_rows, n_cols = X.shape
    base = float(np.mean(y)) if n_rows else 0.0
    scores = np.full(n_rows, base, dtype=float)
    stumps: list[dict] = []
    importances = np.zeros(n_cols, dtype=float)

    for _ in range(max(1, n_estimators)):
        residual = y - scores
        baseline_loss = float(np.mean(residual**2))
        best = _best_stump(X, residual, baseline_loss=baseline_loss)
        if best is None:
            continue
        feature_idx, threshold, left_val, right_val, gain = best
        pred = np.where(X[:, feature_idx] <= threshold, left_val, right_val)
        scores += learning_rate * pred
        stumps.append(
            {
                "feature": int(feature_idx),
                "threshold": float(threshold),
                "left": float(left_val),
                "right": float(right_val),
                "gain": float(gain),
            }
        )
        importances[feature_idx] += max(0.0, float(gain))

    model = {"base": base, "learning_rate": float(learning_rate), "stumps": stumps}
    return model, importances


def _best_stump(
    X: np.ndarray,
    residual: np.ndarray,
    *,
    baseline_loss: float,
) -> Optional[tuple[int, float, float, float, float]]:
    best: Optional[tuple[int, float, float, float, float]] = None
    n_rows, n_cols = X.shape
    if n_rows <= 1:
        return None
    for col in range(n_cols):
        values = X[:, col]
        if np.all(values == values[0]):
            continue
        quantiles = np.quantile(values, np.linspace(0.1, 0.9, 9))
        thresholds = sorted({float(item) for item in quantiles})
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask
            if not np.any(left_mask) or not np.any(right_mask):
                continue
            left_val = float(np.mean(residual[left_mask]))
            right_val = float(np.mean(residual[right_mask]))
            pred = np.where(left_mask, left_val, right_val)
            loss = float(np.mean((residual - pred) ** 2))
            gain = baseline_loss - loss
            if best is None or gain > best[4]:
                best = (col, threshold, left_val, right_val, gain)
    return best


def _time_buckets(timestamp: str) -> tuple[str, str]:
    text = str(timestamp or "").strip()
    if not text:
        return "unknown", "unknown"
    dt = None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            dt = None
    if dt is None:
        return "unknown", "unknown"
    day = dt.strftime("%A").lower()
    hour = dt.hour
    minute = dt.minute
    total_minutes = (hour * 60) + minute
    if total_minutes < 600:
        bucket = "premarket"
    elif total_minutes < 660:
        bucket = "open_hour"
    elif total_minutes < 780:
        bucket = "morning"
    elif total_minutes < 870:
        bucket = "midday"
    elif total_minutes < 930:
        bucket = "afternoon"
    else:
        bucket = "close_hour"
    return day, bucket
