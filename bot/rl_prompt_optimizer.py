"""Reinforcement-style prompt optimizer that learns trade rules from outcomes.

SIMPLE EXPLANATION:
The RL Prompt Optimizer is the "continuous learning" agent. It reviews past trades 
to see what worked and what lost money. Based on those results, it automatically writes 
new rules (like "stop trading tech stocks before CPI data") to improve the other agents' 
prompts in the future, helping the bot get smarter over time.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from bot.config import RLPromptOptimizerConfig
from bot.data_store import dump_json, ensure_data_dir, load_json
from bot.pnl_attribution import DEFAULT_ATTRIBUTION_PATH

DEFAULT_RULES_PATH = Path("bot/data/learned_rules.json")
DEFAULT_EXPLANATIONS_PATH = Path("bot/data/trade_explanations.json")
DEFAULT_TRACK_RECORD_PATH = Path("bot/data/llm_track_record.json")
DEFAULT_AUDIT_LOG_PATH = Path("bot/data/audit_log.jsonl")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_active_rules(
    path: Path | str = DEFAULT_RULES_PATH, *, limit: Optional[int] = None
) -> list[str]:
    """Load learned rule text lines for prompt injection."""
    payload = load_json(path, {"rules": []})
    rows = payload.get("rules", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    rules = [
        str(item.get("rule", "")).strip()
        for item in rows
        if isinstance(item, dict) and str(item.get("rule", "")).strip()
    ]
    if limit is None:
        return rules
    return rules[-max(0, int(limit)) :]


class RLPromptOptimizer:
    """Learn from closed-trade outcomes and maintain rule memory for the CIO."""

    def __init__(
        self,
        config: RLPromptOptimizerConfig,
        *,
        explanations_path: Path | str = DEFAULT_EXPLANATIONS_PATH,
        track_record_path: Path | str = DEFAULT_TRACK_RECORD_PATH,
        attribution_path: Path | str = DEFAULT_ATTRIBUTION_PATH,
    ):
        self.config = config
        self.rules_path = Path(config.learned_rules_file or DEFAULT_RULES_PATH)
        self.audit_path = Path(config.audit_log_file or DEFAULT_AUDIT_LOG_PATH)
        self.explanations_path = Path(explanations_path)
        self.track_record_path = Path(track_record_path)
        self.attribution_path = Path(attribution_path)
        ensure_data_dir(self.rules_path.parent)
        ensure_data_dir(self.audit_path.parent)

    def process_closed_trade(
        self,
        *,
        position_id: str,
        pnl: float,
        trade_context: Optional[dict] = None,
    ) -> dict:
        """Ingest one closed trade and update learned rules if patterns emerge."""
        if not str(position_id or "").strip():
            return {"created_rules": [], "pruned_rules": [], "detected_patterns": []}

        payload = self._load_rules_payload()
        closed_rows = payload.get("recent_closed_trades", [])
        if not isinstance(closed_rows, list):
            closed_rows = []

        row = self._build_closed_trade_row(
            position_id=str(position_id),
            pnl=float(pnl),
            trade_context=trade_context or {},
        )
        closed_rows.append(row)
        rolling_window_size = max(20, int(self.config.rolling_window_size or 100))
        payload["recent_closed_trades"] = closed_rows[-rolling_window_size:]

        patterns = self._detect_patterns(payload["recent_closed_trades"])
        created = self._upsert_rules(payload, patterns)
        pruned = self._prune_rules(payload)

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            payload["meta"] = meta
        meta["last_updated"] = utc_now_iso()

        dump_json(self.rules_path, payload)

        for item in created:
            self._append_audit(
                "rl_rule_created",
                {
                    "rule_id": item.get("rule_id"),
                    "pattern_key": item.get("pattern_key"),
                    "rule": item.get("rule"),
                    "stats": item.get("stats", {}),
                },
            )
        for item in pruned:
            self._append_audit(
                "rl_rule_pruned",
                {
                    "rule_id": item.get("rule_id"),
                    "pattern_key": item.get("pattern_key"),
                    "rule": item.get("rule"),
                },
            )

        return {
            "created_rules": created,
            "pruned_rules": pruned,
            "detected_patterns": patterns,
        }

    def _build_closed_trade_row(
        self, *, position_id: str, pnl: float, trade_context: dict
    ) -> dict:
        explanations_payload = load_json(self.explanations_path, {"positions": {}})
        positions = (
            explanations_payload.get("positions", {})
            if isinstance(explanations_payload, dict)
            else {}
        )
        explanation = (
            positions.get(position_id, {}) if isinstance(positions, dict) else {}
        )

        track_payload = load_json(self.track_record_path, {"trades": []})
        track_rows = (
            track_payload.get("trades", []) if isinstance(track_payload, dict) else []
        )
        track = self._find_track_row(track_rows, position_id)

        attribution = self._find_latest_attribution(position_id)

        context = (
            track.get("context", {}) if isinstance(track.get("context"), dict) else {}
        )
        earnings_proximity = (
            context.get("earnings_proximity", {})
            if isinstance(context.get("earnings_proximity"), dict)
            else {}
        )

        regime = (
            str(trade_context.get("regime", "")).upper()
            or str(context.get("regime", "")).upper()
            or str(track.get("regime", "")).upper()
            or "UNKNOWN"
        )
        strategy = (
            str(trade_context.get("strategy", "")).strip().lower()
            or str(explanation.get("strategy", "")).strip().lower()
            or str(track.get("strategy", "")).strip().lower()
            or "unknown"
        )
        confidence = self._safe_float(
            explanation.get(
                "confidence",
                track.get("confidence", trade_context.get("confidence", 0.0)),
            ),
            0.0,
        )

        return {
            "timestamp": utc_now_iso(),
            "position_id": position_id,
            "strategy": strategy,
            "symbol": str(
                trade_context.get(
                    "symbol", explanation.get("symbol", track.get("symbol", ""))
                )
            ).upper(),
            "regime": regime,
            "pnl": float(pnl),
            "loss": bool(float(pnl) < 0.0),
            "verdict": str(explanation.get("verdict", track.get("verdict", "")))
            .strip()
            .lower()
            or "unknown",
            "confidence": confidence,
            "reasoning": str(track.get("reasoning", "")).strip()[:280],
            "earnings_in_window": bool(
                trade_context.get("earnings_in_window")
                or earnings_proximity.get("in_window")
            ),
            "adversarial": bool(trade_context.get("adversarial", False)),
            "attribution": attribution,
        }

    @staticmethod
    def _find_track_row(rows: object, position_id: str) -> dict:
        if not isinstance(rows, list):
            return {}
        for item in reversed(rows):
            if not isinstance(item, dict):
                continue
            if str(item.get("position_id", "")).strip() == position_id:
                return item
        return {}

    def _find_latest_attribution(self, position_id: str) -> dict:
        payload = load_json(self.attribution_path, {"history": []})
        history = payload.get("history", []) if isinstance(payload, dict) else []
        if not isinstance(history, list):
            return {}
        for day in reversed(history):
            attribution = day.get("attribution", {}) if isinstance(day, dict) else {}
            rows = (
                attribution.get("positions", [])
                if isinstance(attribution, dict)
                else []
            )
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("position_id", "")).strip() == position_id:
                    return {
                        "delta_pnl": self._safe_float(row.get("delta_pnl"), 0.0),
                        "gamma_pnl": self._safe_float(row.get("gamma_pnl"), 0.0),
                        "theta_pnl": self._safe_float(row.get("theta_pnl"), 0.0),
                        "vega_pnl": self._safe_float(row.get("vega_pnl"), 0.0),
                        "rho_pnl": self._safe_float(row.get("rho_pnl"), 0.0),
                        "residual": self._safe_float(row.get("residual"), 0.0),
                    }
        return {}

    def _detect_patterns(self, rows: list[dict]) -> list[dict]:
        patterns: list[dict] = []
        min_trades = max(3, int(self.config.min_trades_for_pattern or 5))
        min_loss_rate = max(
            0.4, min(0.99, float(self.config.loss_rate_threshold or 0.50))
        )
        # Threshold for detecting high-win-rate combinations worth promoting
        min_win_rate_for_positive = 0.70

        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            strategy = str(row.get("strategy", "")).strip().lower()
            regime = str(row.get("regime", "")).strip().upper()
            if not strategy or not regime:
                continue
            grouped.setdefault((strategy, regime), []).append(row)
        for (strategy, regime), items in grouped.items():
            stats = self._pattern_stats(items)
            if stats["count"] < min_trades:
                continue
            if stats["loss_rate"] >= min_loss_rate:
                # Failure pattern: avoid this combo
                patterns.append(
                    {
                        "pattern_key": f"strategy_regime:{strategy}|{regime}",
                        "rule": (
                            f"RULE: Do NOT approve {strategy} when regime is {regime} "
                            "unless volatility risk premium and options-flow context are "
                            "strongly supportive — historical loss rate is high."
                        ),
                        "stats": stats,
                    }
                )
            elif (1.0 - stats["loss_rate"]) >= min_win_rate_for_positive and stats["avg_pnl"] > 0:
                # Positive pattern: promote this combo
                patterns.append(
                    {
                        "pattern_key": f"prefer:{strategy}|{regime}",
                        "rule": (
                            f"RULE: PREFER approving {strategy} when regime is {regime} "
                            "— this combination has a historically high win rate and positive EV."
                        ),
                        "stats": stats,
                    }
                )

        low_conf = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("verdict", "")).lower() in {"approve", "reduce_size"}
            and 0.0 < self._safe_float(row.get("confidence"), 0.0) < 65.0
        ]
        stats = self._pattern_stats(low_conf)
        if stats["count"] >= min_trades and stats["loss_rate"] >= min_loss_rate:
            patterns.append(
                {
                    "pattern_key": "low_confidence_approvals",
                    "rule": (
                        "RULE: Avoid approving trades when CIO confidence is below 65 "
                        "unless position size is explicitly reduced and risk is hedged."
                    ),
                    "stats": stats,
                }
            )

        # Detect adversarial/wrong-regime patterns — any strategy with loss rate >= threshold
        adversarial_rows = [
            row for row in rows
            if isinstance(row, dict) and bool(row.get("adversarial"))
        ]
        adv_stats = self._pattern_stats(adversarial_rows)
        if adv_stats["count"] >= min_trades and adv_stats["loss_rate"] >= min_loss_rate:
            patterns.append(
                {
                    "pattern_key": "wrong_regime_strategy",
                    "rule": (
                        "RULE: Do NOT approve a strategy that runs counter to the detected "
                        "market regime. Regime alignment is the single strongest predictor "
                        "of trade outcome — misaligned trades have a historically high loss rate."
                    ),
                    "stats": adv_stats,
                }
            )

        calendar_earnings = [
            row
            for row in rows
            if isinstance(row, dict)
            and "calendar" in str(row.get("strategy", "")).lower()
            and bool(row.get("earnings_in_window"))
        ]
        stats = self._pattern_stats(calendar_earnings)
        if stats["count"] >= min_trades and stats["loss_rate"] >= min_loss_rate:
            patterns.append(
                {
                    "pattern_key": "calendar_spread_earnings_week",
                    "rule": (
                        "RULE: Do NOT approve calendar spreads during earnings week "
                        "unless event-volatility edge is explicitly confirmed."
                    ),
                    "stats": stats,
                }
            )

        return patterns

    @staticmethod
    def _pattern_stats(rows: list[dict]) -> dict:
        count = len(rows)
        if count <= 0:
            return {"count": 0, "loss_rate": 0.0, "avg_pnl": 0.0}
        losses = 0
        total_pnl = 0.0
        for row in rows:
            pnl = RLPromptOptimizer._safe_float(row.get("pnl"), 0.0)
            total_pnl += pnl
            if pnl < 0:
                losses += 1
        return {
            "count": int(count),
            "loss_rate": round(losses / max(1, count), 4),
            "avg_pnl": round(total_pnl / max(1, count), 2),
        }

    def _upsert_rules(self, payload: dict, patterns: list[dict]) -> list[dict]:
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            rules = []
            payload["rules"] = rules
        created: list[dict] = []
        existing_keys = {
            str(item.get("pattern_key", "")) for item in rules if isinstance(item, dict)
        }
        for pattern in patterns:
            if not isinstance(pattern, dict):
                continue
            pattern_key = str(pattern.get("pattern_key", "")).strip()
            rule_text = str(pattern.get("rule", "")).strip()
            if not pattern_key or not rule_text:
                continue
            if pattern_key in existing_keys:
                continue
            rule_row = {
                "rule_id": uuid.uuid4().hex[:12],
                "created_at": utc_now_iso(),
                "pattern_key": pattern_key,
                "rule": rule_text,
                "stats": pattern.get("stats", {}),
            }
            rules.append(rule_row)
            existing_keys.add(pattern_key)
            created.append(rule_row)
        payload["rules"] = rules
        return created

    def _prune_rules(self, payload: dict) -> list[dict]:
        rules = payload.get("rules", [])
        if not isinstance(rules, list):
            payload["rules"] = []
            return []
        max_rules = max(1, int(self.config.max_rules or 25))
        if len(rules) <= max_rules:
            return []
        pruned = rules[:-max_rules]
        payload["rules"] = rules[-max_rules:]
        return [item for item in pruned if isinstance(item, dict)]

    def _load_rules_payload(self) -> dict:
        payload = load_json(
            self.rules_path, {"rules": [], "recent_closed_trades": [], "meta": {}}
        )
        if not isinstance(payload, dict):
            payload = {"rules": [], "recent_closed_trades": [], "meta": {}}
        if not isinstance(payload.get("rules"), list):
            payload["rules"] = []
        if not isinstance(payload.get("recent_closed_trades"), list):
            payload["recent_closed_trades"] = []
        if not isinstance(payload.get("meta"), dict):
            payload["meta"] = {}
        return payload

    def _append_audit(self, event_type: str, details: dict) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "event_type": str(event_type),
            "details": details
            if isinstance(details, dict)
            else {"value": str(details)},
        }
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.audit_path.exists():
            self.audit_path.touch(mode=0o600)
        with open(self.audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), default=str))
            handle.write("\n")

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return float(default)
