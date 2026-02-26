"""Dynamic strategy synthesis sandbox for regime-break response."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

from bot.backtester import Backtester
from bot.config import StrategySandboxConfig
from bot.data_store import dump_json, ensure_data_dir, load_json
from bot.number_utils import safe_float
from bot.openai_compat import request_openai_json


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_BASE_STRATEGY_PATH = Path("bot/strategies/base.py")


@dataclass
class SandboxDecision:
    triggered: bool
    deployed: bool
    reason: str = ""
    proposal: Optional[dict] = None
    backtest: Optional[dict] = None


class StrategySandboxManager:
    """Manage trigger -> proposal -> backtest -> temporary deployment lifecycle."""

    def __init__(
        self,
        config: StrategySandboxConfig,
        *,
        state_path: Optional[str | Path] = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        base_strategy_path: str | Path = DEFAULT_BASE_STRATEGY_PATH,
        openai_api_key: Optional[str] = None,
        backtester_factory: Optional[Callable[[], Backtester]] = None,
    ):
        self.config = config
        self.state_path = Path(state_path or config.state_file)
        ensure_data_dir(self.state_path.parent)
        self.config_path = Path(config_path)
        self.base_strategy_path = Path(base_strategy_path)
        self.openai_api_key = str(openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.backtester_factory = backtester_factory or (lambda: Backtester())
        self._state = self._load_state()

    def evaluate_cycle(
        self,
        *,
        regime: str,
        strategy_scores: dict[str, float],
        enabled_strategies: set[str],
        market_context: Optional[dict] = None,
        as_of: Optional[date] = None,
    ) -> SandboxDecision:
        """Run the full sandbox lifecycle for one scan cycle."""
        today = as_of or date.today()
        self.expire_if_needed(today=today)

        if self.active_strategy() is not None:
            return SandboxDecision(triggered=False, deployed=False, reason="sandbox_active")

        triggered = self.update_trigger(
            regime=regime,
            strategy_scores=strategy_scores,
            enabled_strategies=enabled_strategies,
        )
        if not triggered:
            return SandboxDecision(triggered=False, deployed=False, reason="trigger_not_met")

        failing = [
            name
            for name in sorted(enabled_strategies)
            if safe_float(strategy_scores.get(name), 0.0) < float(self.config.min_failing_score)
        ]

        proposal = self.propose_strategy(
            regime=regime,
            failing_strategies=failing,
            market_context=market_context or {},
        )
        if not proposal:
            return SandboxDecision(
                triggered=True,
                deployed=False,
                reason="proposal_unavailable",
            )

        backtest = self.run_sandbox_backtest(proposal, as_of=today)
        if not bool(backtest.get("passed")):
            self._append_history(
                {
                    "timestamp": _utc_now_iso(),
                    "event": "sandbox_backtest_failed",
                    "proposal": proposal,
                    "backtest": backtest,
                }
            )
            self._save_state()
            return SandboxDecision(
                triggered=True,
                deployed=False,
                reason="backtest_failed",
                proposal=proposal,
                backtest=backtest,
            )

        deployed = self.deploy_strategy(proposal, backtest=backtest, deployed_on=today)
        if not deployed:
            return SandboxDecision(
                triggered=True,
                deployed=False,
                reason="deployment_blocked",
                proposal=proposal,
                backtest=backtest,
            )

        return SandboxDecision(
            triggered=True,
            deployed=True,
            reason="deployed",
            proposal=proposal,
            backtest=backtest,
        )

    def update_trigger(
        self,
        *,
        regime: str,
        strategy_scores: dict[str, float],
        enabled_strategies: set[str],
    ) -> bool:
        """Update consecutive-failure trigger counters and return trigger status."""
        if not bool(self.config.enabled):
            return False
        if not enabled_strategies:
            return False

        min_score = float(self.config.min_failing_score)
        all_failing = all(
            safe_float(strategy_scores.get(name), 0.0) < min_score
            for name in enabled_strategies
        )

        streak = int(self._state.get("fail_streak", 0) or 0)
        if all_failing:
            streak += 1
        else:
            streak = 0

        self._state["fail_streak"] = streak
        self._state["last_regime"] = str(regime or "UNKNOWN")
        self._state["last_scores"] = {
            name: round(safe_float(strategy_scores.get(name), 0.0), 4)
            for name in sorted(enabled_strategies)
        }
        self._state["updated_at"] = _utc_now_iso()
        self._save_state()

        return bool(all_failing and streak >= int(self.config.consecutive_fail_cycles))

    def propose_strategy(
        self,
        *,
        regime: str,
        failing_strategies: list[str],
        market_context: dict,
    ) -> Optional[dict]:
        """Request a sandbox proposal from the CIO model and parse normalized JSON."""
        if not self.openai_api_key:
            return None

        base_template = ""
        try:
            base_template = self.base_strategy_path.read_text(encoding="utf-8")[:6000]
        except Exception:
            base_template = ""

        payload = {
            "task": "Propose one temporary options strategy configuration as strict JSON.",
            "constraints": {
                "single_strategy": True,
                "output_only_json": True,
                "max_risk_per_trade_usd": "number",
                "dte_range": "[min,max]",
                "delta_range": "[min,max]",
            },
            "regime": str(regime),
            "failing_strategies": list(failing_strategies),
            "market_context": market_context,
            "base_strategy_template": base_template,
            "schema": {
                "name": "snake_case string",
                "type": "debit_spread|credit_spread|calendar_spread|iron_condor|other",
                "direction": "bullish|bearish|neutral",
                "dte_range": [7, 21],
                "delta_range": [0.30, 0.45],
                "max_risk_per_trade": 500,
            },
        }

        try:
            raw = request_openai_json(
                api_key=self.openai_api_key,
                model="gpt-5.2-pro",
                system_prompt=(
                    "You are the CIO for an options desk. Return ONLY valid JSON and no markdown."
                ),
                user_prompt=json.dumps(payload, separators=(",", ":")),
                timeout_seconds=20,
                temperature=0.0,
                reasoning_effort="medium",
                text_verbosity="low",
                max_output_tokens=450,
                chat_fallback_model="gpt-5.2",
                schema_name="sandbox_strategy_proposal",
                schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "direction": {"type": "string"},
                        "dte_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "delta_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "max_risk_per_trade": {"type": "number"},
                    },
                    "required": [
                        "name",
                        "type",
                        "direction",
                        "dte_range",
                        "delta_range",
                        "max_risk_per_trade",
                    ],
                    "additionalProperties": True,
                },
            )
        except Exception:
            return None

        try:
            parsed = json.loads(raw) if raw else {}
        except Exception:
            parsed = {}
        if not isinstance(parsed, dict):
            return None
        return self.parse_strategy_proposal(parsed)

    @staticmethod
    def parse_strategy_proposal(raw: dict) -> dict:
        """Normalize a model proposal into the required deployment schema."""
        name = str(raw.get("name", "sandbox_strategy")).strip().lower().replace(" ", "_")
        name = "".join(ch for ch in name if ch.isalnum() or ch == "_") or "sandbox_strategy"
        strategy_type = str(raw.get("type", "other")).strip().lower() or "other"
        direction = str(raw.get("direction", "neutral")).strip().lower() or "neutral"

        dte_raw = raw.get("dte_range", [7, 21])
        if not isinstance(dte_raw, list) or len(dte_raw) < 2:
            dte_raw = [7, 21]
        dte_a = max(1, int(safe_float(dte_raw[0], 7)))
        dte_b = max(1, int(safe_float(dte_raw[1], 21)))
        dte_range = [min(dte_a, dte_b), max(dte_a, dte_b)]

        delta_raw = raw.get("delta_range", [0.30, 0.45])
        if not isinstance(delta_raw, list) or len(delta_raw) < 2:
            delta_raw = [0.30, 0.45]
        delta_a = max(0.01, min(0.99, safe_float(delta_raw[0], 0.30)))
        delta_b = max(0.01, min(0.99, safe_float(delta_raw[1], 0.45)))
        delta_range = [round(min(delta_a, delta_b), 4), round(max(delta_a, delta_b), 4)]

        max_risk = max(50.0, safe_float(raw.get("max_risk_per_trade"), 500.0))

        return {
            "name": name,
            "type": strategy_type,
            "direction": direction,
            "dte_range": dte_range,
            "delta_range": delta_range,
            "max_risk_per_trade": round(max_risk, 2),
        }

    def run_sandbox_backtest(self, proposal: dict, *, as_of: Optional[date] = None) -> dict:
        """Backtest the proposed strategy in isolation over recent history."""
        end = as_of or date.today()
        start = end - timedelta(days=max(5, int(self.config.backtest_days)))

        try:
            backtester = self.backtester_factory()
            result = backtester.run(start=start.isoformat(), end=end.isoformat())
            report = result.report if hasattr(result, "report") else {}
        except Exception as exc:
            return {
                "passed": False,
                "error": str(exc),
                "sharpe": 0.0,
                "max_drawdown_pct": 100.0,
            }

        sharpe = safe_float((report or {}).get("sharpe_ratio"), 0.0)
        raw_drawdown = safe_float((report or {}).get("max_drawdown"), 0.0)
        drawdown_pct = raw_drawdown * 100.0 if raw_drawdown <= 1.0 else raw_drawdown

        passed = bool(
            sharpe > float(self.config.min_sharpe)
            and drawdown_pct < float(self.config.max_drawdown_pct)
        )
        return {
            "passed": passed,
            "sharpe": round(sharpe, 4),
            "max_drawdown_pct": round(drawdown_pct, 4),
            "report": report if isinstance(report, dict) else {},
            "window": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        }

    def deploy_strategy(self, proposal: dict, *, backtest: dict, deployed_on: Optional[date] = None) -> bool:
        """Deploy a sandbox strategy for a limited window."""
        if self.active_strategy() is not None:
            return False

        day0 = deployed_on or date.today()
        expires_on = day0 + timedelta(days=max(1, int(self.config.deployment_days)))

        sandbox_entry = {
            **proposal,
            "enabled": True,
            "sizing_scalar": round(float(self.config.sizing_scalar), 4),
            "deployed_on": day0.isoformat(),
            "expires_on": expires_on.isoformat(),
        }

        self._write_sandbox_config(enabled=True, entry=sandbox_entry)

        self._state["active"] = {
            "proposal": dict(proposal),
            "deployed_on": day0.isoformat(),
            "expires_on": expires_on.isoformat(),
            "sizing_scalar": round(float(self.config.sizing_scalar), 4),
            "backtest": backtest if isinstance(backtest, dict) else {},
        }
        self._state["fail_streak"] = 0
        self._append_history(
            {
                "timestamp": _utc_now_iso(),
                "event": "sandbox_deployed",
                "proposal": proposal,
                "backtest": backtest,
                "expires_on": expires_on.isoformat(),
            }
        )
        self._save_state()
        return True

    def expire_if_needed(self, *, today: Optional[date] = None) -> Optional[dict]:
        """Auto-disable the active sandbox strategy when deployment window ends."""
        active = self.active_strategy()
        if not isinstance(active, dict):
            return None

        cur_day = today or date.today()
        raw_expires = str(active.get("expires_on", "")).strip()
        try:
            expires_on = date.fromisoformat(raw_expires)
        except ValueError:
            expires_on = cur_day

        if cur_day <= expires_on:
            return None

        self._write_sandbox_config(enabled=False, entry=None)
        record = {
            "timestamp": _utc_now_iso(),
            "event": "sandbox_expired",
            "active": active,
            "expired_on": cur_day.isoformat(),
        }
        self._append_history(record)
        self._state["active"] = None
        self._save_state()
        return record

    def active_strategy(self) -> Optional[dict]:
        active = self._state.get("active") if isinstance(self._state, dict) else None
        return active if isinstance(active, dict) else None

    def _load_state(self) -> dict:
        payload = load_json(self.state_path, {"fail_streak": 0, "active": None, "history": []})
        if not isinstance(payload, dict):
            payload = {"fail_streak": 0, "active": None, "history": []}
        if not isinstance(payload.get("history"), list):
            payload["history"] = []
        return payload

    def _save_state(self) -> None:
        self._state["updated_at"] = _utc_now_iso()
        dump_json(self.state_path, self._state)

    def _append_history(self, row: dict) -> None:
        history = self._state.get("history")
        if not isinstance(history, list):
            history = []
            self._state["history"] = history
        history.append(row)
        self._state["history"] = history[-200:]

    def _write_sandbox_config(self, *, enabled: bool, entry: Optional[dict]) -> None:
        raw: dict = {}
        if self.config_path.exists():
            try:
                loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
                raw = loaded if isinstance(loaded, dict) else {}
            except Exception:
                raw = {}
        strategies = raw.get("strategies")
        if not isinstance(strategies, dict):
            strategies = {}
            raw["strategies"] = strategies

        if enabled and isinstance(entry, dict):
            strategies["sandbox"] = dict(entry)
        else:
            existing = strategies.get("sandbox")
            if isinstance(existing, dict):
                existing["enabled"] = False
                strategies["sandbox"] = existing
            else:
                strategies["sandbox"] = {"enabled": False}

        text = yaml.safe_dump(raw, sort_keys=False)
        self.config_path.write_text(text, encoding="utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
