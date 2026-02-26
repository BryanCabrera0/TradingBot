"""Cycle-level LLM portfolio strategist."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

from bot.config import LLMStrategistConfig
from bot.openai_compat import request_openai_json

logger = logging.getLogger(__name__)
SECRET_PLACEHOLDER_MARKERS = ("your_", "_here", "changeme")


@dataclass
class PortfolioDirective:
    action: str
    reason: str
    payload: dict
    confidence: float = 0.0


class LLMStrategist:
    """High-level portfolio directive generator."""

    def __init__(self, config: LLMStrategistConfig):
        self.config = config

    def review_portfolio(self, context: dict) -> list[PortfolioDirective]:
        prompt = json.dumps(
            {
                "task": "Return high-level portfolio directives only.",
                "directive_categories": {
                    "reduce_delta": "When net delta > 30, suggest which positions to close.",
                    "skip_sector": "When one sector is >35% of deployed risk, name the sector to skip.",
                    "close_long_dte": "When volatility/regime shifts bearish, close high-DTE positions.",
                    "scale_size": "When risk rises or losses cluster, reduce new-trade size.",
                    "add_hedge": "When portfolio is large and tail-risk is cheap, suggest protective hedge.",
                },
                "examples": [
                    {
                        "action": "reduce_delta",
                        "confidence": 78,
                        "reason": "Net delta too positive vs regime risk.",
                        "payload": {"count": 2, "direction": "positive"},
                    },
                    {
                        "action": "skip_sector",
                        "confidence": 82,
                        "reason": "Sector concentration too high.",
                        "payload": {"sector": "Information Technology"},
                    },
                    {
                        "action": "close_long_dte",
                        "confidence": 74,
                        "reason": "Regime deteriorating with rising volatility.",
                        "payload": {"max_dte": 30},
                    },
                    {
                        "action": "scale_size",
                        "confidence": 76,
                        "reason": "Consecutive losses and unstable vol require risk reduction.",
                        "payload": {"factor": 0.8},
                    },
                    {
                        "action": "add_hedge",
                        "confidence": 68,
                        "reason": "Tail-risk hedge is cheap and portfolio size is elevated.",
                        "payload": {"symbol": "SPY", "type": "buy_put"},
                    },
                ],
                "schema": {
                    "directives": [
                        {
                            "action": "reduce_delta|skip_sector|close_long_dte|scale_size|none",
                            "confidence": "0-100",
                            "reason": "string",
                            "payload": "object",
                        }
                    ]
                },
                "context": context,
            },
            separators=(",", ":"),
        )
        raw = self._query(prompt)
        data = _safe_json(raw)
        directives = data.get("directives", []) if isinstance(data, dict) else []
        if not isinstance(directives, list):
            return []

        out: list[PortfolioDirective] = []
        for item in directives[: max(1, int(self.config.max_directives))]:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "none")).strip().lower()
            if action in {"", "none"}:
                continue
            confidence = _clamp_float(item.get("confidence"), 0.0, 100.0)
            out.append(
                PortfolioDirective(
                    action=action,
                    reason=str(item.get("reason", "") or "LLM strategist directive"),
                    payload=item.get("payload", {}) if isinstance(item.get("payload"), dict) else {},
                    confidence=confidence / 100.0 if confidence > 1 else confidence,
                )
            )
        return out

    def _query(self, prompt: str) -> str:
        provider = str(self.config.provider).strip().lower()
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not _is_configured_secret(api_key):
                raise RuntimeError("OPENAI_API_KEY missing for llm_strategist")
            return request_openai_json(
                api_key=api_key,
                model=self.config.model,
                system_prompt="You are a portfolio strategist. Respond ONLY with valid JSON.",
                user_prompt=prompt,
                timeout_seconds=self.config.timeout_seconds,
                temperature=0.2,
                schema_name="portfolio_directives",
                schema={
                    "type": "object",
                    "properties": {
                        "directives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "reason": {"type": "string"},
                                    "payload": {"type": "object"},
                                },
                                "required": ["action", "confidence", "reason", "payload"],
                            },
                        }
                    },
                    "required": ["directives"],
                },
            )
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not _is_configured_secret(api_key):
                raise RuntimeError("ANTHROPIC_API_KEY missing for llm_strategist")
            try:
                from anthropic import Anthropic
            except Exception as exc:
                raise RuntimeError("anthropic package required") from exc
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.config.model or "claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.2,
                system="Return JSON only.",
                messages=[{"role": "user", "content": prompt}],
            )
            parts = []
            for item in getattr(response, "content", []):
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts).strip()
        if provider == "ollama":
            url = "http://127.0.0.1:11434/api/generate"
            body = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            }
            response = requests.post(url, json=body, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
        raise RuntimeError(f"Unsupported llm_strategist provider: {self.config.provider}")


def _safe_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _is_configured_secret(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in SECRET_PLACEHOLDER_MARKERS)


def _clamp_float(value: object, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(maximum, parsed))
