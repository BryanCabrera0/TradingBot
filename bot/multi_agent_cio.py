"""Multi-agent CIO debate orchestration for LLM trade decisions.

SIMPLE EXPLANATION:
The Multi-Agent CIO (Chief Investment Officer) is the final decision-maker. 
It acts like a committee of different expert personas (like a Macro Economist, a Volatility 
Analyst, and a Risk Manager) who "debate" the current market conditions. 
After all personas weigh in, the CIO synthesizes their views and makes a 
final call on whether it's safe to trade overall.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DebateAgent:
    key: str
    persona: str
    focus: str
    system_prompt: str


@dataclass
class DebateResult:
    final_payload: dict
    model_votes: list[dict]
    transcript: dict


class MultiAgentCIO:
    """Run analyst theses + CIO synthesis using one configured provider path."""

    ANALYSTS: tuple[DebateAgent, ...] = (
        DebateAgent(
            key="macro_economist",
            persona="Macro Economist",
            focus="Regime, Federal Reserve policy path, rates/yield-curve dynamics, and cross-asset macro stress.",
            system_prompt=(
                "You are the Macro Economist agent for an options trading desk. "
                "Produce independent macro-regime analysis only. "
                "Respond ONLY with strict JSON."
            ),
        ),
        DebateAgent(
            key="quant_vol_trader",
            persona="Quant Volatility Trader",
            focus="Term structure, skew, vol-of-vol, and implied-vs-realized volatility dislocations.",
            system_prompt=(
                "You are the Quant Volatility Trader agent for an options trading desk. "
                "Focus on volatility microstructure and pricing edge only. "
                "Respond ONLY with strict JSON."
            ),
        ),
        DebateAgent(
            key="risk_manager",
            persona="Risk Manager",
            focus="Portfolio margin load, concentration/correlation, and tail-risk under stressed scenarios.",
            system_prompt=(
                "You are the Risk Manager agent for an options trading desk. "
                "Prioritize downside control, concentration, and survivability. "
                "Respond ONLY with strict JSON."
            ),
        ),
    )

    CIO: DebateAgent = DebateAgent(
        key="cio",
        persona="Aggressive Chief Investment Officer",
        focus="Growth, capital deployment, and executing trades aggressively.",
        system_prompt=(
            "You must arbitrate analyst disagreements and decide final go/no-go and size. "
            "While maintaining a smart and generally conservative profile, be slightly more tolerant of risk to approve more viable setups and avoid missing opportunities. "
            "Respond ONLY with strict JSON."
        ),
    )

    def __init__(
        self,
        *,
        query_model: Callable[..., str],
        parse_decision: Callable[[str], tuple[object, bool]],
        primary_model: str = "gpt-4o",
        fallback_model: str = "gpt-4o-mini",
        provider: str = "openai",
        learned_rules: Optional[list[str]] = None,
    ):
        self._query_model = query_model
        self._parse_decision = parse_decision
        self.primary_model = (
            str(primary_model or "gpt-4o").strip() or "gpt-4o"
        )
        self.fallback_model = str(fallback_model or "gpt-4o-mini").strip() or "gpt-4o-mini"
        self.provider = str(provider or "openai").strip().lower() or "openai"
        self.learned_rules = [
            str(rule).strip() for rule in (learned_rules or []) if str(rule).strip()
        ][:25]

    def run(self, prompt: str) -> DebateResult:
        """Execute one full analyst->CIO debate cycle."""
        trade_packet = self._safe_json_load(prompt) or {"raw_prompt": prompt}
        learned_rules = self._extract_learned_rules(trade_packet)

        first_round: list[dict] = []
        votes: list[dict] = []
        for agent in self.ANALYSTS:
            thesis = self._run_analyst_round(
                agent=agent,
                trade_packet=trade_packet,
                round_number=1,
                prior_round=None,
            )
            first_round.append(thesis)
            votes.append(self._to_model_vote(thesis, round_number=1))

        contradictions = self._detect_contradictions(first_round)
        cio_initial = self._run_cio_round(
            trade_packet=trade_packet,
            analyst_theses=first_round,
            contradictions=contradictions,
            round_number=1,
            rebuttals=None,
            learned_rules=learned_rules,
        )

        debate_needed = bool(contradictions) or bool(cio_initial.get("force_debate"))
        second_round: list[dict] = []
        if debate_needed:
            for agent in self.ANALYSTS:
                rebuttal = self._run_analyst_round(
                    agent=agent,
                    trade_packet=trade_packet,
                    round_number=2,
                    prior_round={
                        "first_round_theses": first_round,
                        "cio_initial": cio_initial,
                    },
                )
                second_round.append(rebuttal)
                votes.append(self._to_model_vote(rebuttal, round_number=2))

            cio_final = self._run_cio_round(
                trade_packet=trade_packet,
                analyst_theses=first_round,
                contradictions=contradictions,
                round_number=2,
                rebuttals=second_round,
                learned_rules=learned_rules,
            )
        else:
            cio_final = cio_initial

        votes.append(
            self._to_model_vote(cio_final, round_number=2 if debate_needed else 1)
        )
        final_payload = self._finalize_cio_payload(
            cio_final, debate_needed=debate_needed
        )

        transcript = {
            "provider": self.provider,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "debate_rounds": 2 if debate_needed else 1,
            "contradictions": contradictions,
            "first_round": first_round,
            "cio_initial": cio_initial,
            "second_round": second_round,
            "cio_final": cio_final,
        }

        return DebateResult(
            final_payload=final_payload,
            model_votes=votes,
            transcript=transcript,
        )

    def _run_analyst_round(
        self,
        *,
        agent: DebateAgent,
        trade_packet: dict,
        round_number: int,
        prior_round: Optional[dict],
    ) -> dict:
        prompt_payload = {
            "task": (
                "Generate your independent thesis for this options signal."
                if round_number == 1
                else "Rebut contradictions and update your thesis with concise, evidence-based changes."
            ),
            "agent": {
                "key": agent.key,
                "persona": agent.persona,
                "focus": agent.focus,
            },
            "trade_packet": trade_packet,
            "prior_round": prior_round or {},
            "output_schema": {
                "verdict": "approve|reject|reduce_size",
                "confidence": "0-100",
                "reasoning": "string",
                "suggested_adjustment": "string|null",
                "risk_adjustment": "0.10-1.00",
                "capital_allocation_scalar": "0.10-1.00",
                "key_risks": ["string"],
                "disagreements": ["string"],
                "bull_case": "string|null",
                "bear_case": "string|null",
                "key_risk": "string|null",
                "expected_duration": "string|null",
                "confidence_drivers": ["string"],
            },
        }
        raw, parsed, model_used, used_fallback = self._query_with_retry_and_fallback(
            prompt=json.dumps(prompt_payload, separators=(",", ":")),
            system_prompt=agent.system_prompt,
        )
        return self._normalize_agent_payload(
            agent=agent,
            parsed=parsed,
            raw=raw,
            model=model_used,
            used_fallback=used_fallback,
            round_number=round_number,
        )

    def _run_cio_round(
        self,
        *,
        trade_packet: dict,
        analyst_theses: list[dict],
        contradictions: list[str],
        round_number: int,
        rebuttals: Optional[list[dict]],
        learned_rules: Optional[list[str]] = None,
    ) -> dict:
        prompt_payload = {
            "task": (
                "Review analyst theses, detect contradictions, and decide if debate is required."
                if round_number == 1
                else "Issue FINAL CIO decision after rebuttals: go/no-go and capital allocation."
            ),
            "cio_focus": self.CIO.focus,
            "trade_packet": trade_packet,
            "analyst_theses": analyst_theses,
            "detected_contradictions": contradictions,
            "rebuttals": rebuttals or [],
            "output_schema": {
                "verdict": "approve|reject|reduce_size",
                "confidence": "0-100",
                "reasoning": "string",
                "suggested_adjustment": "string|null",
                "risk_adjustment": "0.10-1.00",
                "capital_allocation_scalar": "0.10-1.00",
                "force_debate": "boolean",
                "contradiction_summary": ["string"],
                "bull_case": "string|null",
                "bear_case": "string|null",
                "key_risk": "string|null",
                "expected_duration": "string|null",
                "confidence_drivers": ["string"],
            },
        }
        raw, parsed, model_used, used_fallback = self._query_with_retry_and_fallback(
            prompt=json.dumps(prompt_payload, separators=(",", ":")),
            system_prompt=self._compose_cio_system_prompt(learned_rules or []),
        )
        normalized = self._normalize_agent_payload(
            agent=self.CIO,
            parsed=parsed,
            raw=raw,
            model=model_used,
            used_fallback=used_fallback,
            round_number=round_number,
        )
        normalized["force_debate"] = bool(parsed.get("force_debate"))
        contradiction_summary = parsed.get("contradiction_summary")
        if isinstance(contradiction_summary, list):
            normalized["contradiction_summary"] = [
                str(item).strip()[:180]
                for item in contradiction_summary
                if str(item).strip()
            ][:6]
        else:
            normalized["contradiction_summary"] = []
        return normalized

    def _extract_learned_rules(self, trade_packet: dict) -> list[str]:
        rules: list[str] = []
        packet_rules = trade_packet.get("learned_rules", [])
        if isinstance(packet_rules, list):
            for item in packet_rules:
                text = str(item).strip()
                if text:
                    rules.append(text)
        if not rules:
            rules = list(self.learned_rules)
        return rules[:25]

    def _compose_cio_system_prompt(self, rules: list[str]) -> str:
        base = self.CIO.system_prompt
        clean_rules = [str(rule).strip() for rule in rules if str(rule).strip()]
        if not clean_rules:
            return base
        rendered = "\n".join(f"- {rule}" for rule in clean_rules[:25])
        return (
            f"{base} "
            "Hard constraints from post-trade reinforcement learning:\n"
            f"{rendered}\n"
            "Treat these as mandatory risk rules unless explicit context proves an exception."
        )

    def _query_with_retry_and_fallback(
        self,
        *,
        prompt: str,
        system_prompt: str,
    ) -> tuple[str, dict, str, bool]:
        strict_prompt = f"{prompt}\n\nRespond with strict JSON only."
        attempts = (
            (self.primary_model, False),
            (self.primary_model, False),
            (self.fallback_model, True),
            (self.fallback_model, True),
        )
        last_error = ""
        for model_name, used_fallback in attempts:
            try:
                raw = self._query_model(
                    strict_prompt,
                    provider=self.provider,
                    model=model_name,
                    system_prompt=system_prompt,
                )
                parsed = self._safe_json_load(raw)
                decision, valid = self._parse_decision(raw)
                if valid and parsed:
                    normalized = {
                        **parsed,
                        "verdict": str(getattr(decision, "verdict", "reject"))
                        .strip()
                        .lower()
                        or "reject",
                        "confidence": float(
                            getattr(decision, "confidence_pct", 0.0) or 0.0
                        ),
                        "reasoning": str(
                            getattr(
                                decision, "reasoning", "LLM response missing reason"
                            )
                        )[:280],
                        "suggested_adjustment": getattr(
                            decision, "suggested_adjustment", None
                        ),
                        "risk_adjustment": float(
                            getattr(decision, "risk_adjustment", 1.0) or 1.0
                        ),
                    }
                    return raw, normalized, model_name, used_fallback
                raise RuntimeError("invalid or non-JSON decision payload")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue

        fallback = {
            "verdict": "approve",
            "confidence": 100.0,
            "reasoning": (
                "CIO debate pipeline failed to produce valid JSON after retries; defaulting approve."
                + (f" Last error: {last_error}" if last_error else "")
            )[:280],
            "suggested_adjustment": None,
            "risk_adjustment": 1.0,
            "capital_allocation_scalar": 1.0,
        }
        return (
            json.dumps(fallback, separators=(",", ":")),
            fallback,
            self.fallback_model,
            True,
        )

    def _normalize_agent_payload(
        self,
        *,
        agent: DebateAgent,
        parsed: dict,
        raw: str,
        model: str,
        used_fallback: bool,
        round_number: int,
    ) -> dict:
        verdict = str(parsed.get("verdict", "approve")).strip().lower()
        if verdict not in {"approve", "reject", "reduce_size"}:
            verdict = "approve"

        confidence = self._clamp_float(parsed.get("confidence", 0.0), 0.0, 100.0)
        if confidence <= 1.0:
            confidence *= 100.0

        risk_adjustment = self._clamp_float(
            parsed.get("risk_adjustment", 1.0), 0.1, 1.0
        )
        allocation_scalar = self._clamp_float(
            parsed.get("capital_allocation_scalar", risk_adjustment),
            0.1,
            1.0,
        )
        risk_adjustment = min(risk_adjustment, allocation_scalar)

        explanation_drivers = parsed.get("confidence_drivers", [])
        if isinstance(explanation_drivers, list):
            explanation_drivers = [
                str(item).strip()[:120]
                for item in explanation_drivers
                if str(item).strip()
            ][:3]
        else:
            explanation_drivers = []

        disagreements = parsed.get("disagreements", [])
        if isinstance(disagreements, list):
            disagreements = [
                str(item).strip()[:180] for item in disagreements if str(item).strip()
            ][:6]
        else:
            disagreements = []

        key_risks = parsed.get("key_risks", [])
        if isinstance(key_risks, list):
            key_risks = [
                str(item).strip()[:180] for item in key_risks if str(item).strip()
            ][:6]
        else:
            key_risks = []

        return {
            "agent_key": agent.key,
            "persona": agent.persona,
            "focus": agent.focus,
            "provider": self.provider,
            "model": model,
            "used_fallback_model": bool(used_fallback),
            "round": int(round_number),
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": str(
                parsed.get("reasoning", "LLM response missing reason")
            ).strip()[:280],
            "suggested_adjustment": parsed.get("suggested_adjustment"),
            "risk_adjustment": risk_adjustment,
            "capital_allocation_scalar": allocation_scalar,
            "disagreements": disagreements,
            "key_risks": key_risks,
            "bull_case": str(parsed.get("bull_case") or "").strip()[:280],
            "bear_case": str(parsed.get("bear_case") or "").strip()[:280],
            "key_risk": str(parsed.get("key_risk") or "").strip()[:180],
            "expected_duration": str(parsed.get("expected_duration") or "").strip()[
                :120
            ],
            "confidence_drivers": explanation_drivers,
            "raw": raw,
        }

    @staticmethod
    def _to_model_vote(payload: dict, *, round_number: int) -> dict:
        persona = str(payload.get("agent_key", "unknown")).strip().lower()
        provider = str(payload.get("provider", "openai")).strip().lower() or "openai"
        model = str(payload.get("model", "gpt-4o")).strip() or "gpt-4o"
        return {
            "model_id": f"{provider}:{model}:{persona}",
            "provider": provider,
            "model": model,
            "persona": persona,
            "round": int(round_number),
            "used_fallback_model": bool(payload.get("used_fallback_model")),
            "verdict": str(payload.get("verdict", "approve")).lower(),
            "confidence": MultiAgentCIO._clamp_float(
                payload.get("confidence", 0.0), 0.0, 100.0
            ),
            "risk_adjustment": MultiAgentCIO._clamp_float(
                payload.get("risk_adjustment", 1.0), 0.1, 1.0
            ),
            "reasoning": str(payload.get("reasoning", ""))[:280],
            "weight": 1.0,
        }

    @staticmethod
    def _detect_contradictions(theses: list[dict]) -> list[str]:
        verdicts = {
            str(row.get("verdict", "")).lower()
            for row in theses
            if isinstance(row, dict)
        }
        contradictions: list[str] = []
        if "approve" in verdicts and "reject" in verdicts:
            contradictions.append("Analysts disagree on go/no-go (approve vs reject).")
        if len(verdicts) > 1 and "reduce_size" in verdicts:
            contradictions.append(
                "At least one analyst recommends sizing down while others differ."
            )

        confidences = [
            MultiAgentCIO._clamp_float(row.get("confidence", 0.0), 0.0, 100.0)
            for row in theses
            if isinstance(row, dict)
        ]
        if confidences and (max(confidences) - min(confidences) >= 25.0):
            contradictions.append(
                "Large confidence dispersion across analysts (>=25 points)."
            )

        return contradictions[:6]

    def _finalize_cio_payload(self, cio_payload: dict, *, debate_needed: bool) -> dict:
        verdict = str(cio_payload.get("verdict", "")).strip().lower()
        if verdict not in {"approve", "reject", "reduce_size"}:
             verdict = "approve"

        confidence = self._clamp_float(cio_payload.get("confidence", 0.0), 0.0, 100.0)
        if confidence <= 1.0:
            confidence *= 100.0

        allocation = self._clamp_float(
            cio_payload.get("capital_allocation_scalar", 1.0), 0.1, 1.0
        )
        risk_adjustment = self._clamp_float(
            cio_payload.get("risk_adjustment", allocation), 0.1, 1.0
        )
        risk_adjustment = min(risk_adjustment, allocation)

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": str(
                cio_payload.get("reasoning", "CIO decision unavailable")
            ).strip()[:280],
            "suggested_adjustment": cio_payload.get("suggested_adjustment"),
            "risk_adjustment": risk_adjustment,
            "capital_allocation_scalar": allocation,
            "force_debate": bool(cio_payload.get("force_debate")),
            "debate_used": bool(debate_needed),
            "bull_case": str(cio_payload.get("bull_case") or "").strip()[:280],
            "bear_case": str(cio_payload.get("bear_case") or "").strip()[:280],
            "key_risk": str(cio_payload.get("key_risk") or "").strip()[:180],
            "expected_duration": str(
                cio_payload.get("expected_duration") or ""
            ).strip()[:120],
            "confidence_drivers": cio_payload.get("confidence_drivers", []),
            "contradiction_summary": cio_payload.get("contradiction_summary", []),
        }

    @staticmethod
    def _safe_json_load(text: str) -> dict:
        if not text:
            return {}
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
        try:
            parsed = float(str(value))
        except (TypeError, ValueError):
            parsed = minimum
        return max(minimum, min(maximum, parsed))
