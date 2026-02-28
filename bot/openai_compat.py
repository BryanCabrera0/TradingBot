"""Compatibility helpers for OpenAI JSON responses with endpoint fallback."""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
RESPONSES_ONLY_MODELS = {"gpt-5.2-pro"}
DEFAULT_CHAT_FALLBACK_MODEL = "gpt-4.1"


def request_openai_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    temperature: float = 0.0,
    schema_name: str = "response",
    schema: Optional[dict] = None,
    reasoning_effort: Optional[str] = None,
    text_verbosity: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    chat_fallback_model: Optional[str] = None,
) -> str:
    """Request JSON output from OpenAI with Responses->ChatCompletions fallback.

    Fallback is only triggered when Responses API returns 400/404.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    responses_payload: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
    }
    effort = _normalize_reasoning_effort(reasoning_effort)
    verbosity = _normalize_text_verbosity(text_verbosity)
    supports_structured = model_supports_structured_outputs(model)

    if effort:
        responses_payload["reasoning"] = {"effort": effort}
    if _should_send_temperature(model, effort):
        responses_payload["temperature"] = float(temperature)
    if max_output_tokens is not None:
        responses_payload["max_output_tokens"] = max(64, int(max_output_tokens))

    text_payload: dict[str, Any] = {}
    if verbosity:
        text_payload["verbosity"] = verbosity
    if supports_structured:
        if schema is not None:
            text_payload["format"] = {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        else:
            text_payload["format"] = {"type": "json_object"}
    if text_payload:
        responses_payload["text"] = text_payload

    response = requests.post(
        RESPONSES_URL,
        headers=headers,
        json=responses_payload,
        timeout=max(1, int(timeout_seconds)),
    )
    if response.status_code in {400, 404}:
        logger.warning(
            "OpenAI Responses API unavailable (%s). Falling back to Chat Completions.",
            response.status_code,
        )
        fallback_model = str(
            chat_fallback_model or ""
        ).strip() or fallback_model_for_chat(model)
        return _request_chat_completions_json(
            api_key=api_key,
            model=fallback_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=effort,
        )

    response.raise_for_status()
    data = response.json()
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    extracted = extract_responses_output_text(data)
    if extracted:
        return extracted

    raise RuntimeError("OpenAI returned no text output")


def extract_responses_output_text(data: dict) -> str:
    """Best-effort extraction of text from Responses API payloads."""
    output = data.get("output", [])
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def _request_chat_completions_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_seconds: int,
    temperature: float,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    effort = _normalize_reasoning_effort(reasoning_effort)
    if _should_send_temperature(model, effort):
        payload["temperature"] = float(temperature)
    if max_output_tokens is not None:
        payload["max_completion_tokens"] = max(64, int(max_output_tokens))
    if effort:
        payload["reasoning"] = {"effort": effort}

    response = requests.post(
        CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=max(1, int(timeout_seconds)),
    )
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenAI chat completions returned no choices")

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    raise RuntimeError("OpenAI chat completions returned no message content")


def fallback_model_for_chat(model: str) -> str:
    """Return a robust chat-completions fallback model for the given primary model."""
    model_name = str(model or "").strip().lower()
    if model_name in RESPONSES_ONLY_MODELS:
        return DEFAULT_CHAT_FALLBACK_MODEL
    return str(model or "").strip() or DEFAULT_CHAT_FALLBACK_MODEL


def model_supports_structured_outputs(model: str) -> bool:
    """Return whether strict structured outputs should be requested for this model."""
    model_name = str(model or "").strip().lower()
    if model_name in RESPONSES_ONLY_MODELS:
        return False
    return True


def _should_send_temperature(model: str, reasoning_effort: Optional[str]) -> bool:
    """Determine whether the temperature parameter should be sent."""
    model_name = str(model or "").strip().lower()
    if (
        model_name.startswith("gpt-5")
        and reasoning_effort
        and reasoning_effort != "none"
    ):
        return False
    return True


def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    effort = str(value).strip().lower()
    if effort in {"none", "low", "medium", "high", "xhigh"}:
        return effort
    return None


def _normalize_text_verbosity(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    verbosity = str(value).strip().lower()
    if verbosity in {"low", "medium", "high"}:
        return verbosity
    return None
