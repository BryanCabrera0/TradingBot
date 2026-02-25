"""Compatibility helpers for OpenAI JSON responses with endpoint fallback."""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

RESPONSES_URL = "https://api.openai.com/v1/responses"
CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


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
) -> str:
    """Request JSON output from OpenAI with Responses->ChatCompletions fallback.

    Fallback is only triggered when Responses API returns 400/404.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    responses_payload = {
        "model": model,
        "temperature": float(temperature),
        "instructions": system_prompt,
        "input": user_prompt,
    }
    if schema is not None:
        responses_payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        }
    else:
        responses_payload["text"] = {"format": {"type": "json_object"}}

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
        return _request_chat_completions_json(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
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
) -> str:
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
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
