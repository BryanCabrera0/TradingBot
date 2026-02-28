"""Google Gemini model alias helpers for resilient API fallbacks."""

from __future__ import annotations

import threading

GOOGLE_GEMINI_PRO_MODEL = "gemini-3.1-pro-thinking-preview"
GOOGLE_GEMINI_FLASH_MODEL = "gemini-3.1-flash-thinking-preview"

_MODEL_FALLBACKS: dict[str, list[str]] = {
    GOOGLE_GEMINI_PRO_MODEL: [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-pro-latest",
        "gemini-2.5-pro",
    ],
    GOOGLE_GEMINI_FLASH_MODEL: [
        "gemini-3.1-flash-preview",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ],
    "gemini-3.1-pro-preview": [
        "gemini-3-pro-preview",
        "gemini-pro-latest",
        "gemini-2.5-pro",
    ],
    "gemini-3.1-flash-preview": [
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ],
    "gemini-3-pro-preview": ["gemini-pro-latest", "gemini-2.5-pro"],
    "gemini-3-flash-preview": ["gemini-flash-latest", "gemini-2.5-flash"],
}
_RESOLVED_MODEL_CACHE: dict[str, str] = {}
_RESOLVED_MODEL_CACHE_LOCK = threading.Lock()


def candidate_google_models(requested_model: str) -> list[str]:
    """Return ordered model candidates for resilient Gemini API calls."""
    primary = str(requested_model or "").strip() or GOOGLE_GEMINI_PRO_MODEL
    candidates: list[str] = []
    seen: set[str] = set()

    with _RESOLVED_MODEL_CACHE_LOCK:
        cached = _RESOLVED_MODEL_CACHE.get(primary)
    if cached:
        candidates.append(cached)
        seen.add(cached)

    candidates.append(primary)
    seen.add(primary)
    for model_name in _MODEL_FALLBACKS.get(primary, []):
        if model_name in seen:
            continue
        seen.add(model_name)
        candidates.append(model_name)
    return candidates


def remember_google_model_success(requested_model: str, resolved_model: str) -> None:
    """Cache a successful runtime model resolution."""
    primary = str(requested_model or "").strip()
    resolved = str(resolved_model or "").strip()
    if not primary or not resolved:
        return
    with _RESOLVED_MODEL_CACHE_LOCK:
        _RESOLVED_MODEL_CACHE[primary] = resolved


def reset_google_model_cache() -> None:
    """Clear in-memory model-resolution cache (mainly for tests)."""
    with _RESOLVED_MODEL_CACHE_LOCK:
        _RESOLVED_MODEL_CACHE.clear()
