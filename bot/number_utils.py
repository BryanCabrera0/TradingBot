"""Shared numeric parsing helpers for external API payloads."""

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely parse floats from payloads that may contain nulls/strings."""
    try:
        return float(value)  # type: ignore
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely parse integers from payloads that may contain nulls/strings."""
    try:
        return int(float(value))  # type: ignore
    except (TypeError, ValueError):
        return default
