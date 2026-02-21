"""Shared numeric parsing helpers for external API payloads."""


def safe_float(value: object, default: float = 0.0) -> float:
    """Safely parse floats from payloads that may contain nulls/strings."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    """Safely parse integers from payloads that may contain nulls/strings."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
