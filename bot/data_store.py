"""Lightweight helpers for on-disk bot data caches and analytics artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from bot.file_security import atomic_write_private, tighten_file_permissions, validate_sensitive_file

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("bot/data")


def ensure_data_dir(path: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Create and return the data directory used for cached bot artifacts."""
    data_dir = Path(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_json(path: Path | str, default: Any) -> Any:
    """Read JSON from disk and fall back to ``default`` on missing/invalid payloads."""
    target = Path(path)
    if not target.exists():
        return default

    try:
        validate_sensitive_file(target, label=f"data file {target}", allow_missing=False)
        tighten_file_permissions(target, label=f"data file {target}")
        with open(target, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", target, exc)
        return default


def dump_json(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    """Atomically persist JSON payload to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_private(
        target,
        json.dumps(payload, indent=indent, default=str),
        label=f"data file {target}",
    )
