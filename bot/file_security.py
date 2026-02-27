"""Security helpers for handling sensitive local files."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PRIVATE_FILE_MODE = 0o600
_GROUP_OR_OTHER_BITS = stat.S_IRWXG | stat.S_IRWXO


def validate_sensitive_file(
    path: Path,
    label: str,
    allow_missing: bool = True,
) -> None:
    """Validate that an on-disk sensitive path is a regular file (not symlink)."""
    try:
        if path.is_symlink():
            raise RuntimeError(f"Refusing to use symlink for {label}: {path}")

        if not path.exists():
            if allow_missing:
                return
            raise FileNotFoundError(f"{label} does not exist: {path}")

        if not path.is_file():
            raise RuntimeError(f"{label} path is not a regular file: {path}")
    except PermissionError:
        pass  # File is locked by system, so it exists and is likely valid.


def tighten_file_permissions(path: Path, label: str) -> None:
    """Restrict sensitive file permissions to owner read/write when possible."""
    if os.name != "posix":
        return
    try:
        if not path.exists():
            return
    except PermissionError:
        return  # Locked by system, can't change permissions anyway.

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & _GROUP_OR_OTHER_BITS:
            path.chmod(_PRIVATE_FILE_MODE)
            logger.warning(
                "Tightened permissions for %s at %s to 0o600.",
                label,
                path,
            )
    except OSError as exc:
        logger.warning(
            "Failed to tighten permissions for %s at %s: %s",
            label,
            path,
            exc,
        )


def atomic_write_private(path: Path, content: str, label: str) -> None:
    """Atomically write sensitive text content with private file permissions."""
    validate_sensitive_file(path, label=label, allow_missing=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)

    try:
        if os.name == "posix":
            os.fchmod(tmp_fd, _PRIVATE_FILE_MODE)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
        tmp_fd = -1
        os.replace(tmp_path, path)
        tighten_file_permissions(path, label=label)
    finally:
        if tmp_fd != -1:
            os.close(tmp_fd)
        if tmp_path.exists():
            tmp_path.unlink()
