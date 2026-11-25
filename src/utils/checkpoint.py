"""Atomic file writing and persistent Google Drive checkpoint helpers."""

from __future__ import annotations

import os
from pathlib import Path


def cleanup_stale_temps(target: Path) -> None:
    """Remove any leftover partial `.tmp*` file for `target` from an interrupted run."""
    tmp = atomic_temp_path(target)
    if tmp.exists():
        try:
            tmp.unlink()
        except OSError:
            pass


def is_valid_checkpoint(target: Path, force: bool = False) -> bool:
    """Return True if `target` already exists, is non-empty, and `force` is False."""
    cleanup_stale_temps(target)
    if force:
        return False
    return target.exists() and target.stat().st_size > 0


def atomic_temp_path(target: Path) -> Path:
    """Build a sibling temp path preserving the original file suffix (e.g. `.tmp.parquet`)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix
    stem = target.name[: -len(suffix)] if suffix else target.name
    return target.parent / f"{stem}.tmp{suffix}"


def commit_atomic(tmp_path: Path, target: Path) -> None:
    """Atomically promote a completed temporary file to its final destination."""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, target)
