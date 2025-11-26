"""Helpers for exposing vendored dependencies on ``sys.path``."""

from __future__ import annotations

from pathlib import Path
import sys


def repo_root() -> Path:
    """Return the repository root (services/..)."""

    return Path(__file__).resolve().parents[1]


def prepend_repo_path(*relative_parts: str) -> Path:
    """Ensure a vendored directory is importable and return its path."""

    path = repo_root().joinpath(*relative_parts)
    str_path = str(path)
    if path.exists() and str_path not in sys.path:
        sys.path.insert(0, str_path)
    return path


__all__ = ["repo_root", "prepend_repo_path"]
