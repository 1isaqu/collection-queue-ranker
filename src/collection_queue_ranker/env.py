"""Tiny `.env` reader: no dependency, never overrides variables that are already set."""

from __future__ import annotations

import os
from pathlib import Path


def parse_env_text(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines and `#` comments are skipped, quotes are stripped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_env_file(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Load `path` into `os.environ` without overriding. Returns the variables that were set."""
    file = Path(path)
    if not file.is_file():
        return {}
    applied: dict[str, str] = {}
    for key, value in parse_env_text(file.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
