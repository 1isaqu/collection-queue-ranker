"""Build data sources from settings and environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping

from collection_queue_ranker.config import Settings
from collection_queue_ranker.db.base import DataSource
from collection_queue_ranker.db.sqlite import SqliteSource
from collection_queue_ranker.db.sqlserver import SqlServerSource
from collection_queue_ranker.errors import ConfigError

BACKENDS = ("sqlite", "sqlserver")


def parse_sources(spec: str) -> dict[str, str]:
    """Parse `auto=path_or_db,consumer=path_or_db` into an ordered mapping."""
    out: dict[str, str] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        line, sep, target = item.partition("=")
        if not sep or not line.strip() or not target.strip():
            raise ConfigError(f"bad source entry {item!r}; expected <product_line>=<target>")
        out[line.strip()] = target.strip()
    return out


def make_sources(
    settings: Settings,
    env: Mapping[str, str] | None = None,
    *,
    backend: str | None = None,
    sources: Mapping[str, str] | None = None,
) -> list[DataSource]:
    """One source per product line. Explicit arguments beat environment, which beats settings."""
    env = os.environ if env is None else env
    backend = backend or env.get("CQR_BACKEND") or settings.source.backend
    if backend not in BACKENDS:
        raise ConfigError(f"unknown backend {backend!r}; choose one of {BACKENDS}")
    mapping = dict(sources or {})
    if not mapping and env.get("CQR_SOURCES"):
        mapping = parse_sources(env["CQR_SOURCES"])
    if not mapping:
        mapping = dict(settings.source.sources)
    if not mapping:
        raise ConfigError(
            "no sources configured: pass --source <product_line>=<target>, set CQR_SOURCES, "
            "or fill `source.sources` in the settings file"
        )
    if backend == "sqlite":
        return [SqliteSource(line, target) for line, target in mapping.items()]
    return [SqlServerSource(line, target, env) for line, target in mapping.items()]
