"""Optional enrichment from an external provider (for example a credit-risk score).

The package ships no client and no endpoint: a provider is any object with a `fetch` method. You
write the few lines that call your vendor with your credentials, outside this repository, and plug
it in:

    provider = CachedProvider(MyVendorProvider(), TtlCache("data/enrichment.json", ttl_days=30))
    snapshots = build_snapshots(raw, prep, settings, as_of, enricher=make_enricher(provider))

Two things to know before using it:

* **Point in time.** A snapshot row describes the world at its cutoff. An external score fetched
  today describes today. Using it on past snapshots leaks the future into the past. Feed the
  provider only values that were known at `snapshot_at`, or use the enrichment for scoring only.
* **Personal data.** Keys passed to `fetch` are `product_line|debtor_id` strings. Mapping them to
  whatever your vendor needs is your provider's job and stays out of this repository.

When an `external_score` column exists, the backtest adds a baseline strategy that works the highest
external score first, so you can see whether the model adds anything over simply buying the score.
Declare the columns you want the models to use in `features.extra_numeric` / `extra_categorical`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from collection_queue_ranker.util import debtor_key

DEFAULT_COLUMNS = ("external_score", "external_rating")


class EnrichmentProvider(Protocol):
    """Anything that maps debtor keys to a dict of values (missing keys may be left out)."""

    def fetch(self, keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]: ...


class NullProvider:
    """Provides nothing. The default when no enrichment is configured."""

    def fetch(self, keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        return {}


class TtlCache:
    """JSON file cache with a time-to-live. It shields you from vendor outages and repeat cost."""

    def __init__(
        self, path: str | Path, ttl_days: float = 30.0, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = Path(path)
        self.ttl_seconds = ttl_days * 86400
        self._clock = clock
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._data.get(key)
        if entry is None or self._clock() - entry["ts"] >= self.ttl_seconds:
            return None
        return entry["value"]

    def put_many(self, values: Mapping[str, Mapping[str, Any]]) -> None:
        now = self._clock()
        for key, value in values.items():
            self._data[key] = {"ts": now, "value": dict(value)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")


class CachedProvider:
    """Ask the cache first and the wrapped provider only for the keys it cannot answer."""

    def __init__(self, inner: EnrichmentProvider, cache: TtlCache) -> None:
        self.inner = inner
        self.cache = cache
        self.calls = 0

    def fetch(self, keys: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        found: dict[str, Mapping[str, Any]] = {}
        missing = []
        for key in keys:
            hit = self.cache.get(key)
            if hit is None:
                missing.append(key)
            else:
                found[key] = hit
        if missing:
            self.calls += 1
            fresh = self.inner.fetch(missing)
            self.cache.put_many(fresh)
            self.cache.save()
            found.update(fresh)
        return found


def enrich_frame(
    frame: pd.DataFrame, provider: EnrichmentProvider, columns: Sequence[str] = DEFAULT_COLUMNS
) -> pd.DataFrame:
    """Add `columns` from `provider` to `frame` (one lookup per distinct debtor key)."""
    keys = debtor_key(frame)
    values = provider.fetch(sorted(set(keys)))
    out = frame.copy()
    for column in columns:
        mapped = keys.map(lambda k, c=column: values.get(k, {}).get(c))
        numeric = pd.to_numeric(mapped, errors="coerce")
        out[column] = numeric if numeric.notna().sum() == mapped.notna().sum() else mapped
    return out


def make_enricher(
    provider: EnrichmentProvider, columns: Sequence[str] = DEFAULT_COLUMNS
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Adapter for `build_snapshots(..., enricher=...)`."""
    return lambda frame: enrich_frame(frame, provider, columns)
