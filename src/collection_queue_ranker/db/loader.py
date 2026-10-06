"""Load the contract tables from one or more sources into `RawTables`."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.contract import (
    CONTRACT,
    check_available,
    read_available,
    require_ok,
)
from collection_queue_ranker.db.base import DataSource
from collection_queue_ranker.db.queries import build_contract_queries
from collection_queue_ranker.errors import DataError
from collection_queue_ranker.tables import RawTables


def load_raw_tables(
    sources: Sequence[DataSource], settings: Settings, as_of: pd.Timestamp | None = None
) -> RawTables:
    """Read every contract table from every source (one source = one product line).

    Each source is checked against the contract first, so a wrong mapping fails with a readable
    list of what is missing. Rows are tagged with the source name as `product_line`. Debts older
    than `snapshots.history_months` before `as_of` are not read; with `as_of=None` every debt is.
    """
    if not sources:
        raise DataError("no data sources given")
    history_start = None
    if as_of is not None:
        history_start = pd.Timestamp(as_of) - pd.DateOffset(months=settings.snapshots.history_months)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in CONTRACT}
    for source in sources:
        available = read_available(source, settings.source.tables)
        require_ok(check_available(available), source.name)
        queries = build_contract_queries(settings, source.dialect, available, history_start)
        for name, query in queries.items():
            frame = source.query(query.sql, query.params)
            parts[name].append(frame.assign(product_line=source.name))
    frames = {name: _concat(chunks) for name, chunks in parts.items()}
    return RawTables.from_frames(frames, settings)


def _concat(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack per-source frames, skipping empty ones (concatenating them is deprecated)."""
    filled = [c for c in chunks if not c.empty]
    if not filled:
        return chunks[0] if chunks else pd.DataFrame()
    return pd.concat(filled, ignore_index=True)
