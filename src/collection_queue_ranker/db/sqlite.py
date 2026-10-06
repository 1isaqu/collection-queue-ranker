"""SQLite adapter: used for the synthetic demo database and in tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from collection_queue_ranker.db.base import Dialect, quote_identifier, rows_to_frame
from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.sqlguard import assert_read_only


class SqliteSource:
    dialect = Dialect.SQLITE

    def __init__(self, name: str, path: str | Path, *, read_only: bool = True) -> None:
        self.name = name
        self.path = Path(path)
        self.read_only = read_only
        if read_only and not self.path.is_file():
            raise ConfigError(f"sqlite file not found for `{name}`: {self.path}")

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            return sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        return sqlite3.connect(str(self.path))

    def table_columns(self, table: str) -> list[str]:
        quoted = quote_identifier(table, self.dialect)
        with closing(self._connect()) as conn:
            rows = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
        return [str(r[1]).lower() for r in rows]

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        assert_read_only(sql)
        with closing(self._connect()) as conn:
            cursor = conn.execute(sql, tuple(params))
            names = [d[0] for d in cursor.description or ()]
            return rows_to_frame(names, cursor.fetchall())
