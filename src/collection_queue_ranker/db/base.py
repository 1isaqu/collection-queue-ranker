"""Source protocol shared by every adapter, plus the pieces all adapters need."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from collection_queue_ranker.errors import ConfigError


class Dialect(str, Enum):
    SQLITE = "sqlite"
    SQLSERVER = "sqlserver"


class DataSource(Protocol):
    """A read-only source for one product line."""

    name: str
    dialect: Dialect

    def table_columns(self, table: str) -> list[str]:
        """Lower-case column names of `table`, empty when the table does not exist."""
        ...

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        """Run one read-only statement with positional (`?`) parameters."""
        ...


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(name: str, dialect: Dialect) -> str:
    """Quote a table or column name, one dot-separated part at a time.

    Only plain identifiers are accepted. Anything else is refused instead of escaped: map odd
    names to plain ones with a view in your database.
    """
    parts = name.split(".")
    if not 1 <= len(parts) <= 3 or not all(_IDENT.match(p) for p in parts):
        raise ConfigError(f"unsupported identifier {name!r}: use plain names (letters, digits, _)")
    if dialect is Dialect.SQLSERVER:
        return ".".join(f"[{p}]" for p in parts)
    return ".".join(f'"{p}"' for p in parts)


def bind_datetime(value: datetime | pd.Timestamp, dialect: Dialect) -> Any:
    """Dates always travel as bound parameters, never as text spliced into the statement.

    Splicing a literal makes the result depend on the server's date-format setting. SQLite has no
    date type, so it receives the same ISO text it stores.
    """
    ts = pd.Timestamp(value).to_pydatetime()
    if dialect is Dialect.SQLITE:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts


def rows_to_frame(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Build a frame from DB-API rows. Column names are lower-cased."""
    names = [c.lower() for c in columns]
    if not rows:
        return pd.DataFrame(columns=names)
    return pd.DataFrame.from_records([tuple(r) for r in rows], columns=names)
