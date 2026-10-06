"""SQL Server adapter (pyodbc), configured by environment variables only.

This adapter has not been run against a live server in this repository. What is verified is
the connection-string construction, the introspection statement and the parameter passing, using
an injected fake DB-API connection (see `tests/test_adapters.py`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from typing import Any

import pandas as pd

from collection_queue_ranker.db.base import Dialect, rows_to_frame
from collection_queue_ranker.errors import ConfigError
from collection_queue_ranker.sqlguard import assert_read_only

Connect = Callable[..., Any]

_COLUMNS_SQL = (
    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?"
)


def _braced(value: str) -> str:
    return "{" + value.replace("}", "}}") + "}"


def _odbc_value(value: str) -> str:
    """Wrap a value for an ODBC connection string when it holds a delimiter."""
    return _braced(value) if any(ch in value for ch in ";{}=") else value


def _default_connect(connection_string: str, **kwargs: Any) -> Any:
    try:
        import pyodbc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConfigError(
            "the sqlserver backend needs pyodbc: pip install 'collection-queue-ranker[sqlserver]'"
        ) from exc
    return pyodbc.connect(connection_string, **kwargs)


class SqlServerSource:
    dialect = Dialect.SQLSERVER

    def __init__(
        self,
        name: str,
        database: str,
        env: Mapping[str, str] | None = None,
        connect: Connect | None = None,
    ) -> None:
        self.name = name
        self.database = database
        self.env = dict(os.environ if env is None else env)
        self._connect = connect or _default_connect

    def _get(self, key: str, default: str = "") -> str:
        return self.env.get(f"CQR_DB_{key}", default)

    def connection_string(self) -> str:
        host = self._get("HOST", "localhost")
        port = self._get("PORT", "1433")
        parts = [
            "DRIVER=" + _braced(self._get("DRIVER", "ODBC Driver 18 for SQL Server")),
            f"SERVER={_odbc_value(host + ',' + port)}",
            f"DATABASE={_odbc_value(self.database)}",
            "ApplicationIntent=ReadOnly",
            f"TrustServerCertificate={self._get('TRUST_SERVER_CERT', 'no')}",
        ]
        user = self._get("USER")
        if user:
            parts += [f"UID={_odbc_value(user)}", f"PWD={_odbc_value(self._get('PASSWORD'))}"]
        else:
            parts.append("Trusted_Connection=yes")
        return ";".join(parts) + ";"

    def _open(self) -> Any:
        timeout = int(self._get("LOGIN_TIMEOUT", "10"))
        conn = self._connect(self.connection_string(), timeout=timeout, autocommit=True)
        if hasattr(conn, "timeout"):
            conn.timeout = int(self._get("QUERY_TIMEOUT", "120"))
        return conn

    def table_columns(self, table: str) -> list[str]:
        schema, _, bare = table.rpartition(".")
        sql, params = _COLUMNS_SQL, [bare]
        wanted = schema or self._get("SCHEMA")
        if wanted:
            sql += " AND TABLE_SCHEMA = ?"
            params.append(wanted.split(".")[-1])
        with closing(self._open()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [str(r[0]).lower() for r in cursor.fetchall()]

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        assert_read_only(sql)
        with closing(self._open()) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, list(params))
            names = [d[0] for d in cursor.description or ()]
            return rows_to_frame(names, cursor.fetchall())
