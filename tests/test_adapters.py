import sqlite3

import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.contract import check_source, format_checks, require_ok
from collection_queue_ranker.db.base import Dialect, rows_to_frame
from collection_queue_ranker.db.loader import load_raw_tables
from collection_queue_ranker.db.sources import make_sources, parse_sources
from collection_queue_ranker.db.sqlite import SqliteSource
from collection_queue_ranker.db.sqlserver import SqlServerSource
from collection_queue_ranker.errors import ConfigError, ContractError, DataError, UnsafeSqlError


@pytest.fixture
def small_db(tmp_path):
    path = tmp_path / "t.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE debts (debt_id INTEGER, balance REAL, entered_at TEXT)")
        conn.executemany("INSERT INTO debts VALUES (?, ?, ?)",
                         [(1, 10.5, "2024-01-01 00:00:00"), (2, 20.0, "2024-03-01 00:00:00")])
    return path


def test_sqlite_columns_query_and_params(small_db):
    source = SqliteSource("auto", small_db)
    assert source.table_columns("debts") == ["debt_id", "balance", "entered_at"]
    assert source.table_columns("nothing") == []
    frame = source.query("SELECT debt_id FROM debts WHERE entered_at >= ?", ["2024-02-01 00:00:00"])
    assert frame["debt_id"].tolist() == [2]


def test_sqlite_refuses_writes_and_missing_files(small_db, tmp_path):
    source = SqliteSource("auto", small_db)
    with pytest.raises(UnsafeSqlError):
        source.query("DELETE FROM debts")
    with pytest.raises(ConfigError, match="not found"):
        SqliteSource("auto", tmp_path / "absent.db")
    # even a statement that slipped past the guard could not write: the file opens read-only
    with pytest.raises(sqlite3.OperationalError):
        source._connect().execute("DELETE FROM debts")


def test_rows_to_frame_lowercases_and_handles_empty():
    assert rows_to_frame(["A", "B"], [(1, 2)]).columns.tolist() == ["a", "b"]
    assert rows_to_frame(["A"], []).empty


class _Cursor:
    def __init__(self, conn):
        self.conn, self.description = conn, None

    def execute(self, sql, params=None):
        self.conn.calls.append((sql, params))
        if "INFORMATION_SCHEMA" in sql:
            self.description, self.conn.rows = [("COLUMN_NAME",)], [("Debt_ID",), ("BALANCE",)]
        else:
            self.description, self.conn.rows = [("A",), ("B",)], [(1, "x"), (2, "y")]

    def fetchall(self):
        return self.conn.rows


class _Conn:
    def __init__(self):
        self.calls, self.rows, self.closed, self.timeout = [], [], False, 0

    def cursor(self):
        return _Cursor(self)

    def close(self):
        self.closed = True


def _server(env=None, conn=None):
    conn = conn or _Conn()
    seen = {}

    def connect(string, **kwargs):
        seen["string"], seen["kwargs"] = string, kwargs
        return conn

    return SqlServerSource("auto", "ledger_a", env or {}, connect), conn, seen


def test_sqlserver_connection_string_with_sql_login():
    env = {"CQR_DB_HOST": "db.example.invalid", "CQR_DB_PORT": "1444",
           "CQR_DB_USER": "report_reader", "CQR_DB_PASSWORD": "example;pw}x",
           "CQR_DB_TRUST_SERVER_CERT": "yes"}
    source, _, _ = _server(env)
    text = source.connection_string()
    assert text.startswith("DRIVER={ODBC Driver 18 for SQL Server};SERVER=db.example.invalid,1444;")
    assert "DATABASE=ledger_a" in text and "ApplicationIntent=ReadOnly" in text
    assert "TrustServerCertificate=yes" in text and "UID=report_reader" in text
    assert "PWD={example;pw}}x}" in text  # delimiters inside a value are brace-escaped
    assert "Trusted_Connection" not in text


def test_sqlserver_integrated_login_when_no_user():
    text = _server({})[0].connection_string()
    assert "Trusted_Connection=yes" in text and "UID=" not in text and "PWD=" not in text


def test_sqlserver_introspection_uses_bound_parameters_and_closes():
    source, conn, seen = _server({"CQR_DB_SCHEMA": "dbo", "CQR_DB_LOGIN_TIMEOUT": "7"})
    assert source.table_columns("debts") == ["debt_id", "balance"]
    sql, params = conn.calls[0]
    assert "INFORMATION_SCHEMA.COLUMNS" in sql and "TABLE_SCHEMA = ?" in sql
    assert params == ["debts", "dbo"]
    assert seen["kwargs"] == {"timeout": 7, "autocommit": True}
    assert conn.closed
    source.table_columns("sales.debts")
    assert conn.calls[1][1] == ["debts", "sales"]


def test_sqlserver_applies_the_configured_char_encoding():
    class _Decoding(_Conn):
        def __init__(self):
            super().__init__()
            self.decoding = []

        def setdecoding(self, kind, encoding=None):
            self.decoding.append((kind, encoding))

    conn = _Decoding()
    _server({"CQR_DB_CHAR_ENCODING": "cp1252"}, conn)[0].table_columns("debts")
    assert conn.decoding == [(1, "cp1252")]
    untouched = _Decoding()
    _server({}, untouched)[0].table_columns("debts")
    assert untouched.decoding == []


def test_sqlserver_query_is_guarded_and_parameterised():
    source, conn, _ = _server({})
    frame = source.query("SELECT a, b FROM t WHERE x = ?", (5,))
    assert frame.to_dict("list") == {"a": [1, 2], "b": ["x", "y"]}
    assert conn.calls[0] == ("SELECT a, b FROM t WHERE x = ?", [5])
    with pytest.raises(UnsafeSqlError):
        source.query("UPDATE t SET a = 1")
    assert source.dialect is Dialect.SQLSERVER


def test_parse_sources_and_make_sources(small_db):
    assert parse_sources("auto=a.db, consumer=b.db") == {"auto": "a.db", "consumer": "b.db"}
    with pytest.raises(ConfigError):
        parse_sources("auto")
    made = make_sources(Settings(), {}, backend="sqlite", sources={"auto": str(small_db)})
    assert [s.name for s in made] == ["auto"]
    env_made = make_sources(Settings(), {"CQR_SOURCES": f"a={small_db}"}, backend="sqlite")
    assert env_made[0].name == "a"
    server = make_sources(Settings(), {}, backend="sqlserver", sources={"auto": "ledger_a"})
    assert isinstance(server[0], SqlServerSource)
    with pytest.raises(ConfigError, match="no sources"):
        make_sources(Settings(), {})
    with pytest.raises(ConfigError, match="unknown backend"):
        make_sources(Settings(), {}, backend="oracle", sources={"a": "b"})


def test_check_source_reports_missing_tables_and_required_columns(small_db):
    checks = check_source(SqliteSource("auto", small_db))
    by_table = {c.table: c for c in checks}
    assert by_table["debts"].missing_required == ("debtor_id", "due_at")
    assert not by_table["debts"].ok and not by_table["agreements"].ok
    assert by_table["debtors"].ok  # optional table, absent
    assert "MISSING" in format_checks(checks) and "absent" in format_checks(checks)
    with pytest.raises(ContractError) as err:
        require_ok(checks, "auto")
    assert "debts" in str(err.value) and "agreements" in str(err.value)


def test_loader_reads_a_full_contract_database(tmp_path):
    from collection_queue_ranker import synth

    path = synth.write_sqlite(synth.generate(60, seed=2, months=8), tmp_path / "x.db")
    raw = load_raw_tables([SqliteSource("auto", path)], Settings(), pd.Timestamp("2024-12-31"))
    assert set(raw.debts["product_line"]) == {"auto"}
    assert raw.debts["entered_at"].dtype.kind == "M"
    assert {"category", "is_automated"} <= set(raw.events.columns)
    # the excluded-agent rule ran in SQL: no event belongs to the test login (agent_id 11)
    assert 11 not in set(raw.events["agent_id"].dropna())
    with pytest.raises(DataError):
        load_raw_tables([], Settings())


def test_loader_applies_the_history_window_to_debts(tmp_path):
    from collection_queue_ranker import synth

    frames = synth.generate(60, seed=2, months=8)
    path = synth.write_sqlite(frames, tmp_path / "x.db")
    source = SqliteSource("auto", path)
    everything = load_raw_tables([source], Settings())
    recent = load_raw_tables([source], Settings().with_overrides(
        {"snapshots": {"history_months": 2}}), pd.Timestamp("2024-12-31"))
    assert len(recent.debts) < len(everything.debts)
    assert recent.debts["entered_at"].min() >= pd.Timestamp("2024-10-31")
