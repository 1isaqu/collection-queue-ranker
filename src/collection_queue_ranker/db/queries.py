"""SELECT statements against the data contract.

One statement per contract table. Only columns the source really has are selected (optional
columns may be absent), dates are bound parameters, and the excluded-agent rule is applied in SQL,
never after the fact. Nothing here talks to a database; `tests/test_queries.py` checks the text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from collection_queue_ranker.config import Settings
from collection_queue_ranker.contract import CONTRACT, TableSpec
from collection_queue_ranker.db.base import Dialect, bind_datetime, quote_identifier
from collection_queue_ranker.errors import ContractError


@dataclass(frozen=True)
class Query:
    sql: str
    params: tuple[Any, ...] = ()


def select_columns(spec: TableSpec, available: Sequence[str]) -> list[str]:
    """Contract columns the source has, in contract order. Raises if a required one is absent."""
    have = {c.lower() for c in available}
    missing = [c for c in spec.required if c not in have]
    if missing:
        raise ContractError(f"`{spec.name}` lacks required column(s) {missing}")
    return [c for c in spec.columns if c in have]


def excluded_agent_clause(
    agents_table: str, patterns: Sequence[str], dialect: Dialect
) -> tuple[str, tuple[str, ...]]:
    """`NOT EXISTS` predicate dropping rows made by an excluded agent (LIKE patterns on login)."""
    if not patterns:
        return "", ()
    like = " OR ".join("LOWER(a.login) LIKE ?" for _ in patterns)
    table = quote_identifier(agents_table, dialect)
    sql = f"NOT EXISTS (SELECT 1 FROM {table} AS a WHERE a.agent_id = t.agent_id AND ({like}))"
    return sql, tuple(p.lower() for p in patterns)


def select_query(
    table: str,
    columns: Sequence[str],
    dialect: Dialect,
    where: Sequence[str] = (),
    params: Sequence[Any] = (),
) -> Query:
    names = ", ".join(f"t.{quote_identifier(c, dialect)}" for c in columns)
    sql = f"SELECT {names} FROM {quote_identifier(table, dialect)} AS t"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return Query(sql, tuple(params))


def _filters(
    name: str,
    columns: Sequence[str],
    settings: Settings,
    dialect: Dialect,
    agents_table: str | None,
    history_start: datetime | pd.Timestamp | None,
) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []
    if name == "debts" and history_start is not None:
        where.append(f"t.{quote_identifier('entered_at', dialect)} >= ?")
        params.append(bind_datetime(history_start, dialect))
    if name in ("contact_events", "agreements") and "agent_id" in columns and agents_table:
        clause, patterns = excluded_agent_clause(
            agents_table, settings.agents.excluded_patterns, dialect
        )
        if clause:
            where.append(clause)
            params.extend(patterns)
    return where, params


def build_contract_queries(
    settings: Settings,
    dialect: Dialect,
    available: Mapping[str, Sequence[str]],
    history_start: datetime | pd.Timestamp | None,
) -> dict[str, Query]:
    """Build one query per contract table. `history_start=None` reads every debt.

    `available` maps a contract table name to the columns the source reports for it. Physical
    table names come from `settings.source.tables` (contract name -> your table or view).
    """
    tables = settings.source.tables
    agents_table = tables.get("agents", "agents") if available.get("agents") else None
    queries: dict[str, Query] = {}
    for name, spec in CONTRACT.items():
        if spec.optional_table and not available.get(name):
            continue
        columns = select_columns(spec, available.get(name, ()))
        where, params = _filters(name, columns, settings, dialect, agents_table, history_start)
        queries[name] = select_query(tables.get(name, name), columns, dialect, where, params)
    return queries
