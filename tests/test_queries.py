import pandas as pd
import pytest

from collection_queue_ranker.config import Settings
from collection_queue_ranker.contract import CONTRACT
from collection_queue_ranker.db.base import Dialect, bind_datetime, quote_identifier
from collection_queue_ranker.db.queries import (
    build_contract_queries,
    excluded_agent_clause,
    select_columns,
    select_query,
)
from collection_queue_ranker.errors import ConfigError, ContractError
from collection_queue_ranker.sqlguard import assert_read_only


def _available(**overrides):
    cols = {name: list(spec.columns) for name, spec in CONTRACT.items()}
    cols.update(overrides)
    return cols


def test_identifiers_are_quoted_per_dialect():
    assert quote_identifier("debts", Dialect.SQLITE) == '"debts"'
    assert quote_identifier("dbo.debts", Dialect.SQLSERVER) == "[dbo].[debts]"


@pytest.mark.parametrize("name", ["bad name", "t; DROP TABLE x", "a.b.c.d", "1abc", "x-y", ""])
def test_unsafe_identifiers_are_refused(name):
    with pytest.raises(ConfigError):
        quote_identifier(name, Dialect.SQLSERVER)


def test_dates_are_bound_not_spliced():
    stamp = pd.Timestamp("2024-02-03 04:05:06")
    assert bind_datetime(stamp, Dialect.SQLITE) == "2024-02-03 04:05:06"
    assert bind_datetime(stamp, Dialect.SQLSERVER) == stamp.to_pydatetime()


def test_select_query_text_sqlserver():
    q = select_query("dbo.debts", ["debt_id", "balance"], Dialect.SQLSERVER,
                     ["t.[entered_at] >= ?"], ["x"])
    assert q.sql == ("SELECT t.[debt_id], t.[balance] FROM [dbo].[debts] AS t "
                     "WHERE t.[entered_at] >= ?")
    assert q.params == ("x",)


def test_every_generated_query_is_read_only_and_uses_only_contract_columns():
    for dialect in Dialect:
        queries = build_contract_queries(Settings(), dialect, _available(),
                                         pd.Timestamp("2023-01-01"))
        assert set(queries) == set(CONTRACT)
        for q in queries.values():
            assert_read_only(q.sql)
            assert q.sql.count("?") == len(q.params)


def test_debts_query_binds_the_history_start_and_can_skip_it():
    with_start = build_contract_queries(Settings(), Dialect.SQLITE, _available(),
                                        pd.Timestamp("2023-01-01"))
    assert with_start["debts"].params == ("2023-01-01 00:00:00",)
    assert "entered_at" in with_start["debts"].sql.split("WHERE")[1]
    without = build_contract_queries(Settings(), Dialect.SQLITE, _available(), None)
    assert without["debts"].params == () and "WHERE" not in without["debts"].sql


def test_excluded_agents_are_filtered_in_sql_with_bound_patterns():
    queries = build_contract_queries(Settings(), Dialect.SQLSERVER, _available(),
                                     pd.Timestamp("2023-01-01"))
    events = queries["contact_events"]
    assert "NOT EXISTS" in events.sql and "LOWER(a.login) LIKE ?" in events.sql
    assert events.params == ("test%",)
    assert "NOT EXISTS" in queries["agreements"].sql
    custom = Settings().with_overrides({"agents": {"excluded_patterns": ["qa%", "bot_%"]}})
    two = build_contract_queries(custom, Dialect.SQLITE, _available(), None)["contact_events"]
    assert two.params == ("qa%", "bot_%") and two.sql.count("LIKE ?") == 2


def test_no_exclusion_clause_without_patterns_or_agent_column():
    none = Settings().with_overrides({"agents": {"excluded_patterns": []}})
    q = build_contract_queries(none, Dialect.SQLITE, _available(), None)["contact_events"]
    assert "NOT EXISTS" not in q.sql
    assert excluded_agent_clause("agents", [], Dialect.SQLITE) == ("", ())
    no_agent_col = _available(contact_events=["event_id", "debtor_id", "result_id", "occurred_at"])
    q2 = build_contract_queries(Settings(), Dialect.SQLITE, no_agent_col, None)["contact_events"]
    assert "NOT EXISTS" not in q2.sql


def test_table_map_renames_physical_tables():
    settings = Settings().with_overrides({"source": {"tables": {"debts": "dbo.v_debts"}}})
    q = build_contract_queries(settings, Dialect.SQLSERVER, _available(), None)["debts"]
    assert "FROM [dbo].[v_debts] AS t" in q.sql


def test_only_present_columns_are_selected_and_required_ones_enforced():
    cols = _available(debts=["debt_id", "debtor_id", "entered_at", "due_at", "balance"])
    q = build_contract_queries(Settings(), Dialect.SQLITE, cols, None)["debts"]
    assert "closed_at" not in q.sql and "product_id" not in q.sql
    with pytest.raises(ContractError, match="balance"):
        select_columns(CONTRACT["debts"], ["debt_id", "debtor_id", "entered_at", "due_at"])


def test_optional_tables_may_be_absent_but_required_ones_may_not():
    cols = _available(debtors=[], agents=[], agreement_debts=[], debtor_emails=[])
    queries = build_contract_queries(Settings(), Dialect.SQLITE, cols, None)
    assert "debtors" not in queries and "agents" not in queries
    assert "NOT EXISTS" not in queries["contact_events"].sql  # no agents table to join
    with pytest.raises(ContractError):
        build_contract_queries(Settings(), Dialect.SQLITE, _available(debts=[]), None)
