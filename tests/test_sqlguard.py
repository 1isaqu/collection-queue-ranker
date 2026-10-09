import pytest

from collection_queue_ranker.errors import UnsafeSqlError
from collection_queue_ranker.sqlguard import assert_read_only


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select a, b from t where x = ?",
    "WITH c AS (SELECT 1 AS n) SELECT n FROM c;",
    "SELECT 'drop table x' AS note FROM t",
    "SELECT \"update\" FROM t",
    "SELECT [delete] FROM t",
    "-- a comment mentioning insert\nSELECT 1",
    "/* delete everything */ SELECT 1",
    "SELECT into_flag, reset_at FROM t",
])
def test_plain_reads_pass(sql):
    assert_read_only(sql)


@pytest.mark.parametrize("sql", [
    "INSERT INTO t VALUES (1)",
    "update t set a = 1",
    "DELETE FROM t",
    "DROP TABLE t",
    "SELECT * INTO copy FROM t",
    "SELECT 1; SELECT 2",
    "SELECT 1; DROP TABLE t",
    "EXEC sp_who",
    "PRAGMA table_info(t)",
    "CREATE TABLE t (a int)",
    "MERGE t USING s ON 1=1 WHEN MATCHED THEN DELETE",
    "",
    "   ;  ",
])
def test_anything_else_is_refused(sql):
    with pytest.raises(UnsafeSqlError):
        assert_read_only(sql)


def test_statement_must_start_with_select_or_with():
    with pytest.raises(UnsafeSqlError, match="must start"):
        assert_read_only("VALUES (1)")


def test_a_literal_cannot_hide_a_second_statement():
    # the quote handling must not let a doubled quote swallow the rest of the text
    with pytest.raises(UnsafeSqlError):
        assert_read_only("SELECT 'it''s'; DROP TABLE t")
