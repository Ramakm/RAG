import pytest

from agentic_rag.sql_safety import validate_sql


def test_valid_select_gets_limit():
    out = validate_sql("select model, sum(units_returned) from returns group by model")
    assert out.upper().endswith("LIMIT 200")


def test_existing_limit_is_kept():
    assert validate_sql("select batch_id from returns limit 3").endswith("LIMIT 3")


def test_cte_is_allowed():
    validate_sql("with x as (select batch_id from returns) select * from x")


@pytest.mark.parametrize("sql", [
    "drop table returns",
    "delete from returns",
    "update returns set units_returned = 0",
    "insert into returns values (1,'a','b','c','d',1)",
    "select 1; delete from returns",
    "with x as (select 1) insert into returns select * from x",
    "select * from sqlite_master",
    "select * from users",
    "pragma table_info(returns)",
    "this is not sql",
])
def test_rejected(sql):
    with pytest.raises(ValueError):
        validate_sql(sql)


def test_connection_is_read_only(seeded):
    import sqlite3

    from agentic_rag.sql_safety import run_readonly

    with pytest.raises(sqlite3.Error):
        run_readonly("delete from returns")
