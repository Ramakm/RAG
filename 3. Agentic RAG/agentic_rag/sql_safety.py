"""SQL safety: parse -> allowlist -> force LIMIT -> execute on a read-only, time-limited connection."""
import sqlite3

import sqlglot
from sqlglot import exp

from .config import settings

SCHEMA = """\
production_batches(batch_id TEXT PK, model TEXT, factory TEXT, produced_date TEXT, units_produced INT, capacitor_supplier TEXT)
  -- one row per production batch. model: 'CF-1200' and 'CF-900' (ceiling fans), 'TF-400' (table fan), 'EF-150' (exhaust fan).
  -- factory: 'Pune' or 'Hanoi'. capacitor_supplier: 'VoltCap', 'Kapex' or 'Zenith'.
sales(sale_id INT PK, model TEXT, batch_id TEXT, region TEXT, sale_date TEXT, units_sold INT)
  -- region: 'EU', 'US' or 'APAC'.
returns(return_id INT PK, model TEXT, batch_id TEXT, return_date TEXT, reason TEXT, units_returned INT)
  -- reason: 'capacitor failure', 'noise', 'wobble', 'remote issue', 'blade damage' or 'switch fault'.
qc_inspections(inspection_id INT PK, batch_id TEXT, inspection_date TEXT, units_inspected INT, units_failed INT, defect_type TEXT)
  -- end-of-line inspection sample per batch. Defect rate = SUM(units_failed) * 1.0 / SUM(units_inspected).
Dates are ISO text YYYY-MM-DD. Join tables on batch_id. Return rate = SUM(units_returned) / SUM(units_sold).
Quarters: Q1 = months 01-03, Q2 = 04-06, Q3 = 07-09, Q4 = 10-12. Fiscal year = calendar year."""

ALLOWED_TABLES = {"production_batches", "sales", "returns", "qc_inspections"}
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command, exp.Pragma)


def validate_sql(sql: str) -> str:
    """Return a safe, LIMIT-bounded SQL string, or raise ValueError."""
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.SqlglotError as e:
        raise ValueError(f"SQL does not parse: {e}")
    if len(statements) != 1 or statements[0] is None:
        raise ValueError("Exactly one statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")
    if stmt.find(*FORBIDDEN):
        raise ValueError("Write/DDL operations are not allowed.")
    cte_names = {cte.alias for cte in stmt.find_all(exp.CTE)}
    unknown = {t.name for t in stmt.find_all(exp.Table)} - cte_names - ALLOWED_TABLES
    if unknown:
        raise ValueError(f"Unknown or disallowed tables: {sorted(unknown)}. Allowed: {sorted(ALLOWED_TABLES)}")
    limit = stmt.args.get("limit")
    if limit is None:
        stmt = stmt.limit(settings.max_rows)
    return stmt.sql(dialect="sqlite")


def connect_readonly() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)  # OS-level read-only
    con.execute("PRAGMA query_only = ON")
    return con


def run_readonly(sql: str) -> tuple[list[str], list[tuple]]:
    """Execute already-validated SQL. Aborts runaway queries; returns (columns, rows)."""
    con = connect_readonly()
    try:
        ticks = {"n": 0}

        def guard() -> int:
            ticks["n"] += 1
            return 1 if ticks["n"] > 20_000 else 0  # nonzero aborts the query

        con.set_progress_handler(guard, 1000)
        cur = con.execute(sql)
        cols = [c[0] for c in cur.description]
        return cols, cur.fetchmany(settings.max_rows)
    finally:
        con.close()


def date_range() -> str:
    con = connect_readonly()
    try:
        lo, hi = con.execute("SELECT MIN(return_date), MAX(return_date) FROM returns").fetchone()
    finally:
        con.close()
    return f"{lo} to {hi}"
