"""Retrieval tools: vector search and guarded text-to-SQL."""
import sqlite3

from . import llm, vectorstore
from .config import settings
from .prompts import SQL_PROMPT
from .sql_safety import SCHEMA, run_readonly, validate_sql
from .state import SQLQuery


def vector_tool(sub_query: dict) -> list[tuple[str, dict]]:
    filters = {
        "model": sub_query.get("fan_model"),
        "batch_id": sub_query.get("batch_id"),
        "supplier": sub_query.get("supplier"),
        "doc_type": sub_query.get("doc_type"),
    }
    return vectorstore.search(sub_query["text"], settings.top_k, filters)


def sql_tool(query: str) -> tuple[str | None, str | None]:
    """Generate -> validate -> execute. Returns (evidence_text, error). One repair attempt using the error."""
    error = ""
    for _ in range(2):
        sql = (
            llm.get_llm(settings.strong_model)
            .with_structured_output(SQLQuery)
            .invoke(SQL_PROMPT.format(schema=SCHEMA, error=error or "none", query=query, max_rows=settings.max_rows))
            .sql
        )
        try:
            safe = validate_sql(sql)
            cols, rows = run_readonly(safe)
        except (ValueError, sqlite3.Error) as e:
            error = f"{e} | SQL was: {sql}"
            continue
        shown = rows[: settings.rows_in_prompt]
        lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in shown]
        note = f" (showing first {len(shown)})" if len(rows) > len(shown) else ""
        body = "\n".join(lines) if rows else "(0 rows)"
        return f"SQL: {safe}\nRows ({len(rows)}){note}:\n{body}", None
    return None, error
