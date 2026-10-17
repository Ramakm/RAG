"""Golden-set evaluation. Checks are deterministic (no LLM judge), so runs are cheap and comparable over time.

Per case: gold-SQL entity must appear in the answer, required/forbidden phrases, refusal behaviour, and the
database must be unchanged afterwards. Aggregate: pass rate by category, latency, tokens, retries, invalid citations.
"""
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from . import agent
from .config import ROOT, settings
from .nodes import REFUSAL
from .sql_safety import connect_readonly, run_readonly

GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS = ROOT / "eval" / "results"


def _norm(s: str) -> str:
    return s.lower().replace(",", "")


def _db_fingerprint() -> tuple:
    con = connect_readonly()
    try:
        return tuple(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("returns", "sales", "production_batches", "qc_inspections"))
    finally:
        con.close()


def check_case(case: dict, result: dict) -> list[str]:
    """Return a list of failure messages (empty = pass)."""
    answer, norm = result["answer"], _norm(result["answer"])
    fails = []
    if case.get("gold_sql"):
        _, rows = run_readonly(case["gold_sql"])
        expected = str(rows[0][0]) if rows else ""
        if not expected or _norm(expected) not in norm:
            fails.append(f"gold value {expected!r} not in answer")
    for s in case.get("must_contain", []):
        if _norm(s) not in norm:
            fails.append(f"missing {s!r}")
    any_of = case.get("must_contain_any")
    if any_of and not any(_norm(s) in norm for s in any_of):
        fails.append(f"none of {any_of} present")
    for s in case.get("must_not_contain", []):
        if _norm(s) in norm:
            fails.append(f"forbidden {s!r} present")
    if case.get("expect_refusal") and result["answer"] != REFUSAL and not any(w in norm for w in ("cannot", "can't", "outside", "read-only", "not able", "unable")):
        fails.append("expected a refusal")
    if result["invalid_citations"]:
        fails.append(f"invalid citations {result['invalid_citations']}")
    return fails


def run_eval(only: list[str] | None = None) -> int:
    cases = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if only:
        cases = [c for c in cases if c["id"] in only]
    before = _db_fingerprint()
    rows = []
    for c in cases:
        try:
            r = agent.ask(c["question"])
            fails = check_case(c, r)
        except Exception as e:
            r = {"answer": "", "latency_s": 0, "tokens_in": 0, "tokens_out": 0, "retries": 0, "cost_usd": 0, "trace": [], "invalid_citations": []}
            fails = [f"crashed: {type(e).__name__}: {e}"]
        rows.append({**c, "result": r, "failures": fails, "passed": not fails})
        print(f"{'PASS' if not fails else 'FAIL'}  {c['id']}  {r['latency_s']:>6.1f}s  {c['question'][:70]}" + ("" if not fails else f"\n      -> {'; '.join(fails)}"))
    if _db_fingerprint() != before:
        print("!! DATABASE CHANGED DURING EVAL: the read-only guard failed")
        return 2

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["passed"])
    lat = sorted(r["result"]["latency_s"] for r in rows)
    summary = {
        "cases": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / max(len(rows), 1), 3),
        "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()},
        "latency_p50_s": round(statistics.median(lat), 1) if lat else 0,
        "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0,
        "avg_tokens_in": round(statistics.mean(r["result"]["tokens_in"] for r in rows)) if rows else 0,
        "avg_tokens_out": round(statistics.mean(r["result"]["tokens_out"] for r in rows)) if rows else 0,
        "avg_cost_usd": round(statistics.mean(r["result"]["cost_usd"] for r in rows), 4) if rows else 0,
        "needed_rewrite_or_regen": sum(1 for r in rows if r["result"]["retries"] > 0),
        "hit_retry_cap": sum(1 for r in rows if r["result"]["retries"] >= settings.max_retries),
        "models": {"strong": settings.strong_model, "cheap": settings.cheap_model},
    }
    print("\n" + json.dumps(summary, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, indent=2, default=str))
    print(f"\nSaved {out}")
    return 0 if summary["pass_rate"] >= 0.8 else 1
