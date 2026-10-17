"""Public entry point: run one question through the graph and return answer + diagnostics."""
import re
import time
from contextlib import nullcontext

from .config import settings
from .graph import get_graph


def _usage_ctx():
    try:
        from langchain_core.callbacks import get_usage_metadata_callback

        return get_usage_metadata_callback()
    except ImportError:  # older langchain-core: token accounting is skipped
        return nullcontext()


def ask(question: str) -> dict:
    start = time.perf_counter()
    with _usage_ctx() as cb:
        state = get_graph().invoke({"question": question, "trace": []}, {"recursion_limit": 40})
    latency = time.perf_counter() - start

    tokens_in = tokens_out = 0
    for u in (getattr(cb, "usage_metadata", None) or {}).values():
        tokens_in += u.get("input_tokens", 0)
        tokens_out += u.get("output_tokens", 0)
    cost = (tokens_in * settings.price_in_per_mtok + tokens_out * settings.price_out_per_mtok) / 1e6

    evidence_ids = {e["id"] for e in state.get("evidence", [])}
    cited_in_text = set(re.findall(r"\[(E\d+)\]", state.get("answer", "")))
    return {
        "question": question,
        "answer": state["final_answer"],
        "sources": state.get("sources", []),
        "trace": state.get("trace", []),
        "retries": state.get("retries", 0),
        "grounded": state.get("grounded"),
        "invalid_citations": sorted(cited_in_text - evidence_ids),
        "latency_s": round(latency, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 5),
    }
