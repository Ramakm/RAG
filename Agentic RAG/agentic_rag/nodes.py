"""Graph nodes. Each takes the state and returns a partial state update."""
from datetime import date

from . import llm
from .llm import ConfigError
from .config import settings
from .prompts import GENERATOR_PROMPT, GRADER_PROMPT, PLANNER_PROMPT, REWRITER_PROMPT, SELF_CHECK_PROMPT
from .sql_safety import SCHEMA, date_range
from .state import Answer, AgentState, Evidence, Grade, Groundedness, Plan, Rewrite
from .tools import sql_tool, vector_tool

REFUSAL = "That question is outside the scope of the company's QC, supplier, sales and returns data (or asks for a change the assistant cannot make: it is read-only)."
NO_EVIDENCE = "I could not find enough information in the documents or database to answer this reliably."


def fmt_evidence(items: list[Evidence]) -> str:
    return "\n\n".join(f"[{e['id']}] ({e['source']}; {e['meta']})\n{e['content']}" for e in items) or "(none)"


def planner(state: AgentState) -> dict:
    base = {"retries": 0, "evidence": [], "candidates": [], "evidence_counter": 0, "sql_errors": []}
    prompt = PLANNER_PROMPT.format(
        schema=SCHEMA, date_range=date_range(), today=date.today().isoformat(), question=state["question"]
    )
    try:
        plan = llm.get_llm(settings.strong_model).with_structured_output(Plan).invoke(prompt)
    except ConfigError:
        raise
    except Exception as e:  # never crash on the planner: fall back to one vector search
        return {
            **base,
            "question_type": "lookup",
            "sub_queries": [{"text": state["question"], "tool": "vector"}],
            "trace": [f"planner failed ({type(e).__name__}); fell back to vector search"],
        }
    return {
        **base,
        "question_type": plan.question_type,
        "sub_queries": [q.model_dump() for q in plan.sub_queries],
        "trace": [f"plan: {plan.question_type}, {len(plan.sub_queries)} sub-queries"],
    }


def retrieve(state: AgentState) -> dict:
    counter = state.get("evidence_counter", 0)
    candidates: list[Evidence] = []
    errors = list(state.get("sql_errors", []))
    for sq in state["sub_queries"]:
        try:
            if sq["tool"] == "vector":
                for doc, meta in vector_tool(sq):
                    counter += 1
                    candidates.append({"id": f"E{counter}", "source": "vector", "content": doc, "meta": meta, "query": sq["text"]})
            else:
                text, err = sql_tool(sq["text"])
                if text:
                    counter += 1
                    candidates.append({"id": f"E{counter}", "source": "sql", "content": text, "meta": {"tool": "sql"}, "query": sq["text"]})
                else:
                    errors.append(err)
        except ConfigError:
            raise
        except Exception as e:  # a tool failure must not kill the run
            errors.append(f"{sq['tool']} tool error: {type(e).__name__}: {e}")
    return {
        "candidates": candidates,
        "evidence_counter": counter,
        "sql_errors": errors,
        "trace": [f"retrieve: {len(candidates)} candidates"],
    }


def grade(state: AgentState) -> dict:
    cands = state.get("candidates", [])
    prior = state.get("evidence", [])
    if not cands:
        return {"sufficient": False, "missing": "No results were returned.", "trace": ["grade: no candidates"]}
    prompt = GRADER_PROMPT.format(
        question=state["question"],
        sub_queries=[q["text"] for q in state["sub_queries"]],
        evidence=fmt_evidence(prior + cands),
    )
    try:
        g = llm.get_llm(settings.cheap_model).with_structured_output(Grade).invoke(prompt)
    except ConfigError:
        raise
    except Exception:
        # Fail closed: keep everything for the generator, but never mark it sufficient.
        return {"evidence": prior + cands, "candidates": [], "sufficient": False, "missing": "grader error", "trace": ["grade: error"]}
    relevant = set(g.relevant_ids)
    kept = prior + [c for c in cands if c["id"] in relevant]
    return {
        "evidence": kept,
        "candidates": [],
        "sufficient": g.sufficient,
        "missing": g.missing,
        "trace": [f"grade: kept {len(kept)} total, sufficient={g.sufficient}"],
    }


def rewrite(state: AgentState) -> dict:
    prompt = REWRITER_PROMPT.format(
        question=state["question"],
        sub_queries=state["sub_queries"],
        missing=state.get("missing", ""),
        errors=state.get("sql_errors", [])[-2:],
        schema=SCHEMA,
    )
    try:
        new = llm.get_llm(settings.cheap_model).with_structured_output(Rewrite).invoke(prompt)
        sub_queries = [q.model_dump() for q in new.sub_queries] or state["sub_queries"]
    except ConfigError:
        raise
    except Exception:
        sub_queries = state["sub_queries"]  # retry counter still advances, so the loop terminates
    n = state["retries"] + 1
    return {"sub_queries": sub_queries, "retries": n, "trace": [f"rewrite #{n}"]}


def generate(state: AgentState) -> dict:
    if not state.get("evidence"):  # nothing to ground an answer on: skip the LLM call; finalize emits the refusal
        return {"answer": "", "cited_ids": [], "trace": ["generate: skipped, no evidence"]}
    feedback = ""
    if state.get("unsupported"):
        feedback = "Your previous answer had unsupported claims; fix them or remove them:\n- " + "\n- ".join(state["unsupported"])
    prompt = GENERATOR_PROMPT.format(feedback=feedback, question=state["question"], evidence=fmt_evidence(state.get("evidence", [])))
    ans = llm.get_llm(settings.strong_model).with_structured_output(Answer).invoke(prompt)
    return {"answer": ans.text, "cited_ids": ans.cited_ids, "trace": ["generate"]}


def self_check(state: AgentState) -> dict:
    if not state.get("evidence"):
        return {"grounded": True, "unsupported": [], "trace": ["self_check: no evidence (refusal)"]}
    prompt = SELF_CHECK_PROMPT.format(answer=state["answer"], evidence=fmt_evidence(state["evidence"]))
    try:
        r = llm.get_llm(settings.strong_model).with_structured_output(Groundedness).invoke(prompt)
        grounded, unsupported = r.grounded and not r.unsupported_claims, r.unsupported_claims
    except ConfigError:
        raise
    except Exception:
        grounded, unsupported = False, []  # fail closed
    update = {"grounded": grounded, "unsupported": unsupported, "trace": [f"self_check: grounded={grounded}"]}
    if not grounded:
        update["retries"] = state["retries"] + 1
    return update


def finalize(state: AgentState) -> dict:
    evidence = state.get("evidence", [])
    text = state.get("answer", "")
    if not evidence:
        text = NO_EVIDENCE
    elif not state.get("grounded", True):
        text += "\n\n⚠ Some statements could not be verified against the sources: " + "; ".join(state.get("unsupported", []) or ["(checker unavailable)"])
    cited = set(state.get("cited_ids", []))
    sources = []
    for e in evidence:
        if e["id"] in cited:
            label = e["meta"].get("source_path", "SQL query") if e["source"] == "vector" else "SQL: " + e["content"].splitlines()[0][5:]
            sources.append({"id": e["id"], "source": e["source"], "label": label, "meta": e["meta"], "excerpt": e["content"][:400]})
    refs = "\n".join(f"[{s['id']}] {s['label']}" for s in sources)
    return {"final_answer": text + (f"\n\nSources:\n{refs}" if refs else ""), "sources": sources}


def out_of_scope(state: AgentState) -> dict:
    return {"final_answer": REFUSAL, "sources": [], "evidence": [], "trace": ["refused: out of scope"]}
