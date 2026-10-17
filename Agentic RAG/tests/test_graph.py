"""End-to-end graph tests with a scripted fake LLM (real vector store, real SQL, real guards)."""
from agentic_rag import agent
from agentic_rag.nodes import NO_EVIDENCE, REFUSAL
from agentic_rag.sql_safety import connect_readonly
from agentic_rag.state import Grade, Groundedness, Plan, SQLQuery


def test_happy_path(fake_llm):
    fake = fake_llm()
    r = agent.ask("Why did returns spike for CF-1200?")
    assert r["retries"] == 0 and "B-2025-07-A" in r["answer"]
    assert r["sources"] and r["invalid_citations"] == []
    assert fake.calls["Plan"] == 1 and fake.calls["Grade"] == 1


def test_sql_result_reaches_the_evidence(fake_llm):
    fake_llm()
    r = agent.ask("anything")
    assert any(s["source"] == "sql" or s["source"] == "vector" for s in r["sources"])
    assert any("retrieve: " in t for t in r["trace"])


def test_rewrite_loop_runs_then_succeeds(fake_llm):
    ids = lambda p: __import__("re").findall(r"^\[(E\d+)\]", p, __import__("re").M)
    fake = fake_llm(Grade=lambda p, n: Grade(relevant_ids=ids(p), sufficient=n >= 2, missing="need more"))
    r = agent.ask("q")
    assert r["retries"] == 1 and any(t.startswith("rewrite #1") for t in r["trace"])
    assert fake.calls["Rewrite"] == 1


def test_retry_cap_is_enforced(fake_llm):
    ids = lambda p: __import__("re").findall(r"^\[(E\d+)\]", p, __import__("re").M)
    fake = fake_llm(Grade=lambda p, n: Grade(relevant_ids=ids(p), sufficient=False, missing="never enough"))
    r = agent.ask("q")
    assert r["retries"] == 3 and fake.calls["Rewrite"] == 3 and r["answer"]


def test_ungrounded_answer_is_flagged_after_cap(fake_llm):
    fake = fake_llm(Groundedness=lambda p, n: Groundedness(unsupported_claims=["invented number"], grounded=False))
    r = agent.ask("q")
    assert r["retries"] == 3 and fake.calls["Answer"] == 3
    assert "could not be verified" in r["answer"] and "invented number" in r["answer"]


def test_out_of_scope_skips_retrieval(fake_llm):
    fake = fake_llm(Plan=lambda p, n: Plan(question_type="out_of_scope", sub_queries=[]))
    r = agent.ask("weather in Paris?")
    assert r["answer"] == REFUSAL and "Grade" not in fake.calls and "SQLQuery" not in fake.calls


def test_destructive_sql_is_blocked_and_db_untouched(fake_llm):
    con = connect_readonly()
    before = con.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    con.close()
    fake = fake_llm(
        Plan=lambda p, n: Plan(question_type="lookup", sub_queries=[{"text": "delete returns", "tool": "sql"}]),
        SQLQuery=lambda p, n: SQLQuery(sql="DELETE FROM returns"),
        Grade=lambda p, n: Grade(relevant_ids=[], sufficient=False, missing="no data"),
    )
    r = agent.ask("Delete all returns")
    con = connect_readonly()
    assert con.execute("SELECT COUNT(*) FROM returns").fetchone()[0] == before
    con.close()
    assert r["answer"] == NO_EVIDENCE


def test_planner_failure_falls_back_to_vector_search(fake_llm):
    def boom(p, n):
        raise RuntimeError("API down")

    fake_llm(Plan=boom)
    r = agent.ask("CF-1200 humming")
    assert any("planner failed" in t for t in r["trace"]) and r["answer"]


def test_api_endpoints(fake_llm):
    from fastapi.testclient import TestClient

    from agentic_rag.api import app

    fake_llm()
    c = TestClient(app)
    assert c.get("/health").json()["chunks_indexed"] > 100
    assert c.post("/ask", json={"question": "Why did returns spike?"}).json()["answer"]
    assert c.post("/ask", json={"question": "x"}).status_code == 422
