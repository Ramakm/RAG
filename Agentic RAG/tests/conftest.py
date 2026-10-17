import os
import re
import tempfile

# Must happen before the package is imported: isolate all generated data in a temp dir.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="arag_test_")

import pytest  # noqa: E402

from agentic_rag import llm as llm_module  # noqa: E402
from agentic_rag.state import Answer, Grade, Groundedness, Plan, Rewrite, SQLQuery, SubQuery  # noqa: E402

TOP_BATCH_SQL = "SELECT batch_id, SUM(units_returned) AS r FROM returns GROUP BY batch_id ORDER BY r DESC LIMIT 5"


@pytest.fixture(scope="session")
def seeded():
    from agentic_rag import seed
    from agentic_rag.ingestion.pipeline import ingest

    seed.seed_all()
    stats = ingest()
    assert not stats["errors"], stats["errors"]
    return stats


class FakeLLM:
    """Scripted stand-in for ChatAnthropic. `handlers` maps a schema name to fn(prompt, call_number) -> instance."""

    def __init__(self, handlers):
        self.handlers, self.calls = handlers, {}

    def __call__(self, model):  # replaces llm.get_llm
        return self

    def with_structured_output(self, schema):
        fake = self

        class Bound:
            def invoke(self, prompt):
                n = fake.calls[schema.__name__] = fake.calls.get(schema.__name__, 0) + 1
                return fake.handlers[schema.__name__](prompt, n)

        return Bound()


def default_handlers():
    def ids(prompt):
        return re.findall(r"^\[(E\d+)\]", prompt, re.M)

    return {
        "Plan": lambda p, n: Plan(
            question_type="root_cause",
            sub_queries=[SubQuery(text="capacitor failure noise CF-1200", tool="vector", fan_model="CF-1200"),
                         SubQuery(text="returns by batch", tool="sql")],
        ),
        "SQLQuery": lambda p, n: SQLQuery(sql=TOP_BATCH_SQL),
        "Grade": lambda p, n: Grade(relevant_ids=ids(p), sufficient=True, missing=""),
        "Rewrite": lambda p, n: Rewrite(sub_queries=[SubQuery(text="capacitor failure", tool="vector")]),
        "Answer": lambda p, n: Answer(text=f"Batch B-2025-07-A [{ids(p)[0]}]", cited_ids=[ids(p)[0]]),
        "Groundedness": lambda p, n: Groundedness(unsupported_claims=[], grounded=True),
    }


@pytest.fixture
def fake_llm(monkeypatch, seeded):
    def install(**overrides):
        fake = FakeLLM({**default_handlers(), **overrides})
        monkeypatch.setattr(llm_module, "get_llm", fake)
        return fake

    return install
