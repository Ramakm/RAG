"""Shared graph state and the structured outputs the LLM nodes return."""
import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class SubQuery(BaseModel):
    text: str = Field(description="Self-contained sub-question or search query.")
    tool: Literal["vector", "sql"]
    # Optional metadata filters for the vector tool. Set ONLY when the question names the value explicitly.
    fan_model: Optional[str] = Field(default=None, description="e.g. 'CF-1200'")
    batch_id: Optional[str] = Field(default=None, description="e.g. 'B-2025-07-A'")
    supplier: Optional[str] = None
    doc_type: Optional[Literal["qc_report", "supplier_spec", "manual", "service_ticket"]] = None


class Plan(BaseModel):
    question_type: Literal["lookup", "aggregate", "root_cause", "comparison", "summary", "out_of_scope"]
    sub_queries: list[SubQuery] = Field(max_length=4)


class SQLQuery(BaseModel):
    sql: str


class Grade(BaseModel):
    relevant_ids: list[str]
    sufficient: bool
    missing: str = Field(default="", description="What is still missing, or empty if sufficient.")


class Rewrite(BaseModel):
    sub_queries: list[SubQuery] = Field(max_length=4)


class Answer(BaseModel):
    text: str
    cited_ids: list[str]


class Groundedness(BaseModel):
    unsupported_claims: list[str]
    grounded: bool


class Evidence(TypedDict):
    id: str  # "E1": stable across loops, cited by the generator
    source: Literal["vector", "sql"]
    content: str
    meta: dict
    query: str


class AgentState(TypedDict, total=False):
    question: str
    question_type: str
    sub_queries: list[dict]
    candidates: list[Evidence]  # fresh results from the latest retrieve step
    evidence: list[Evidence]  # accumulated, grader-approved
    evidence_counter: int
    sufficient: bool
    missing: str
    sql_errors: list[str]
    retries: int  # shared budget: rewrites + regenerations
    answer: str
    cited_ids: list[str]
    grounded: bool
    unsupported: list[str]
    final_answer: str
    sources: list[dict]
    trace: Annotated[list[str], operator.add]
