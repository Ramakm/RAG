"""LangGraph wiring: planner -> retrieve -> grade -> (rewrite -> retrieve)* -> generate -> self_check -> (generate)* -> finalize."""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from . import nodes
from .config import settings
from .state import AgentState


def after_planner(state: AgentState) -> str:
    return "out_of_scope" if state["question_type"] == "out_of_scope" or not state["sub_queries"] else "retrieve"


def after_grade(state: AgentState) -> str:
    return "generate" if state["sufficient"] or state["retries"] >= settings.max_retries else "rewrite"


def after_check(state: AgentState) -> str:
    return "finalize" if state["grounded"] or state["retries"] >= settings.max_retries else "generate"


def build_graph():
    g = StateGraph(AgentState)
    for name in ["planner", "retrieve", "grade", "rewrite", "generate", "self_check", "finalize", "out_of_scope"]:
        g.add_node(name, getattr(nodes, name))
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, ["retrieve", "out_of_scope"])
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "self_check")
    g.add_conditional_edges("self_check", after_check, ["finalize", "generate"])
    g.add_edge("finalize", END)
    g.add_edge("out_of_scope", END)
    return g.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
