"""Chat UI.  Run from the project root:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from agentic_rag import agent, vectorstore  # noqa: E402
from agentic_rag.config import settings  # noqa: E402
from agentic_rag.ingestion.pipeline import ingest  # noqa: E402

st.set_page_config(page_title="Fan Ops Assistant", page_icon="🌀", layout="wide")
st.title("🌀 Fan Ops Assistant")
st.caption("Agentic RAG over QC reports, supplier specs, manuals, service tickets and the sales/returns database.")

with st.sidebar:
    st.subheader("Index")
    st.write(f"Chunks indexed: **{vectorstore.count()}**")
    if st.button("Re-run ingestion"):
        with st.spinner("Ingesting..."):
            st.json(ingest())
    st.subheader("Try")
    examples = [
        "Why did returns spike for the CF-1200 ceiling fan in Q3 2025, and which batch caused it?",
        "Which supplier's capacitors have the highest failure rate this year?",
        "Summarize all customer complaints about noise for model TF-400.",
        "Compare defect rates between our two factories for Q3 2025.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

if "history" not in st.session_state:
    st.session_state["history"] = []

for turn in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])

question = st.chat_input("Ask about returns, batches, suppliers, complaints...") or st.session_state.pop("pending", None)
if question:
    if not settings.db_path.exists():
        st.error("No demo data yet. Run `python -m agentic_rag.cli seed` then `ingest`.")
        st.stop()
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Planning, retrieving, checking..."):
            r = agent.ask(question)
        st.markdown(r["answer"])
        st.caption(f"{r['latency_s']}s · {r['tokens_in']} in / {r['tokens_out']} out tokens · retries: {r['retries']}")
        with st.expander("How the agent got here"):
            st.write("\n".join(f"- {t}" for t in r["trace"]))
        for s in r["sources"]:
            with st.expander(f"[{s['id']}] {s['label']}"):
                st.text(s["excerpt"])
    st.session_state["history"].append({"question": question, "answer": r["answer"]})
