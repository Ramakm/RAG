import json
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FastAPI RAG", page_icon="📄")
st.title("📄 Chat with your documents")
st.caption("FastAPI backend · open-source models via Hugging Face · no Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("Sources"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']}** — similarity {s['score']:.2f}")
            st.caption(s["text"])


def stream_answer(question, sources_out):
    with requests.post(f"{API}/ask", json={"question": question}, stream=True, timeout=600) as r:
        if r.status_code != 200:
            yield f"⚠️ {r.json().get('detail', r.text)}"
            return
        event = None
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:])
                if event == "sources":
                    sources_out.extend(payload)
                elif event == "token":
                    yield payload


with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("PDF, DOCX, TXT or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if files and st.button("Index documents", type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                r = requests.post(f"{API}/documents", files={"file": (f.name, f.getvalue())})
            if r.ok:
                st.success(f"{f.name}: {r.json()['chunks']} chunks")
            else:
                st.error(f"{f.name}: {r.json().get('detail', r.text)}")
    try:
        for d in requests.get(f"{API}/documents", timeout=5).json():
            c1, c2 = st.columns([4, 1])
            c1.write(f"{d['name']} ({d['chunks']})")
            if c2.button("✕", key=f"del-{d['name']}"):
                requests.delete(f"{API}/documents/{d['name']}")
                st.rerun()
    except requests.RequestException:
        st.error(f"Backend not reachable at {API}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        sources: list = []
        answer = st.write_stream(stream_answer(question, sources))
        if sources:
            show_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
