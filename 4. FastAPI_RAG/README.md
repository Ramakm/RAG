# 4. FastAPI RAG

A basic document Q&A app: upload files, ask questions, get streamed answers grounded in your documents.
Everything runs locally on **open-source Hugging Face models** — no Ollama, no API key.

| Layer | Choice |
|---|---|
| API | FastAPI (upload, list, delete, streaming `/ask` over SSE) |
| UI | Streamlit chat with source citations |
| Parsing | pypdf, python-docx, plain text / markdown |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | In-memory NumPy (cosine similarity) |
| LLM | `Qwen/Qwen2.5-1.5B-Instruct` via `transformers` |

## Workflow

```
upload → extract text → chunk (800 chars, 150 overlap) → embed → store
question → embed → top-4 chunks → prompt Qwen with context → stream answer + sources
```

## Run

```bash
./run.sh
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

The first start downloads the models (~3 GB). The index is in memory, so it resets when the backend restarts.

## Configuration

Environment variables: `LLM_MODEL` (e.g. `Qwen/Qwen2.5-3B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`),
`EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MAX_NEW_TOKENS`, `API_URL` (for the UI).

## Layout

```
backend/
  main.py     FastAPI routes
  ingest.py   text extraction + chunking
  store.py    embeddings + vector search
  llm.py      local LLM with token streaming
  config.py
app.py        Streamlit UI
```
