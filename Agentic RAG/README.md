# Agentic RAG for a Fan Manufacturer

Ask questions in plain English across QC reports, supplier specs, manuals, service tickets **and** the sales/returns database. A LangGraph agent plans the query, retrieves from the right source (vector search and/or guarded SQL), grades the evidence, rewrites weak queries (max 3 retries), writes a cited answer and fact-checks it against the sources.

Design rationale, prompts, evaluation plan and roadmap: [DESIGN.md](DESIGN.md).

```
question -> planner -> retrieve (vector | SQL) -> grader -> [rewriter -> retrieve]* -> generator -> groundedness check -> answer + sources
```

## Quick start

```bash
cd "Agentic RAG"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY

make demo                       # build demo DB + 82 documents, then index them (first run downloads a ~80 MB embedding model)
make ask Q="Why did returns spike for the CF-1200 ceiling fan in Q3 2025, and which batch caused it?"
make ui                         # Streamlit chat at http://localhost:8501
make api                        # REST API at http://localhost:8000/docs
make eval                       # 20-question golden set (needs the API key)
make test                       # 31 offline tests, no API key needed
```

## Layout

| Path | What |
|---|---|
| `agentic_rag/graph.py`, `nodes.py` | The LangGraph workflow and its 8 nodes |
| `agentic_rag/prompts.py` | Every prompt |
| `agentic_rag/sql_safety.py` | SQL guard: sqlglot allowlist, forced LIMIT, read-only connection, timeout |
| `agentic_rag/tools.py`, `vectorstore.py` | Vector search (Chroma) and text-to-SQL tools |
| `agentic_rag/ingestion/` | Parsers (PDF, DOCX, XLSX, MD/TXT), type-aware chunking, metadata, incremental pipeline |
| `agentic_rag/seed.py` | Deterministic demo data with two planted stories (a bad capacitor lot; a noisy table-fan batch) |
| `agentic_rag/api.py`, `app/streamlit_app.py` | FastAPI service and chat UI |
| `agentic_rag/evaluation.py`, `eval/golden.jsonl` | Deterministic golden-set eval (no LLM judge) |
| `tests/` | Offline tests using a scripted fake LLM against the real index and database |

## Using your own data

1. Point `DATA_DIR` at a folder containing `docs/<type>/...` where `<type>` is one of `qc_reports`, `supplier_specs`, `manuals`, `service_tickets`, and a SQLite file `fans.db`.
2. Edit `SCHEMA` in `sql_safety.py` (table/column meanings and example values) and `ALLOWED_TABLES`. This description is the biggest SQL-accuracy lever.
3. Adapt the ID patterns in `ingestion/metadata.py` (batch, model, lot, supplier names).
4. `python -m agentic_rag.cli ingest` is incremental: unchanged files are skipped, changed ones re-indexed, deleted ones removed. Schedule it nightly.

## Status

Verified: 31 tests pass offline, covering the graph (happy path, rewrite loop, 3-retry cap, ungrounded caveat, out-of-scope, blocked destructive SQL, planner failure), the SQL guard, ingestion (incremental, docx tables, Excel row metadata) and the API.

Not yet verified: runs against the live Anthropic API (needs your key), so `make eval` is the first real quality measurement; PDF parsing (implemented with pypdf, no sample PDF in the demo set; scanned PDFs need OCR); the Streamlit UI beyond a syntax check.

Known simplifications vs. DESIGN.md: SQLite + Chroma instead of Postgres + pgvector; Chroma's default embedder instead of bge-m3; no reranker, hybrid search, web tool, parallel sub-queries or Langfuse yet.
