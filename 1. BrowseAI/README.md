# BrowseAI — Chat With Any Website

Paste a URL and BrowseAI turns the whole website into a searchable knowledge base.
Then ask it questions and get answers that cite the pages they came from.

<img width="1267" height="684" alt="image" src="https://github.com/user-attachments/assets/72eb706e-df73-4b30-9752-70eb4b97ade1" />


Everything runs locally: open-source embeddings (sentence-transformers), a FAISS vector
index, and an open-source LLM served by [Ollama](https://ollama.com).

```
                         ┌──────────────────── INGESTION PIPELINE ────────────────────┐
  https://site.com  ──►  │ Crawl ─► Extract ─► Clean ─► Chunk ─► Embed ─► Index       │
                         │ httpx    trafilatura  cross-page  heading-  MiniLM   FAISS   │
                         │ (+JS)    + bs4        dedupe      aware     384-d    (disk)  │
                         └────────────────────────────────────────────────────────────┘
                                                                              │
  "How do I…?"  ──►  embed query ─► hybrid search (dense + keyword) ─► top-k chunks
                                                                              │
                     cited answer  ◄── Ollama LLM (streamed) ◄── prompt + context
```

## Quick start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
# 1. Start the LLM server and pull a model (one time)
ollama serve                 # leave running in its own terminal
ollama pull llama3.2         # ~2 GB; any chat model works (qwen2.5, mistral, gemma3…)

# 2. Start BrowseAI (creates .venv and installs everything on first run)
./run.sh
```

Open **http://localhost:8000**, paste a URL, press **Build knowledge base**, then **Start chatting**.

> No Ollama? BrowseAI still works: answers fall back to showing the most relevant passages.

### Optional: JavaScript-heavy sites

Single-page apps (React/Vue/Next) often ship empty HTML. Enable rendering with a headless browser:

```bash
.venv/bin/pip install playwright
.venv/bin/python -m playwright install chromium
```

Then choose **JavaScript rendering → Auto** (renders only pages that look empty) or **Always**.

## Using the app

| Area | What it does |
|---|---|
| **URL bar + Crawl settings** | Max pages, link depth, JS rendering mode, and "stay within this path" (e.g. only `/docs/`). |
| **Live pipeline** | Watch each stage light up, with counters for pages, words kept, noise removed and chunks, plus a live log of every page. |
| **Chat** | Streaming answers with numbered citations. Click a `[n]` chip to open the matching source card and see the exact passage used. |
| **Pages** | Browse and filter every page that was indexed. |
| **Model picker** | Switch between any chat model installed in Ollama. |
| **Sidebar** | All knowledge bases are saved to disk and reopen instantly. |

## How ingestion deals with messy websites

Websites are noisy, and that's the hard part. Each step targets a specific kind of junk:

1. **Crawl** ([backend/crawler.py](backend/crawler.py))
   - Breadth-first and same-site only. It follows `robots.txt`, seeds from `sitemap.xml`, and runs 6 requests at a time.
   - Normalizes URLs (drops fragments and `utm_*` tracking params, treats `/a` and `/a/` as the same page) and follows redirects without fetching a page twice.
   - Skips binaries, media, feeds and auth/cart pages.
   - **JS rendering:** if the static HTML has almost no text, or looks like an app shell (`<div id="root">`), the page is re-fetched in headless Chromium.
2. **Extract** ([backend/extractor.py](backend/extractor.py))
   - [trafilatura](https://trafilatura.readthedocs.io) pulls out the main content as Markdown, so headings, lists, tables and code survive.
   - A BeautifulSoup fallback strips `nav`/`header`/`footer`/`aside`, ARIA landmarks, and elements whose class or id looks like a menu, cookie banner, share widget or ad. It also drops link-heavy blocks.
3. **Clean** ([backend/cleaner.py](backend/cleaner.py))
   - **Cross-page boilerplate:** any block that appears on at least half of the pages is removed. Repeated CTAs, disclaimers and "related posts" blurbs disappear this way.
   - Removes junk lines (cookie notices, "All rights reserved", "Skip to content"…), duplicate blocks within a page, duplicate pages, and near-empty pages.
   - Repairs malformed code fences from extraction so code doesn't swallow the following text.
4. **Chunk** ([backend/chunker.py](backend/chunker.py))
   - Heading-aware: new sections start new chunks, long sections split on sentence boundaries (~1000 chars, 150 overlap).
   - Every chunk keeps its breadcrumb (`Page › Section › Subsection`), which is embedded alongside the text.
5. **Embed + Index** ([backend/store.py](backend/store.py))
   - `all-MiniLM-L6-v2` produces normalized vectors, stored in a FAISS inner-product index (cosine similarity). It's saved to `data/knowledge_bases/<id>/`.

**Retrieval** blends dense similarity (80%) with keyword overlap (20%). It caps results at 2 chunks per page so answers draw on several sources. It also pulls in the homepage intro for "what is this site about?" questions. Short follow-ups ("and the pricing?") are searched together with the previous question.

## Project structure

```
BrowseAI/
├── run.sh                 # one-command setup + launch
├── requirements.txt
├── backend/
│   ├── app.py             # FastAPI: REST + Server-Sent Events, serves the UI
│   ├── config.py          # settings (env-overridable)
│   ├── crawler.py         # async crawler, robots/sitemap, optional Playwright
│   ├── extractor.py       # HTML → clean Markdown
│   ├── cleaner.py         # cross-page boilerplate & duplicate removal
│   ├── chunker.py         # heading-aware chunking
│   ├── store.py           # embeddings, FAISS index, hybrid search
│   ├── llm.py             # Ollama prompt + streaming, offline fallback
│   └── ingest.py          # pipeline orchestration + progress events
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js             # vanilla JS, no build step
└── data/                  # created at runtime: saved knowledge bases
```

## Configuration

Set these environment variables before `./run.sh` if needed:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Web server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `BROWSEAI_LLM_MODEL` | auto (prefers llama3.2, qwen2.5, mistral…) | Force a specific model |
| `BROWSEAI_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model (e.g. `BAAI/bge-small-en-v1.5`). Rebuild knowledge bases after changing it. |
| `BROWSEAI_CHUNK_SIZE` / `BROWSEAI_CHUNK_OVERLAP` | `1000` / `150` | Chunking, in characters |
| `BROWSEAI_DATA_DIR` | `./data` | Where indexes are stored |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Ollama status, installed models, embedding model |
| `POST` | `/api/ingest` | `{url, max_pages, max_depth, js_mode, same_path}` → `{job_id, kb_id}` |
| `GET` | `/api/ingest/{job_id}/events` | SSE stream: `stage`, `page`, `log`, `stats`, `complete`, `error` |
| `GET` | `/api/kbs` · `/api/kbs/{id}` | List knowledge bases / details incl. pages |
| `DELETE` | `/api/kbs/{id}` | Delete a knowledge base |
| `POST` | `/api/kbs/{id}/chat` | `{question, history, model}` → SSE: `sources`, `meta`, `token`…, `done` |

Interactive docs: http://localhost:8000/docs

## Troubleshooting

- **"Ollama offline" in the sidebar.** Run `ollama serve`. If it says "no models pulled", run `ollama pull llama3.2`.
- **First build is slow.** The embedding model (~90 MB) downloads once on the first build.
- **Few or empty pages.** The site may need JavaScript: install Playwright (see above) and pick **Always**. Some sites also block crawlers in `robots.txt`, and BrowseAI respects that.
- **Answers miss things.** Raise **Max pages** or **Link depth**, or start from a more specific URL with **Stay within this path** turned on.

## Connect with me

[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/techwith_ram)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ramakm)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ramakrushnamohapatra/)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwith.ram)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@techwithram)
