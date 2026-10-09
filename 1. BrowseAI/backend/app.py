"""BrowseAI API server + static UI.

Run from the BrowseAI folder:  uvicorn backend.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import EMBED_MODEL, FRONTEND_DIR
from .ingest import Job, run_ingest
from .llm import build_messages, extractive_answer, ollama_status, stream_answer
from .store import KBManager

app = FastAPI(title="BrowseAI", version="1.0")
kbs = KBManager()
jobs: dict[str, Job] = {}
_tasks: set[asyncio.Task] = set()
KB_ID = re.compile(r"^[a-z0-9-]{1,80}$")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_kb(kb_id: str):
    if not KB_ID.match(kb_id):
        raise HTTPException(400, "Invalid knowledge base id")
    try:
        return kbs.get(kb_id)
    except KeyError:
        raise HTTPException(404, "Knowledge base not found")


class IngestRequest(BaseModel):
    url: str
    max_pages: int = Field(25, ge=1, le=300)
    max_depth: int = Field(2, ge=0, le=5)
    js_mode: str = Field("auto", pattern="^(off|auto|always)$")
    same_path: bool = False


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = []
    model: str | None = None
    top_k: int = Field(5, ge=1, le=12)


@app.get("/api/health")
async def health():
    return {"ollama": await ollama_status(), "embed_model": EMBED_MODEL}


@app.get("/api/kbs")
def list_kbs():
    return kbs.list()


@app.get("/api/kbs/{kb_id}")
def kb_detail(kb_id: str):
    return get_kb(kb_id).meta


@app.delete("/api/kbs/{kb_id}")
def delete_kb(kb_id: str):
    get_kb(kb_id)
    kbs.delete(kb_id)
    return {"ok": True}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    url = req.url.strip()
    if not urlparse(url).scheme:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or "." not in parsed.netloc and "localhost" not in parsed.netloc:
        raise HTTPException(400, "Please enter a valid website URL, e.g. https://example.com")
    slug = re.sub(r"[^a-z0-9]+", "-", parsed.netloc.lower().removeprefix("www."))[:40].strip("-")
    job = Job(kb_id=f"{slug}-{uuid.uuid4().hex[:6]}")
    jobs[job.id] = job
    opts = req.model_dump(exclude={"url"})
    task = asyncio.create_task(run_ingest(job, kbs, url, opts))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job_id": job.id, "kb_id": job.kb_id}


@app.get("/api/ingest/{job_id}/events")
async def ingest_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def gen():
        async for ev in job.listen():
            yield sse(ev["type"], ev)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.post("/api/kbs/{kb_id}/chat")
async def chat(kb_id: str, req: ChatRequest):
    kb = get_kb(kb_id)
    # Short follow-ups ("and the price?") are retrieved together with the previous question.
    query = req.question
    prev_q = next((t["content"] for t in reversed(req.history) if t.get("role") == "user"), "")
    if prev_q and len(req.question.split()) < 8:
        query = f"{prev_q} {req.question}"
    sources = await asyncio.to_thread(kb.search, query, req.top_k)

    async def gen():
        yield sse("sources", [{k: s[k] for k in ("url", "title", "section", "text", "score", "similarity")}
                              for s in sources])
        status = await ollama_status()
        model = req.model if req.model in status["models"] else status["default"]
        if not status["online"] or not model:
            yield sse("token", {"t": extractive_answer(sources)})
            yield sse("done", {"model": None, "fallback": True})
            return
        yield sse("meta", {"model": model})
        try:
            messages = build_messages(req.question, sources, req.history, kb.meta["url"])
            async for token in stream_answer(model, messages):
                yield sse("token", {"t": token})
            yield sse("done", {"model": model})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")
