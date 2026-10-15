"""OmniRAG API server + static UI.

Run from the OmniRAG folder:  uvicorn backend.app:app --port 8001
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (CLIP_MODEL, FRONTEND_DIR, MAX_UPLOAD_MB, TESSERACT, TEXT_EMBED_MODEL, WHISPER_MODEL,
                     WHISPER_SIZES, modality_of)
from .ingest import Job, run_ingest
from .llm import build_messages, extractive_answer, ollama_status, stream_answer
from .store import Collection, CollectionManager, VISUAL_KINDS
from .vision import caption, clip_image_bytes, image_b64

app = FastAPI(title="OmniRAG", version="1.0")
colls = CollectionManager()
jobs: dict[str, Job] = {}
_tasks: set[asyncio.Task] = set()
KB_ID = re.compile(r"^[a-z0-9-]{1,80}$")


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_coll(kb_id: str) -> Collection:
    if not KB_ID.match(kb_id):
        raise HTTPException(400, "Invalid collection id")
    try:
        return colls.get(kb_id)
    except KeyError:
        raise HTTPException(404, "Collection not found")


class ChatRequest(BaseModel):
    question: str = Field("", max_length=2000)
    history: list[dict] = []
    model: str | None = None
    top_k: int = Field(6, ge=1, le=12)
    image: str | None = Field(None, max_length=15_000_000)  # data URL or base64


@app.get("/api/health")
async def health():
    return {"ollama": await ollama_status(), "embed_model": TEXT_EMBED_MODEL, "clip_model": CLIP_MODEL,
            "whisper_model": WHISPER_MODEL, "ocr": bool(TESSERACT)}


@app.get("/api/kbs")
def list_kbs():
    return colls.list()


@app.get("/api/kbs/{kb_id}")
def kb_detail(kb_id: str):
    return get_coll(kb_id).meta


@app.delete("/api/kbs/{kb_id}")
def delete_kb(kb_id: str):
    get_coll(kb_id)
    colls.delete(kb_id)
    return {"ok": True}


@app.get("/api/kbs/{kb_id}/raw/{path:path}")
def raw_file(kb_id: str, path: str):
    """Original uploads (files/) and generated thumbnails/frames (assets/). Supports range requests."""
    coll = get_coll(kb_id)
    root = coll.path.resolve()
    target = (root / path).resolve()
    rel = target.relative_to(root).parts if target.is_relative_to(root) else ()
    if len(rel) < 2 or rel[0] not in ("files", "assets") or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(target)


@app.post("/api/ingest")
async def ingest(files: list[UploadFile] = File(...), kb_id: str = Form(""), name: str = Form(""),
                 options: str = Form("{}")):
    try:
        raw = json.loads(options or "{}")
    except ValueError:
        raw = {}
    status = await ollama_status()
    opts = {
        "whisper": raw.get("whisper") if raw.get("whisper") in WHISPER_SIZES else WHISPER_MODEL,
        "frame_interval": float(min(120, max(2, raw.get("frame_interval", 15)))),
        "ocr": bool(raw.get("ocr", True)),
        "captions": bool(raw.get("captions", True)),
        "vision_model": status["default_vision"],
    }

    now = datetime.now(timezone.utc).isoformat()
    if kb_id:
        coll, is_new = get_coll(kb_id), False
    else:
        title = name.strip() or Path(files[0].filename or "Collection").stem
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower())[:40].strip("-") or "collection"
        cid = f"{slug}-{uuid.uuid4().hex[:6]}"
        coll = Collection.create(cid, {"id": cid, "name": title[:80], "created_at": now, "updated_at": now,
                                       "files": [], "counts": {}})
        is_new = True

    known = {f["sha256"] for f in coll.meta["files"]}
    saved, skipped = [], []
    for up in files:
        fname = Path(up.filename or "file").name
        modality = modality_of(fname)
        if not modality:
            skipped.append(f"{fname} (unsupported type)")
            continue
        fid = uuid.uuid4().hex[:10]
        stored = f"{fid}{Path(fname).suffix.lower()}"
        dest = coll.path / "files" / stored
        h, size = hashlib.sha256(), 0
        with dest.open("wb") as out:
            while chunk := await up.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * (1 << 20):
                    break
                h.update(chunk)
                out.write(chunk)
        digest = h.hexdigest()
        if size > MAX_UPLOAD_MB * (1 << 20):
            dest.unlink(missing_ok=True)
            skipped.append(f"{fname} (over {MAX_UPLOAD_MB} MB)")
        elif digest in known:
            dest.unlink(missing_ok=True)
            skipped.append(f"{fname} (already in this collection)")
        else:
            known.add(digest)
            saved.append({"id": fid, "name": fname, "stored": stored, "modality": modality, "size": size,
                          "sha256": digest, "added_at": now})

    if not saved:
        if is_new:
            shutil.rmtree(coll.path, ignore_errors=True)
        raise HTTPException(400, "Nothing to index: " + ("; ".join(skipped) or "no files received"))

    job = Job(kb_id=coll.id)
    jobs[job.id] = job
    for s in skipped:
        await job.emit("log", level="warn", message=f"Skipped {s}")

    async def run():
        await run_ingest(job, coll, saved, opts)
        if job.events[-1]["type"] == "complete":
            colls.put(coll)
        elif is_new:
            shutil.rmtree(coll.path, ignore_errors=True)

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job_id": job.id, "kb_id": coll.id, "files": len(saved), "skipped": skipped}


@app.get("/api/ingest/{job_id}/events")
async def ingest_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def gen():
        async for ev in job.listen():
            yield sse(ev["type"], ev)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def _public(s: dict) -> dict:
    keys = ("id", "file_id", "file_name", "modality", "kind", "loc", "page", "start", "end", "asset", "src",
            "text", "caption", "ocr", "score", "via")
    return {k: s.get(k) for k in keys}


@app.post("/api/kbs/{kb_id}/chat")
async def chat(kb_id: str, req: ChatRequest):
    coll = get_coll(kb_id)
    image_bytes = None
    if req.image:
        try:
            image_bytes = base64.b64decode(req.image.split(",", 1)[-1])
        except ValueError:
            raise HTTPException(400, "Couldn't read the attached image")
    if not req.question.strip() and not image_bytes:
        raise HTTPException(400, "Ask a question or attach an image")
    question = req.question.strip() or "What is this image, and where does it appear in the collection?"

    status = await ollama_status()
    model = req.model if req.model in status["models"] else status["default"]
    can_see = bool(model and model in status["vision_models"])

    # Short follow-ups ("and the price?") are retrieved together with the previous question.
    query = req.question.strip()
    prev_q = next((t["content"] for t in reversed(req.history) if t.get("role") == "user"), "")
    if prev_q and query and len(query.split()) < 8:
        query = f"{prev_q} {query}"

    async def gen():
        image_vec, query_text = None, query
        if image_bytes:
            try:
                image_vec = await asyncio.to_thread(clip_image_bytes, image_bytes)
            except Exception:
                yield sse("error", {"message": "That attachment isn't a readable image."})
                return
            # Without a vision chat model, describe the image with any vision model so text search can use it.
            if status["default_vision"] and not can_see:
                yield sse("status", {"text": "Looking at your image…"})
                try:
                    desc = await asyncio.to_thread(caption, status["default_vision"], image_bytes)
                    query_text = f"{query} {desc}".strip()
                    yield sse("image_caption", {"text": desc})
                except Exception:
                    pass
        sources = await asyncio.to_thread(coll.search, query_text, req.top_k, image_vec)
        yield sse("sources", [_public(s) for s in sources])

        if not status["online"] or not model:
            yield sse("token", {"t": extractive_answer(sources)})
            yield sse("done", {"model": None, "fallback": True})
            return
        yield sse("meta", {"model": model, "vision": can_see})

        attached: list[tuple[str, str]] = []
        if can_see:
            if image_bytes:
                attached.append(("user", image_b64(image_bytes)))
            for i, s in enumerate(sources, 1):
                if s["kind"] in VISUAL_KINDS and s.get("asset") and len(attached) < 4:
                    attached.append((f"[{i}]", image_b64(coll.path / s["asset"])))
        try:
            messages = build_messages(question, sources, req.history, coll.meta["name"], attached, bool(image_bytes))
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
