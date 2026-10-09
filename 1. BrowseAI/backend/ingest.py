"""The ingestion pipeline: Crawl → Extract → Clean → Chunk → Embed → Index.

Each step emits progress events that the UI renders live.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np

from .chunker import chunk_page, embedding_text
from .cleaner import clean_pages
from .config import EMBED_MODEL
from .crawler import Crawler
from .store import KBManager, KnowledgeBase, embed, get_embedder


class Job:
    """An ingestion run whose events can be replayed and streamed to any number of listeners."""

    def __init__(self, kb_id: str):
        self.id = uuid.uuid4().hex[:12]
        self.kb_id = kb_id
        self.events: list[dict] = []
        self.done = False
        self._cond = asyncio.Condition()

    async def emit(self, type: str, **data):
        async with self._cond:
            self.events.append({"type": type, "ts": time.time(), **data})
            if type in ("complete", "error"):
                self.done = True
            self._cond.notify_all()

    async def listen(self):
        i = 0
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda: len(self.events) > i or self.done)
                batch, finished = self.events[i:], self.done
            for ev in batch:
                yield ev
            i += len(batch)
            if finished and i >= len(self.events):
                return


async def run_ingest(job: Job, kbs: KBManager, url: str, opts: dict):
    emit = job.emit
    started = time.time()
    try:
        # 1+2. Crawl & extract (extraction happens per page as it's fetched)
        await emit("stage", stage="crawl", status="running", detail="Discovering pages")
        await emit("stage", stage="extract", status="running", detail="Pulling main content")
        crawler = Crawler(url, max_pages=opts["max_pages"], max_depth=opts["max_depth"],
                          same_path=opts["same_path"], js_mode=opts["js_mode"], emit=emit)
        pages = await crawler.run()
        if not pages:
            raise RuntimeError("Couldn't fetch any pages. Check the URL is reachable and public.")
        rendered = sum(p.rendered for p in pages)
        await emit("stage", stage="crawl", status="done",
                   detail=f"{len(pages)} pages fetched" + (f", {crawler.failed} skipped" if crawler.failed else ""))
        await emit("stage", stage="extract", status="done",
                   detail=f"{rendered} rendered with JS" if rendered else "Static HTML")

        # 3. Clean
        await emit("stage", stage="clean", status="running", detail="Removing boilerplate")
        raw = [{"url": p.url, "title": p.title, "text": p.text} for p in pages]
        cleaned, stats = await asyncio.to_thread(clean_pages, raw)
        if not cleaned:
            raise RuntimeError("Pages were fetched but no readable content survived cleaning. "
                               "If the site is JavaScript-heavy, try JS rendering 'Always'.")
        removed = stats["boilerplate_blocks"] + stats["junk_blocks"] + stats["duplicate_blocks"]
        await emit("stats", **stats)
        await emit("stage", stage="clean", status="done",
                   detail=f"{removed} noisy blocks removed, {len(cleaned)} pages kept")

        # 4. Chunk
        await emit("stage", stage="chunk", status="running", detail="Splitting by section")
        chunks = [c for page in cleaned for c in chunk_page(page)]
        for i, c in enumerate(chunks):
            c["id"] = i
        await emit("stage", stage="chunk", status="done", detail=f"{len(chunks)} chunks")

        # 5. Embed
        await emit("stage", stage="embed", status="running", detail="Loading embedding model")
        await asyncio.to_thread(get_embedder)
        texts = [embedding_text(c) for c in chunks]
        vectors, batch = [], 64
        for i in range(0, len(texts), batch):
            vectors.append(await asyncio.to_thread(embed, texts[i:i + batch]))
            done = min(i + batch, len(texts))
            await emit("stage", stage="embed", status="running", detail=f"{done}/{len(texts)} vectors",
                       progress=done / len(texts))
        matrix = np.vstack(vectors)
        await emit("stage", stage="embed", status="done", detail=f"{matrix.shape[1]}-dim vectors")

        # 6. Index
        await emit("stage", stage="index", status="running", detail="Building FAISS index")
        host = urlparse(crawler.start_url).netloc.removeprefix("www.")
        meta = {
            "id": job.kb_id,
            "name": host,
            "url": crawler.start_url,
            "home_title": cleaned[0]["title"] if cleaned else host,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "page_count": len(cleaned),
            "chunk_count": len(chunks),
            "word_count": stats["words_after"],
            "embed_model": EMBED_MODEL,
            "settings": opts,
            "stats": stats,
            "duration_s": round(time.time() - started, 1),
            "pages": [{"url": p["url"], "title": p["title"], "words": p["words"]} for p in cleaned],
        }
        kb = await asyncio.to_thread(KnowledgeBase.build, job.kb_id, meta, chunks, matrix)
        kbs.put(kb)
        await emit("stage", stage="index", status="done", detail=f"{len(chunks)} vectors indexed")
        summary = {k: v for k, v in meta.items() if k != "pages"}
        await emit("complete", kb=summary)
    except Exception as e:
        await emit("error", message=str(e) or e.__class__.__name__)
