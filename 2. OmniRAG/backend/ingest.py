"""The ingestion pipeline: Upload → Parse → Transcribe → Vision → Embed → Index.

Heavy work runs in a worker thread; every step emits progress events that the UI renders live.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .chunker import chunk_section, chunk_transcript, clock, embedding_text
from .config import MAX_CAPTIONS, MAX_FRAMES, TESSERACT
from .media import probe, sample_frames, transcribe
from .parsers import parse_document
from .store import CLIP_DIM, Collection, embed_text
from .vision import caption, clip_images, ocr

VISUAL_PRIORITY = {"image": 0, "figure": 1, "frame": 2}


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


def _item(f: dict, kind: str, text: str = "", **extra) -> dict:
    return {"file_id": f["id"], "file_name": f["name"], "modality": f["modality"], "kind": kind,
            "loc": "", "page": None, "start": None, "end": None, "asset": None,
            "src": f"files/{f['stored']}", "text": text.strip(), "caption": "", "ocr": "", **extra}


class Throttle:
    def __init__(self, every: float = 0.3):
        self.every, self.last = every, 0.0

    def ready(self) -> bool:
        now = time.time()
        if now - self.last >= self.every:
            self.last = now
            return True
        return False


def _pipeline(coll: Collection, files: list[dict], opts: dict, emit) -> dict:
    started = time.time()
    assets = coll.path / "assets"
    total_mb = sum(f["size"] for f in files) / 1e6
    emit("stage", stage="upload", status="done", detail=f"{len(files)} file{'s' * (len(files) != 1)} · {total_mb:.1f} MB")

    # ───── 1. Parse ─────
    emit("stage", stage="parse", status="running", detail="Reading files")
    items: list[dict] = []
    visuals: list[dict] = []        # candidate visual items, embedded with CLIP in the Vision stage
    scans: list[tuple[dict, dict]] = []
    media: list[dict] = []
    for n, f in enumerate(files, 1):
        path = coll.path / "files" / f["stored"]
        info: dict = {}
        try:
            if f["modality"] == "document":
                res = parse_document(path, assets, f["id"])
                for sec in res["sections"]:
                    items += [_item(f, "text", ch["text"], loc=ch["loc"], page=ch["page"], asset=ch["asset"])
                              for ch in chunk_section(sec)]
                visuals += [_item(f, "figure", loc=im["loc"], page=im["page"], asset=f"assets/{im['path'].name}",
                                  _path=im["path"]) for im in res["images"]]
                scans += [(f, s) for s in res["ocr"]]
                info = {**res["info"], "figures": len(res["images"]), "scanned_pages": len(res["ocr"])}
                n_pages = info.get("pages") or info.get("slides") or 0
                unit = "slide" if "slides" in info else "page"
                bits = [f"{n_pages} {unit}{'s' * (n_pages != 1)}" if n_pages else "",
                        f"{info['figures']} figure{'s' * (info['figures'] != 1)}" if info["figures"] else "",
                        f"{info['scanned_pages']} scanned" if info["scanned_pages"] else ""]
                detail = " · ".join(b for b in bits if b) or "text"
                if info.get("repeated_lines_removed"):
                    emit("log", level="info", message=f"{f['name']}: removed {info['repeated_lines_removed']} "
                                                      "repeated header/footer line(s)")
            elif f["modality"] == "image":
                img = Image.open(path)
                info = {"width": img.width, "height": img.height}
                img = img.convert("RGB")
                img.thumbnail((1280, 1280))
                thumb = assets / f"{f['id']}.jpg"
                img.save(thumb, "JPEG", quality=86)
                visuals.append(_item(f, "image", asset=f"assets/{thumb.name}", _path=thumb))
                detail = f"{info['width']}×{info['height']}"
            else:
                info = probe(path)
                detail = clock(info["duration"])
                if f["modality"] == "video" and info["has_video"]:
                    frames = sample_frames(path, assets, f["id"], opts["frame_interval"], MAX_FRAMES, info["duration"])
                    visuals += [_item(f, "frame", loc=clock(fr["t"]), start=fr["t"], asset=f"assets/{fr['path'].name}",
                                      _path=fr["path"]) for fr in frames]
                    info["frames_sampled"] = len(frames)
                    detail += f" · {len(frames)} frames"
                if info["has_audio"]:
                    media.append(f)
                else:
                    detail += " · no audio"
        except Exception as e:  # one bad file shouldn't sink the whole build
            f["error"] = str(e)[:200]
            emit("log", level="warn", message=f"Couldn't read {f['name']}: {f['error']}")
            detail = "failed"
        f["info"] = info
        emit("file", name=f["name"], modality=f["modality"], detail=detail, count=n, total=len(files),
             duration=info.get("duration", 0) if f["modality"] in ("audio", "video") else 0,
             error=bool(f.get("error")))
        emit("stage", stage="parse", status="running", detail=f"{n} / {len(files)} files")
    emit("stage", stage="parse", status="done",
         detail=f"{len(items)} text chunks · {len(visuals)} visuals")

    # ───── 2. Transcribe ─────
    if media:
        emit("stage", stage="transcribe", status="running", detail=f"Loading Whisper {opts['whisper']}")
        for f in media:
            th = Throttle()

            def progress(p, name=f["name"]):
                if th.ready():
                    emit("stage", stage="transcribe", status="running", detail=f"{name[:22]} · {int(p * 100)}%",
                         progress=p)
            try:
                segs, lang = transcribe(coll.path / "files" / f["stored"], opts["whisper"], progress)
            except Exception as e:
                emit("log", level="warn", message=f"Couldn't transcribe {f['name']}: {str(e)[:200]}")
                continue
            frames = [v for v in visuals if v["file_id"] == f["id"]]
            for ch in chunk_transcript(segs):
                near = min(frames, key=lambda v: abs(v["start"] - ch["start"]), default=None)
                items.append(_item(f, "transcript", ch["text"], loc=ch["loc"], start=ch["start"], end=ch["end"],
                                   asset=near["asset"] if near else None))
            words = sum(len(s["text"].split()) for s in segs)
            f["info"].update(language=lang, words=words)
            emit("log", level="info", message=f"Transcribed {f['name']}: {words:,} words"
                                              + (f" ({lang})" if lang else "") + ", silence skipped")
        speech = sum(f["info"].get("duration", 0) for f in media)
        emit("stage", stage="transcribe", status="done", detail=f"{clock(speech)} of audio")
    else:
        emit("stage", stage="transcribe", status="skipped", detail="No audio or video")

    # ───── 3. Vision ─────
    vis_vecs = np.zeros((0, CLIP_DIM), dtype="float32")
    stats = {"duplicates": 0, "ocr": 0, "captions": 0}
    if visuals or scans:
        if visuals:
            emit("stage", stage="vision", status="running", detail=f"CLIP · {len(visuals)} visuals")
            vecs = clip_images([v["_path"] for v in visuals])
            # Drop near-duplicates: static video shots and logos repeated across slides/pages.
            kept, kept_vecs, last_frame = [], [], {}
            for v, vec in zip(visuals, vecs):
                if v["kind"] == "frame":
                    prev = last_frame.get(v["file_id"])
                    dup = prev is not None and float(prev @ vec) > 0.95
                else:
                    dup = any(k["file_id"] == v["file_id"] and k["kind"] != "frame" and float(kv @ vec) > 0.97
                              for k, kv in zip(kept, kept_vecs))
                if dup:
                    stats["duplicates"] += 1
                    continue
                if v["kind"] == "frame":
                    last_frame[v["file_id"]] = vec
                kept.append(v)
                kept_vecs.append(vec)
            visuals = kept
            vis_vecs = np.asarray(kept_vecs, dtype="float32").reshape(-1, CLIP_DIM)
            if stats["duplicates"]:
                emit("log", level="info", message=f"Dropped {stats['duplicates']} near-duplicate visual(s)")

        if opts["ocr"] and TESSERACT:
            th = Throttle()
            for n, v in enumerate(visuals, 1):
                v["ocr"] = ocr(v["_path"])
                stats["ocr"] += bool(v["ocr"])
                if th.ready():
                    emit("stage", stage="vision", status="running", detail=f"OCR · {n} / {len(visuals)}")
            for n, (f, scan) in enumerate(scans, 1):
                emit("stage", stage="vision", status="running", detail=f"OCR scanned page {n} / {len(scans)}")
                text = ocr(scan["path"], min_conf=50)
                scan["path"].unlink(missing_ok=True)
                if text:
                    sec = {"loc": scan["loc"], "page": scan["page"], "text": text, "asset": scan["asset"]}
                    items += [_item(f, "text", ch["text"], loc=ch["loc"], page=ch["page"], asset=ch["asset"])
                              for ch in chunk_section(sec)]
                    stats["ocr"] += 1
        elif scans or visuals:
            emit("log", level="warn", message="OCR skipped" + ("" if opts["ocr"] else " (turned off)") +
                 ("" if TESSERACT else ": install tesseract to read text inside images and scanned pages"))

        vmodel = opts.get("vision_model")
        if visuals and opts["captions"] and vmodel:
            order = sorted(range(len(visuals)), key=lambda i: VISUAL_PRIORITY[visuals[i]["kind"]])[:MAX_CAPTIONS]
            for n, i in enumerate(order, 1):
                emit("stage", stage="vision", status="running", detail=f"Describing {n} / {len(order)}",
                     progress=n / len(order))
                try:
                    visuals[i]["caption"] = caption(vmodel, visuals[i]["_path"])
                    stats["captions"] += 1
                except Exception as e:
                    emit("log", level="warn", message=f"Vision model failed on {visuals[i]['file_name']}: {str(e)[:120]}")
                    break
            emit("log", level="info", message=f"{vmodel} described {stats['captions']} visual(s)")
        elif visuals and opts["captions"]:
            emit("log", level="info", message="No vision model in Ollama, so images are matched visually (CLIP) "
                                              "and by OCR. Run `ollama pull moondream` to add descriptions.")
        for v in visuals:
            v["text"] = "\n".join(x for x in (v["caption"], v["ocr"]) if x)
            v.pop("_path", None)
        emit("stage", stage="vision", status="done",
             detail=" · ".join(x for x in (f"{len(visuals)} visuals", f"{stats['ocr']} OCR" if stats["ocr"] else "",
                                           f"{stats['captions']} described" if stats["captions"] else "") if x))
    else:
        emit("stage", stage="vision", status="skipped", detail="No images")
    emit("stats", text_chunks=len(items), visuals=len(visuals), duplicates=stats["duplicates"])

    # ───── 4. Embed ─────
    all_items = items + visuals
    text_items = [it for it in all_items if it["text"].strip()]
    if not all_items:
        raise RuntimeError("No content could be extracted from these files.")
    emit("stage", stage="embed", status="running", detail=f"0 / {len(text_items)} chunks")
    vecs = []
    for i in range(0, len(text_items), 64):
        vecs.append(embed_text([embedding_text(it) for it in text_items[i:i + 64]]))
        done = min(i + 64, len(text_items))
        emit("stage", stage="embed", status="running", detail=f"{done} / {len(text_items)} chunks",
             progress=done / len(text_items))
    text_vecs = np.vstack(vecs) if vecs else np.zeros((0, 384), dtype="float32")
    emit("stage", stage="embed", status="done", detail=f"{len(text_items)} text + {len(visuals)} visual vectors")

    # ───── 5. Index ─────
    emit("stage", stage="index", status="running", detail="Writing FAISS indexes")
    for f in files:
        f["items"] = sum(1 for it in all_items if it["file_id"] == f["id"])
    coll.meta["files"].extend(files)
    now = datetime.now(timezone.utc).isoformat()
    coll.meta["updated_at"] = now
    coll.add(all_items, text_vecs, vis_vecs, visuals)
    m = coll.meta
    m["counts"] = {
        "files": len(m["files"]),
        "chunks": sum(it["kind"] in ("text", "transcript") for it in coll.items),
        "visuals": coll.visual_index.ntotal,
        "media_seconds": round(sum(f.get("info", {}).get("duration", 0) for f in m["files"]), 1),
        "by_modality": {mod: sum(f["modality"] == mod for f in m["files"]) for mod in ("document", "image", "audio", "video")},
    }
    coll.save()
    emit("stage", stage="index", status="done", detail=f"{coll.text_index.ntotal + coll.visual_index.ntotal} vectors")
    return {**{k: v for k, v in m.items() if k != "files"}, "duration_s": round(time.time() - started, 1),
            "added": len(files), "added_items": len(all_items)}


async def run_ingest(job: Job, coll: Collection, files: list[dict], opts: dict):
    loop = asyncio.get_running_loop()

    def emit(type: str, **data):
        asyncio.run_coroutine_threadsafe(job.emit(type, **data), loop)

    try:
        summary = await asyncio.to_thread(_pipeline, coll, files, opts, emit)
        await asyncio.sleep(0.05)  # let queued progress events land before the final one
        await job.emit("complete", kb=summary)
    except Exception as e:
        await asyncio.sleep(0.05)
        await job.emit("error", message=str(e) or e.__class__.__name__)
