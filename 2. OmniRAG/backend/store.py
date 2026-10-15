"""Collections: two FAISS indexes per collection, persisted to disk.

- text index  (MiniLM, 384-d): document chunks, transcripts, and image captions/OCR
- visual index (CLIP, 512-d):  photos, figures and video frames, searchable by text or by image

Search fuses both with calibrated scores, so a query like "the slide with the revenue chart"
can match a figure visually even when no text mentions it.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from .chunker import embedding_text
from .config import KB_DIR, TEXT_EMBED_MODEL

TEXT_DIM, CLIP_DIM = 384, 512
STOPWORDS = set("""a an and are as at be but by can do does for from has have how i in is it its me my
of on or our so that the their them there these this to was what when where which who why will with
you your about tell explain describe give list show find any all""".split())
OVERVIEW = re.compile(r"\b(summar\w*|overview|what (is|are) (this|these|in|the) (collection|files?|documents?|about)|tl;?dr|key (points|takeaways))\b", re.I)
MODALITY_WORDS = {
    "video": re.compile(r"\b(video|clip|footage|scene|frame|movie|recording)\b", re.I),
    "audio": re.compile(r"\b(audio|podcast|recording|call|speech|said|says|speaker|mention(ed)?)\b", re.I),
    "image": re.compile(r"\b(image|images|photo|picture|figure|diagram|chart|graph|screenshot|logo|visual|looks?)\b", re.I),
    "document": re.compile(r"\b(pdf|document|doc|report|paper|slide|deck|page|table|section)\b", re.I),
}
VISUAL_KINDS = ("image", "figure", "frame")

_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_EMBED_MODEL)


def embed_text(texts: list[str]) -> np.ndarray:
    with _model_lock:
        model = _text_model()
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def _cal(score: float, lo: float, hi: float) -> float:
    """Map a raw cosine score from one embedding space onto a shared 0–1 relevance scale."""
    return float(min(1.0, max(0.0, (score - lo) / (hi - lo))))


class Collection:
    def __init__(self, cid: str, meta: dict, items: list[dict], text_index, text_ids, visual_index, visual_ids):
        self.id, self.meta, self.items = cid, meta, items
        self.text_index, self.text_ids = text_index, text_ids
        self.visual_index, self.visual_ids = visual_index, visual_ids
        self.lock = threading.Lock()

    @property
    def path(self) -> Path:
        return KB_DIR / self.id

    @classmethod
    def create(cls, cid: str, meta: dict) -> "Collection":
        c = cls(cid, meta, [], faiss.IndexFlatIP(TEXT_DIM), [], faiss.IndexFlatIP(CLIP_DIM), [])
        (c.path / "files").mkdir(parents=True, exist_ok=True)
        (c.path / "assets").mkdir(parents=True, exist_ok=True)
        return c

    @classmethod
    def load(cls, cid: str) -> "Collection":
        p = KB_DIR / cid
        meta = json.loads((p / "meta.json").read_text())
        state = json.loads((p / "items.json").read_text()) if (p / "items.json").exists() else {}
        ti = faiss.read_index(str(p / "text.faiss")) if (p / "text.faiss").exists() else faiss.IndexFlatIP(TEXT_DIM)
        vi = faiss.read_index(str(p / "visual.faiss")) if (p / "visual.faiss").exists() else faiss.IndexFlatIP(CLIP_DIM)
        return cls(cid, meta, state.get("items", []), ti, state.get("text_ids", []), vi, state.get("visual_ids", []))

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(self.path / "text.faiss"))
        faiss.write_index(self.visual_index, str(self.path / "visual.faiss"))
        state = {"items": self.items, "text_ids": self.text_ids, "visual_ids": self.visual_ids}
        (self.path / "items.json").write_text(json.dumps(state, ensure_ascii=False))
        (self.path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))

    def add(self, items: list[dict], text_vecs: np.ndarray, visual_vecs: np.ndarray, visual_items: list[dict]):
        """`items` get ids assigned here; every item with text goes into the text index,
        `visual_items` (a subset of `items`) into the visual index, aligned with `visual_vecs`."""
        with self.lock:
            base = len(self.items)
            for i, it in enumerate(items):
                it["id"] = base + i
            self.items.extend(items)
            text_items = [it for it in items if it["text"].strip()]
            if len(text_items):
                self.text_index.add(text_vecs)
                self.text_ids.extend(it["id"] for it in text_items)
            if len(visual_items):
                self.visual_index.add(visual_vecs)
                self.visual_ids.extend(it["id"] for it in visual_items)
            self.save()

    # ───────────── retrieval ─────────────
    def search(self, query: str, k: int = 6, image_vec: np.ndarray | None = None) -> list[dict]:
        from .vision import clip_text

        cands: dict[int, list[tuple[str, float]]] = {}

        def hit(item_id: int, via: str, rel: float):
            cands.setdefault(item_id, []).append((via, rel))

        q_terms = _terms(query)
        if query.strip() and self.text_index.ntotal:
            scores, ids = self.text_index.search(embed_text([query]), min(60, self.text_index.ntotal))
            for s, i in zip(scores[0], ids[0]):
                if i >= 0:
                    hit(self.text_ids[i], "text", _cal(float(s), 0.12, 0.72))
        if query.strip() and self.visual_index.ntotal:
            scores, ids = self.visual_index.search(clip_text([query]), min(40, self.visual_index.ntotal))
            for s, i in zip(scores[0], ids[0]):
                # Images containing text score ~0.25 against almost any query, so the floor sits above that.
                rel = _cal(float(s), 0.24, 0.34)
                if i >= 0 and rel >= 0.35:
                    hit(self.visual_ids[i], "visual", 0.85 * rel)
        if image_vec is not None and self.visual_index.ntotal:
            scores, ids = self.visual_index.search(image_vec, min(40, self.visual_index.ntotal))
            for s, i in zip(scores[0], ids[0]):
                rel = _cal(float(s), 0.55, 0.95)
                if i >= 0 and rel > 0.1:
                    hit(self.visual_ids[i], "image", rel * 1.1)

        # Hints in the question ("in the video", "the chart", "report.pdf") boost matching items.
        wanted = {m for m, rx in MODALITY_WORDS.items() if rx.search(query)}
        named = {f["id"] for f in self.meta.get("files", [])
                 if (_terms(Path(f["name"]).stem) & q_terms) or f["name"].lower() in query.lower()}

        ranked = []
        for item_id, hits in cands.items():
            it = self.items[item_id]
            rels = sorted((r for _, r in hits), reverse=True)
            if rels[0] < 0.12:  # boosts below can reorder relevant items, never rescue irrelevant ones
                continue
            score = rels[0] + 0.15 * sum(rels[1:])
            if q_terms and it["text"]:
                score += 0.12 * len(q_terms & _terms(embedding_text(it))) / len(q_terms)
            if it["modality"] in wanted or (it["kind"] in VISUAL_KINDS and "image" in wanted):
                score += 0.08
            if it["file_id"] in named:
                score += 0.15
            ranked.append((score, it, sorted({v for v, _ in hits})))
        ranked.sort(key=lambda r: r[0], reverse=True)

        results, per_file, seen = [], {}, set()
        multi = len(self.meta.get("files", [])) > 1
        cap = max(2, k // 2) if multi else k

        def take(it, score, via):
            results.append({**it, "score": round(min(score, 1.0), 4), "via": via})
            seen.add(it["id"])
            per_file[it["file_id"]] = per_file.get(it["file_id"], 0) + 1

        if OVERVIEW.search(query):
            # Broad questions: start with the opening of each file so every file is represented.
            for f in self.meta.get("files", [])[: k // 2 + 1]:
                first = next((it for it in self.items if it["file_id"] == f["id"] and it["text"]), None)
                if first:
                    take(first, 0.9, ["overview"])
        floor = 0.4 * ranked[0][0] if ranked else 0
        for score, it, via in ranked:
            if len(results) >= k or score < floor:
                break
            if it["id"] in seen or (multi and per_file.get(it["file_id"], 0) >= cap):
                continue
            take(it, score, via)
        return results


class CollectionManager:
    def __init__(self):
        self._cache: dict[str, Collection] = {}

    def list(self) -> list[dict]:
        out = []
        for meta_file in KB_DIR.glob("*/meta.json"):
            meta = json.loads(meta_file.read_text())
            meta.pop("files", None)
            out.append(meta)
        return sorted(out, key=lambda m: m.get("updated_at", ""), reverse=True)

    def get(self, cid: str) -> Collection:
        if cid not in self._cache:
            if not (KB_DIR / cid / "meta.json").exists():
                raise KeyError(cid)
            self._cache[cid] = Collection.load(cid)
        return self._cache[cid]

    def put(self, c: Collection):
        self._cache[c.id] = c

    def delete(self, cid: str):
        self._cache.pop(cid, None)
        shutil.rmtree(KB_DIR / cid, ignore_errors=True)
