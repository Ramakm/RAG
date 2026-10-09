"""Embeddings (sentence-transformers) + vector index (FAISS), persisted per knowledge base."""
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
from .config import EMBED_MODEL, KB_DIR

STOPWORDS = set("""a an and are as at be but by can do does for from has have how i in is it its me my
of on or our so that the their them there these this to was what when where which who why will with
you your about tell explain describe give list""".split())

OVERVIEW = re.compile(r"\b(summar\w*|overview|what is (this|it|the site|.+ about)|about this|purpose of|tl;?dr)\b", re.I)

_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def get_embedder():
    with _model_lock:
        return _load_model()


def embed(texts: list[str]) -> np.ndarray:
    vecs = get_embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


class KnowledgeBase:
    def __init__(self, kb_id: str, meta: dict, chunks: list[dict], index: faiss.Index):
        self.id, self.meta, self.chunks, self.index = kb_id, meta, chunks, index

    @property
    def path(self) -> Path:
        return KB_DIR / self.id

    @classmethod
    def build(cls, kb_id: str, meta: dict, chunks: list[dict], vectors: np.ndarray) -> "KnowledgeBase":
        index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity on normalized vectors
        index.add(vectors)
        kb = cls(kb_id, meta, chunks, index)
        kb.save()
        return kb

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        (self.path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False))
        (self.path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, kb_id: str) -> "KnowledgeBase":
        path = KB_DIR / kb_id
        meta = json.loads((path / "meta.json").read_text())
        chunks = json.loads((path / "chunks.json").read_text())
        return cls(kb_id, meta, chunks, faiss.read_index(str(path / "index.faiss")))

    def search(self, query: str, k: int = 5, per_page: int = 2) -> list[dict]:
        """Hybrid retrieval: dense cosine similarity + a light keyword-overlap boost,
        with a per-page cap so answers draw on diverse sources."""
        qv = embed([query])
        n = min(max(k * 6, 20), self.index.ntotal)
        scores, ids = self.index.search(qv, n)
        q_terms = _terms(query)
        ranked = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            overlap = len(q_terms & _terms(embedding_text(chunk))) / len(q_terms) if q_terms else 0.0
            ranked.append((0.8 * float(score) + 0.2 * overlap, float(score), chunk))
        ranked.sort(key=lambda r: r[0], reverse=True)

        results, per_url = [], {}
        if OVERVIEW.search(query) and self.meta.get("pages"):
            # Broad questions ("what is this site about?") are best answered by the start page's intro.
            home = self.meta["pages"][0]["url"]
            intro = next((c for c in self.chunks if c["url"] == home), None)
            if intro:
                results.append({**intro, "score": 1.0, "similarity": 1.0})
                per_url[home] = 1
                ranked = [r for r in ranked if r[2] is not intro]
        for hybrid, dense, chunk in ranked:
            if per_url.get(chunk["url"], 0) >= per_page:
                continue
            per_url[chunk["url"]] = per_url.get(chunk["url"], 0) + 1
            results.append({**chunk, "score": round(hybrid, 4), "similarity": round(dense, 4)})
            if len(results) == k:
                break
        return results


class KBManager:
    def __init__(self):
        self._cache: dict[str, KnowledgeBase] = {}

    def list(self) -> list[dict]:
        out = []
        for meta_file in KB_DIR.glob("*/meta.json"):
            meta = json.loads(meta_file.read_text())
            meta.pop("pages", None)
            out.append(meta)
        return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)

    def get(self, kb_id: str) -> KnowledgeBase:
        if kb_id not in self._cache:
            if not (KB_DIR / kb_id / "meta.json").exists():
                raise KeyError(kb_id)
            self._cache[kb_id] = KnowledgeBase.load(kb_id)
        return self._cache[kb_id]

    def put(self, kb: KnowledgeBase):
        self._cache[kb.id] = kb

    def delete(self, kb_id: str):
        self._cache.pop(kb_id, None)
        shutil.rmtree(KB_DIR / kb_id, ignore_errors=True)
