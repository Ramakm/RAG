"""Shared helpers for the RAG Fundamentals notebooks.

Everything runs locally: sentence-transformers for embeddings, Ollama for the LLM
and plain numpy for the vector store, so every notebook stays readable.
"""
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LLM_MODEL = os.getenv("RAG_LLM", "llama3.2")
VISION_MODEL = os.getenv("RAG_VISION", "llava")
EMBED_MODEL = os.getenv("RAG_EMBED", "all-MiniLM-L6-v2")

_embedder = None


# ---------------------------------------------------------------- models
def embed(texts):
    """Return L2-normalised embeddings, shape (n, dim)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    if isinstance(texts, str):
        texts = [texts]
    return np.asarray(_embedder.encode(list(texts), normalize_embeddings=True), dtype="float32")


def llm(prompt, system="You are a helpful assistant.", temperature=0.0):
    import ollama

    reply = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return reply["message"]["content"].strip()


def llm_json(prompt, system="Reply with valid JSON only."):
    """Call the LLM and parse the first JSON object/array in its reply (None on failure)."""
    text = llm(prompt, system)
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def llm_yes(prompt):
    """Ask a yes/no question and return True for 'yes'."""
    return llm(prompt + "\nAnswer with exactly one word: yes or no.").lower().startswith("yes")


# ---------------------------------------------------------------- data
def load_text(name="ai_handbook.txt"):
    return (DATA / name).read_text()


def load_eval_questions():
    return json.loads((DATA / "eval_questions.json").read_text())


def chunk_text(text, size=500, overlap=50):
    """Fixed-size character chunks that never cut a word in half."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            end = space if space > start else end
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_sections(text):
    """Split a markdown document on '## ' headings -> list of (heading, body)."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    sections = []
    for part in parts:
        heading, _, body = part.partition("\n")
        sections.append((heading.strip(), body.strip()))
    return sections


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------- vector store
class VectorStore:
    """Tiny in-memory cosine-similarity store."""

    def __init__(self):
        self.texts, self.meta = [], []
        self.vecs = np.zeros((0, 1), dtype="float32")

    def add(self, texts, meta=None):
        texts = list(texts)
        vecs = embed(texts)
        self.vecs = vecs if len(self.texts) == 0 else np.vstack([self.vecs, vecs])
        self.texts += texts
        self.meta += list(meta) if meta else [{} for _ in texts]
        return self

    def search(self, query, k=4, mask=None):
        scores = self.vecs @ embed(query)[0]
        if mask is not None:
            scores = np.where(mask, scores, -1.0)
        top = np.argsort(-scores)[:k]
        return [
            {"idx": int(i), "text": self.texts[i], "score": float(scores[i]), "meta": self.meta[i]}
            for i in top
            if scores[i] > -1.0
        ]

    def __len__(self):
        return len(self.texts)


def build_store(chunks, meta=None):
    return VectorStore().add(chunks, meta)


# ---------------------------------------------------------------- generation
RAG_SYSTEM = (
    "Answer the question using only the provided context. "
    "If the context does not contain the answer, say you don't know."
)


def answer(query, contexts):
    """Generate an answer from a list of context strings."""
    context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return llm(f"Context:\n{context}\n\nQuestion: {query}", system=RAG_SYSTEM)


def rag(query, store, k=4):
    """Plain retrieve-then-generate. Returns (answer, retrieved hits)."""
    hits = store.search(query, k)
    return answer(query, [h["text"] for h in hits]), hits


def show(hits, width=160):
    for h in hits:
        print(f"{h['score']:.3f}  {h['text'][:width].replace(chr(10), ' ')}...")


def hit_at_k(hits, evidence):
    """True if any retrieved text contains the evidence phrase."""
    return any(evidence.lower() in h["text"].lower() for h in hits)


# ---------------------------------------------------------------- knowledge graph
def extract_triples(chunk):
    """LLM entity/relation extraction -> list of (subject, relation, object)."""
    data = llm_json(
        "Extract knowledge-graph triples from the text. Reply with a JSON list of "
        '[subject, relation, object] arrays, using short lowercase noun phrases.\n\nText: ' + chunk
    )
    triples = []
    for t in data or []:
        if isinstance(t, list) and len(t) == 3 and all(isinstance(x, str) for x in t):
            triples.append(tuple(x.strip().lower() for x in t))
    return triples


# ---------------------------------------------------------------- multimodal samples
def make_figures():
    """Render three small chart images used by the multimodal notebooks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = DATA / "figures"
    out.mkdir(exist_ok=True)
    charts = {
        "chunk_size_vs_hit_rate.png": (
            "Hit rate by chunk size", ["200", "400", "800", "1600"], [0.62, 0.78, 0.74, 0.55], "Hit rate"
        ),
        "latency_breakdown.png": (
            "Latency per pipeline stage (ms)", ["Embed", "Search", "Rerank", "Generate"], [20, 15, 120, 900], "ms"
        ),
        "retrieval_methods_recall.png": (
            "Recall@5 by retrieval method", ["Dense", "BM25", "Hybrid", "Hybrid+Rerank"], [0.71, 0.64, 0.82, 0.90], "Recall@5"
        ),
    }
    for name, (title, labels, values, ylabel) in charts.items():
        path = out / name
        if path.exists():
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return sorted(out.glob("*.png"))
