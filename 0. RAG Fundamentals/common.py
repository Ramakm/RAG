"""Shared helpers for the RAG Fundamentals notebooks.

Everything runs locally: sentence-transformers for embeddings, Ollama for the LLM
and plain numpy for the vector store, so every notebook stays readable.
"""
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LLM_MODEL = os.getenv("RAG_LLM", "llama3.2")
VISION_MODEL = os.getenv("RAG_VISION", "moondream")
EMBED_MODEL = os.getenv("RAG_EMBED", "all-MiniLM-L6-v2")

_embedder = None


# ---------------------------------------------------------------- models
def embed(texts):
    """Return L2-normalised embeddings, shape (n, dim)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        # tiny model: run on CPU so the GPU memory stays free for Ollama
        _embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    if isinstance(texts, str):
        texts = [texts]
    return np.asarray(_embedder.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype="float32")


CACHE_DIR = ROOT / ".cache"
_llm_cache = None


def _cache_path():
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / "llm_cache.json"


def llm(prompt, system="You are a helpful assistant.", temperature=0.0, use_cache=True, schema=None):
    """Chat with the local Ollama model. Deterministic calls (temperature 0) are cached on disk,
    so re-running a notebook is instant and gives identical output. Delete `.cache/` to reset."""
    global _llm_cache
    import hashlib

    import ollama

    key = hashlib.sha256(f"{LLM_MODEL}|{system}|{prompt}|{temperature}|{schema}".encode()).hexdigest()
    cacheable = use_cache and temperature == 0.0
    if cacheable:
        if _llm_cache is None:
            path = _cache_path()
            _llm_cache = json.loads(path.read_text()) if path.exists() else {}
        if key in _llm_cache:
            return _llm_cache[key]
    reply = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        options={"temperature": temperature, "seed": 7},
        format=schema,   # a JSON schema forces valid structured output
    )
    text = reply["message"]["content"].strip()
    if cacheable:
        _llm_cache[key] = text
        _cache_path().write_text(json.dumps(_llm_cache))
    return text


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
    """Ask a yes/no question and return True for 'yes'.

    Small models are much more reliable when they give a one-sentence reason first, so the
    verdict is parsed from a final 'Verdict: yes/no' line.
    """
    out = llm(prompt + "\nGive a one-sentence reason, then finish with a final line exactly like "
                       "'Verdict: yes' or 'Verdict: no'.")
    found = re.findall(r"verdict:\s*\**\s*(yes|no)", out.lower())
    if found:
        return found[-1] == "yes"
    return out.lower().strip().startswith("yes")


def llm_score(prompt, lo=1, hi=5, temperature=0.0, use_cache=True):
    """Ask for an integer score in [lo, hi] (reason first, then a final 'Score: N' line)."""
    out = llm(prompt + f"\nGive a one-sentence reason, then finish with a final line exactly like "
                       f"'Score: N' where N is an integer from {lo} to {hi}.", temperature=temperature, use_cache=use_cache)
    found = re.findall(r"score:\s*\**\s*(\d+)", out.lower()) or re.findall(r"\b(\d+)\b", out)
    return min(max(int(found[-1]), lo), hi) if found else None


def faithfulness(response, contexts):
    """Fraction of the answer's sentences that the retrieved context supports (claim-level check)."""
    ctx = "\n".join(contexts)
    claims = split_sentences(response)
    if not claims:
        return 0.0
    ok = [llm_yes(f"Context:\n{ctx}\n\nClaim: {c}\nIs the claim supported by the context?") for c in claims]
    return round(sum(ok) / len(ok), 2)


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


def evidence_rank(hits, evidence):
    """1-based rank of the first hit containing the evidence phrase, or None."""
    for i, h in enumerate(hits, 1):
        if evidence.lower() in h["text"].lower():
            return i
    return None


def evaluate_retriever(search_fn, questions=None, k=3):
    """Score a retriever on the golden set. `search_fn(question, k)` must return hit dicts.

    Returns hit rate@k, MRR and the per-question ranks (None = evidence not retrieved).
    """
    questions = questions or load_eval_questions()
    ranks = [evidence_rank(search_fn(q["question"], k), q["evidence"]) for q in questions]
    return {
        "hit_rate@1": round(sum(r == 1 for r in ranks) / len(ranks), 2),
        f"hit_rate@{k}": round(sum(r is not None for r in ranks) / len(ranks), 2),
        "MRR": round(sum(1 / r for r in ranks if r) / len(ranks), 2),
        "ranks": ranks,
    }


def compare_retrievers(methods, questions=None, k=3):
    """`methods` maps a name to a search_fn(question, k). Returns hit@1, hit@k and MRR per method."""
    import pandas as pd

    rows = []
    for name, fn in methods.items():
        r = evaluate_retriever(fn, questions, k)
        rows.append({"method": name, "hit@1": r["hit_rate@1"], f"hit@{k}": r[f"hit_rate@{k}"], "MRR": r["MRR"]})
    return pd.DataFrame(rows).set_index("method")


def keyword_recall(reference, candidate):
    """Fraction of the reference answer's content words/numbers that appear in the candidate."""
    stop = {"the", "and", "that", "with", "from", "this", "which", "their", "for", "are", "its", "into"}
    words = [w for w in re.findall(r"[a-z0-9.\-]+", reference.lower()) if len(w) > 3 or w.isdigit()]
    words = [w.strip(".-") for w in words if w not in stop]
    if not words:
        return 1.0
    cand = candidate.lower()
    return round(sum(w in cand for w in words) / len(words), 2)


def compare_pipelines(pipelines, questions=None, n=5):
    """Run end-to-end pipelines (`name -> fn(question) -> answer`) on the first `n` golden questions.

    Reports keyword recall (deterministic), an LLM-judge verdict and average latency per question.
    """
    import time

    import pandas as pd

    questions = (questions or load_eval_questions())[:n]
    rows = []
    for name, fn in pipelines.items():
        recalls, judged, times = [], [], []
        for q in questions:
            t0 = time.time()
            try:
                response = fn(q["question"])
            except Exception as e:  # keep the comparison going if one pipeline fails
                response = f"ERROR: {e}"
            times.append(time.time() - t0)
            recalls.append(keyword_recall(q["answer"], response))
            judged.append(llm_yes(
                f"Reference answer: {q['answer']}\nCandidate answer: {response}\n"
                "Does the candidate give the same key information as the reference?"))
        rows.append({"pipeline": name, "keyword_recall": round(sum(recalls) / len(recalls), 2),
                     "judge_correct": round(sum(judged) / len(judged), 2),
                     "avg_seconds": round(sum(times) / len(times), 1)})
    return pd.DataFrame(rows).set_index("pipeline")


# ---------------------------------------------------------------- knowledge graph
TRIPLE_SCHEMA = {
    "type": "object",
    "properties": {"triples": {"type": "array", "items": {
        "type": "object",
        "properties": {"subject": {"type": "string"}, "relation": {"type": "string"}, "object": {"type": "string"}},
        "required": ["subject", "relation", "object"]}}},
    "required": ["triples"],
}
_GENERIC = {"it", "this", "that", "they", "yes", "no", "model", "system", "thing", "which", "these", "text"}


def extract_triples(chunk):
    """LLM entity/relation extraction -> list of (subject, relation, object).

    Uses Ollama structured output so a small model always returns valid JSON, plus a worked
    example and a filter that drops vague entities such as 'it' or 'model'.
    """
    prompt = (
        "Extract factual knowledge-graph triples from the text.\n"
        "Rules: entities must be specific named concepts (e.g. 'hnsw', 'reciprocal rank fusion', "
        "'all-minilm-l6-v2'), never vague words like 'it', 'model' or 'system'. Resolve pronouns. "
        "The relation is a short verb phrase.\n\n"
        "Example text: HNSW builds a layered graph. FAISS is a library for in-process vector search.\n"
        'Example output: {"triples": [{"subject": "hnsw", "relation": "builds", "object": "layered graph"}, '
        '{"subject": "faiss", "relation": "is", "object": "library for in-process vector search"}]}\n\n'
        "Text: " + chunk
    )
    try:
        data = json.loads(llm(prompt, schema=TRIPLE_SCHEMA)).get("triples", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    triples = []
    for t in data:
        s, r, o = (str(t.get(k, "")).strip().lower() for k in ("subject", "relation", "object"))
        if s and r and o and s != o and s not in _GENERIC and o not in _GENERIC and max(len(s), len(o)) <= 60:
            triples.append((s, r, o))
    return list(dict.fromkeys(triples))


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
