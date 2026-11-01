import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL, TOP_K


class VectorStore:
    """In-memory store: normalized embeddings + cosine similarity via dot product."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks: list[dict] = []  # {"source": str, "text": str}
        self.vectors = np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

    def add(self, source: str, texts: list[str]) -> int:
        self.remove(source)  # re-uploading a file replaces it
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        self.vectors = np.vstack([self.vectors, vecs.astype(np.float32)])
        self.chunks += [{"source": source, "text": t} for t in texts]
        return len(texts)

    def remove(self, source: str) -> None:
        keep = [i for i, c in enumerate(self.chunks) if c["source"] != source]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
        return counts

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        if not self.chunks:
            return []
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
