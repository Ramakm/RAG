"""Thin wrapper over a persistent Chroma collection (swap for pgvector in production, see DESIGN.md)."""
import chromadb

from .config import settings

_collection = None


def collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        # default embedder: all-MiniLM-L6-v2 (ONNX, local). Production: bge-m3.
        _collection = client.get_or_create_collection("fan_docs", metadata={"hnsw:space": "cosine"})
    return _collection


def upsert(ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    col = collection()
    for i in range(0, len(ids), 500):
        col.upsert(ids=ids[i : i + 500], documents=documents[i : i + 500], metadatas=metadatas[i : i + 500])


def delete_doc(doc_id: str) -> None:
    collection().delete(where={"doc_id": doc_id})


def count() -> int:
    return collection().count()


def build_where(filters: dict) -> dict | None:
    clauses = [{k: v} for k, v in filters.items() if v]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search(query: str, k: int, filters: dict | None = None) -> list[tuple[str, dict]]:
    """Return [(document, metadata)]. If the filtered search is empty, retry without filters."""
    col = collection()
    if col.count() == 0:
        return []
    where = build_where(filters or {})
    res = col.query(query_texts=[query], n_results=k, where=where)
    hits = list(zip(res["documents"][0], res["metadatas"][0]))
    if not hits and where:
        res = col.query(query_texts=[query], n_results=k)
        hits = list(zip(res["documents"][0], res["metadatas"][0]))
    return hits


def reset_for_tests() -> None:
    global _collection
    _collection = None
