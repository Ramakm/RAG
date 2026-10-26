"""Incremental ingestion: hash ledger -> parse -> chunk -> metadata -> embed/upsert. Re-run any time (e.g. nightly)."""
import hashlib
import json
from pathlib import Path

from .. import vectorstore
from ..config import settings
from . import metadata
from .chunking import chunk_sections
from .parsers import PARSERS, parse, Section

CHUNKER_VERSION = "1"


def _load_ledger() -> dict:
    p = settings.ledger_path
    return json.loads(p.read_text()) if p.exists() else {}


def _save_ledger(ledger: dict) -> None:
    settings.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(meta: dict) -> dict:
    """Chroma accepts only str/int/float/bool values."""
    return {k: v for k, v in meta.items() if v not in (None, "")}


def build_chunks(path: Path, doc_id: str) -> tuple[list[str], list[dict]]:
    """Return (documents_to_embed, metadatas) for one file."""
    doc_type = metadata.infer_doc_type(path)
    sections: list[Section] = parse(path)
    chunks = chunk_sections(sections, doc_type)
    # A multi-row ticket file (Excel) must not leak the first row's batch/model onto every other row.
    multi_record = doc_type == "service_ticket" and len(sections) > 1
    doc_meta = {} if multi_record else metadata.extract("\n".join(s.text for s in sections)[:4000])
    docs, metas = [], []
    for i, c in enumerate(chunks):
        meta = {**doc_meta, **metadata.extract(c.text), **c.meta}  # chunk-level beats doc-level; structured columns beat regex
        meta.update(doc_id=doc_id, doc_type=doc_type, source_path=doc_id, chunk_index=i, chunker_version=CHUNKER_VERSION)
        if c.page is not None:
            meta["page"] = c.page
        if c.section:
            meta["section"] = c.section
        # Context prefix improves retrieval; the clean chunk text follows it.
        header = f"[{doc_type} | {doc_id}" + (f" | batch {meta['batch_id']}" if "batch_id" in meta else "") + (f" | model {meta['model']}" if "model" in meta else "") + "]"
        docs.append(f"{header}\n{c.text}")
        metas.append(_clean(meta))
    return docs, metas


def ingest(docs_dir: Path | None = None, force: bool = False) -> dict:
    docs_dir = docs_dir or settings.docs_dir
    ledger = _load_ledger()
    stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "empty": 0, "chunks": 0, "errors": []}
    seen = set()
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in PARSERS or path.name.startswith("~$"):
            continue
        doc_id = path.relative_to(docs_dir).as_posix()
        seen.add(doc_id)
        digest = _file_hash(path)
        if not force and ledger.get(doc_id) == digest:
            stats["skipped"] += 1
            continue
        try:
            docs, metas = build_chunks(path, doc_id)
        except Exception as e:  # one bad file must not stop the run
            stats["errors"].append(f"{doc_id}: {type(e).__name__}: {e}")
            continue
        vectorstore.delete_doc(doc_id)  # replace all chunks of a changed file
        if not docs:
            stats["empty"] += 1
            ledger[doc_id] = digest
            continue
        vectorstore.upsert([f"{doc_id}::{i}" for i in range(len(docs))], docs, metas)
        stats["updated" if doc_id in ledger else "added"] += 1
        stats["chunks"] += len(docs)
        ledger[doc_id] = digest
    for gone in set(ledger) - seen:  # file deleted from the share
        vectorstore.delete_doc(gone)
        del ledger[gone]
        stats["removed"] += 1
    _save_ledger(ledger)
    stats["total_chunks_in_index"] = vectorstore.count()
    return stats
