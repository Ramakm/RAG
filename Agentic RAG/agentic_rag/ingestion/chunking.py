"""Type-aware chunking. Tickets: one per chunk. Tables: whole. Everything else: paragraph packing."""
import re
from dataclasses import dataclass, field

from .parsers import Section

MAX_CHARS = {"qc_report": 1500, "supplier_spec": 1500, "manual": 1600, "service_ticket": 3000, "other": 1500}
OVERLAP = {"manual": 200}
TABLE_MAX = 6000


@dataclass
class Chunk:
    text: str
    page: int | None = None
    section: str = ""
    meta: dict = field(default_factory=dict)


def _pieces(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(s) > max_chars:  # no sentence boundary: hard split
            out.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def split_text(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for piece in (x for p in paras for x in _pieces(p, max_chars)):
        if cur and len(cur) + len(piece) + 2 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            cur = f"{cur}\n\n{piece}".strip() if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def chunk_sections(sections: list[Section], doc_type: str) -> list[Chunk]:
    max_chars = MAX_CHARS.get(doc_type, MAX_CHARS["other"])
    out: list[Chunk] = []
    for s in sections:
        if s.is_table and len(s.text) <= TABLE_MAX:
            parts = [s.text]
        else:
            parts = split_text(s.text, max_chars, OVERLAP.get(doc_type, 0))
        out += [Chunk(text=p, page=s.page, section=s.heading, meta=dict(s.meta)) for p in parts]
    return out
