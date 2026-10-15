"""Chunking for every modality.

- Documents: heading-aware packing that never crosses a page/slide boundary.
- Transcripts: speech segments grouped into ~45 s windows with start/end timestamps.
Every chunk carries a human-readable location ("Page 3 › Results", "01:20–02:05")
that is shown in citations and prepended to the text before embedding.
"""
from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE, TRANSCRIPT_WINDOW

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def _split_long(block: str, size: int) -> list[str]:
    if len(block) <= size:
        return [block]
    parts, buf = [], ""
    for sent in SENTENCE_SPLIT.split(block):
        while len(sent) > size:
            parts.append(sent[:size])
            sent = sent[size:]
        if buf and len(buf) + len(sent) + 1 > size:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def _tail(text: str, overlap: int) -> str:
    out = ""
    for sent in reversed(SENTENCE_SPLIT.split(text)):
        if len(out) + len(sent) > overlap:
            break
        out = f"{sent} {out}".strip()
    return out


def chunk_section(section: dict, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split one page/slide/document into chunks of roughly `size` characters."""
    chunks: list[dict] = []
    headings: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_heading = ""

    def where() -> str:
        return " › ".join(h for _, h in headings)

    def flush(carry: bool):
        nonlocal buf, buf_heading
        body = "\n\n".join(buf).strip()
        if body and not all(HEADING.match(b) for b in buf):
            loc = " › ".join(x for x in (section["loc"], buf_heading) if x)
            chunks.append({"loc": loc, "page": section.get("page"), "asset": section.get("asset"), "text": body})
        tail = _tail(body, overlap) if carry else ""
        buf = [tail] if tail else []
        buf_heading = where()

    for block in re.split(r"\n\s*\n", section["text"]):
        block = block.strip()
        if not block:
            continue
        m = HEADING.match(block)
        if m:
            level = len(m.group(1))
            if sum(len(b) for b in buf) >= 200:
                flush(carry=False)
            headings = [h for h in headings if h[0] < level] + [(level, m.group(2).strip())]
            if not buf:
                buf_heading = where()
            buf.append(block)
            continue
        for piece in _split_long(block, size):
            if buf and sum(len(b) + 2 for b in buf) + len(piece) > size:
                flush(carry=True)
            if not buf:
                buf_heading = where()
            buf.append(piece)
    flush(carry=False)
    return chunks


def chunk_transcript(segments: list[dict], window: float = TRANSCRIPT_WINDOW) -> list[dict]:
    """Group whisper segments into time windows, overlapping by one segment for context."""
    chunks, cur = [], []
    for seg in segments:
        cur.append(seg)
        if cur[-1]["end"] - cur[0]["start"] >= window:
            chunks.append(cur)
            cur = cur[-1:]
    if cur and (not chunks or cur != chunks[-1][-1:]):
        chunks.append(cur)
    return [{"loc": f"{clock(c[0]['start'])}–{clock(c[-1]['end'])}",
             "start": round(c[0]["start"], 2), "end": round(c[-1]["end"], 2),
             "text": " ".join(s["text"].strip() for s in c)} for c in chunks]


def embedding_text(item: dict) -> str:
    crumbs = " › ".join(x for x in (item["file_name"], item.get("loc")) if x)
    return f"{crumbs}\n{item['text']}"
