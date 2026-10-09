"""Heading-aware chunking.

Chunks follow the page's section structure where possible and every chunk
remembers its breadcrumb (e.g. "Pricing › Enterprise"), which is prepended to
the text before embedding so retrieval understands the context.
"""
from __future__ import annotations

import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_long(block: str, size: int) -> list[str]:
    if len(block) <= size:
        return [block]
    parts, buf = [], ""
    for sent in SENTENCE_SPLIT.split(block):
        while len(sent) > size:  # a single enormous "sentence"
            parts.append(sent[:size])
            sent = sent[size:]
        if len(buf) + len(sent) + 1 > size and buf:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def _tail(text: str, overlap: int) -> str:
    """The last few whole sentences of a chunk, up to `overlap` characters."""
    if overlap <= 0:
        return ""
    out = ""
    for sent in reversed(SENTENCE_SPLIT.split(text)):
        if len(out) + len(sent) > overlap:
            break
        out = f"{sent} {out}".strip()
    return out


def chunk_page(page: dict, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks: list[dict] = []
    headings: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_section = ""

    def section() -> str:
        return " › ".join(h for _, h in headings)

    def flush(carry: bool):
        nonlocal buf, buf_section
        body = "\n\n".join(buf).strip()
        if body and not all(HEADING.match(b) for b in buf):
            chunks.append({"url": page["url"], "title": page["title"], "section": buf_section, "text": body})
        tail = _tail(body, overlap) if carry else ""
        buf = [tail] if tail else []
        buf_section = section()

    for block in page["text"].split("\n\n"):
        m = HEADING.match(block)
        if m:
            level = len(m.group(1))
            if sum(len(b) for b in buf) >= 200:
                flush(carry=False)  # new section: start a fresh chunk
            headings = [h for h in headings if h[0] < level] + [(level, m.group(2).strip())]
            if not buf:
                buf_section = section()
            buf.append(block)
            continue
        for piece in _split_long(block, size):
            if sum(len(b) + 2 for b in buf) + len(piece) > size and buf:
                flush(carry=True)
            if not buf:
                buf_section = section()
            buf.append(piece)
    flush(carry=False)
    return chunks


def embedding_text(chunk: dict) -> str:
    crumbs = " › ".join(x for x in (chunk["title"], chunk["section"]) if x)
    return f"{crumbs}\n{chunk['text']}" if crumbs else chunk["text"]
