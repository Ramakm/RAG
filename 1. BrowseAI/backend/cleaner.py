"""Cross-page cleaning: remove site-wide boilerplate, junk lines and duplicate pages.

Main-content extraction works one page at a time, so text that repeats on
every page (a newsletter blurb, a "related posts" header, a disclaimer) can
still slip through. Looking across *all* pages lets us spot and drop it.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

JUNK = re.compile(
    r"(accept (all )?cookies|cookie (policy|settings|preferences)|we use cookies|all rights reserved|"
    r"skip to (main )?content|subscribe to (our )?newsletter|sign up for (our )?newsletter|"
    r"share (this|on) (post|page|twitter|facebook|linkedin|x)|^copyright\b|^©|back to top|"
    r"was this (page|article) helpful|enable javascript|javascript is (disabled|required))",
    re.I,
)
MIN_WORDS_PER_PAGE = 25
FENCE = re.compile(r"^```[\w+-]*$")


def _blocks(markdown: str) -> list[str]:
    """Split markdown into blocks, keeping fenced code blocks intact.

    trafilatura sometimes renders inline <code> as ``` mid-line (e.g. "... and ```"
    followed by a few lines and a closing fence). Those are folded back into the
    prose line as `inline code` so they can't swallow the rest of the page.
    """
    blocks, buf, in_code, inline = [], [], False, None
    for raw in markdown.splitlines():
        line = raw.strip().replace("\u00b6", "").replace("\u200b", "")  # heading anchors, zero-width spaces
        if inline is not None:  # inside a mid-line code span
            if FENCE.match(line):
                blocks.append(re.sub(r"[ \t]+", " ", inline + "`"))
                inline = None
            elif line:
                inline += " " + line
            continue
        if FENCE.match(line):
            buf.append(line)
            if in_code:
                blocks.append("\n".join(buf))
                buf = []
            in_code = not in_code
            continue
        if in_code:
            if line:  # blank lines would split the block during chunking
                buf.append(raw.rstrip())
            continue
        if not line:
            continue
        if line.count("```") % 2 == 1:
            inline = line.replace("```", "`").rstrip()
            continue
        blocks.append(re.sub(r"[ \t]+", " ", line.replace("```", "`")))
    if inline is not None:
        blocks.append(inline)
    if buf:
        blocks.append("\n".join(buf + ["```"]))
    return blocks


def _key(block: str) -> str:
    return re.sub(r"\W+", " ", block.lower()).strip()


def clean_pages(pages: list[dict]) -> tuple[list[dict], dict]:
    """pages: [{url, title, text}] -> (cleaned pages, stats)."""
    per_page = [_blocks(p["text"]) for p in pages]
    n = len(pages)

    # How many distinct pages does each block appear on?
    freq = Counter()
    for blocks in per_page:
        freq.update({_key(b) for b in blocks})
    threshold = max(3, int(n * 0.5)) if n >= 4 else n + 1  # needs enough pages to judge

    stats = {"boilerplate_blocks": 0, "junk_blocks": 0, "duplicate_blocks": 0,
             "duplicate_pages": 0, "thin_pages": 0, "words_before": 0, "words_after": 0}
    seen_pages: set[str] = set()
    cleaned = []
    for page, blocks in zip(pages, per_page):
        stats["words_before"] += sum(len(b.split()) for b in blocks)
        kept, seen_blocks = [], set()
        for b in blocks:
            k = _key(b)
            if not k:
                continue
            if k in seen_blocks:
                stats["duplicate_blocks"] += 1
                continue
            if freq[k] >= threshold and not b.startswith("#"):
                stats["boilerplate_blocks"] += 1
                continue
            if len(b) < 250 and JUNK.search(b):
                stats["junk_blocks"] += 1
                continue
            seen_blocks.add(k)
            kept.append(b)
        # Drop trailing headings that lost their content.
        while kept and kept[-1].startswith("#"):
            kept.pop()
        text = "\n\n".join(kept)
        words = len(text.split())
        if words < MIN_WORDS_PER_PAGE:
            stats["thin_pages"] += 1
            continue
        digest = hashlib.sha1(_key(text).encode()).hexdigest()
        if digest in seen_pages:
            stats["duplicate_pages"] += 1
            continue
        seen_pages.add(digest)
        stats["words_after"] += words
        cleaned.append({**page, "text": text, "words": words})
    return cleaned, stats
