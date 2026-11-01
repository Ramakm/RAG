import io
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if ext in {".txt", ".md"}:
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type '{ext}'. Use: {', '.join(sorted(SUPPORTED))}")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fixed-size character windows with overlap, preferring to break on whitespace."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            end = space if space != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]
