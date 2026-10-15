"""Seeing images: CLIP embeddings (text↔image search), OCR and vision-model captions."""
from __future__ import annotations

import base64
import io
import re
import threading
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .config import CLIP_MODEL, OLLAMA_URL, TESSERACT

CAPTION_PROMPT = ("Describe this image for a search index in 2-4 factual sentences: the main subject, "
                  "any people, objects, setting, charts or diagrams and what they show, and any important "
                  "visible text. Do not speculate.")

_lock = threading.Lock()


@lru_cache(maxsize=1)
def _clip():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CLIP_MODEL)


def clip_model():
    with _lock:
        return _clip()


def clip_images(paths: list[Path], batch: int = 16) -> np.ndarray:
    model = clip_model()
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        out.append(model.encode(imgs, batch_size=batch, normalize_embeddings=True, show_progress_bar=False))
    return np.asarray(np.vstack(out), dtype="float32") if out else np.zeros((0, 512), dtype="float32")


def clip_image_bytes(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(clip_model().encode([img], normalize_embeddings=True, show_progress_bar=False), dtype="float32")


def clip_text(texts: list[str]) -> np.ndarray:
    # CLIP's text tower only sees 77 tokens; keep queries short.
    texts = [" ".join(t.split()[:50]) for t in texts]
    return np.asarray(clip_model().encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


def ocr(path: Path, min_conf: int = 60) -> str:
    """Tesseract OCR keeping only confident words, so photos don't produce gibberish."""
    if not TESSERACT:
        return ""
    import pytesseract

    try:
        img = Image.open(path).convert("L")
        if max(img.size) < 1000:
            img = img.resize((img.width * 2, img.height * 2))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, timeout=30)
    except Exception:
        return ""
    lines: dict[tuple, list[str]] = {}
    for word, conf, b, p, ln in zip(data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]):
        if word.strip() and float(conf) >= min_conf:
            lines.setdefault((b, p, ln), []).append(word.strip())
    text = "\n".join(" ".join(ws) for ws in lines.values())
    real_words = re.findall(r"[A-Za-z]{3,}", text)
    return text if len(real_words) >= 2 else ""


def image_b64(path_or_bytes, max_side: int = 768) -> str:
    img = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def caption(model: str, image, prompt: str = CAPTION_PROMPT, timeout: float = 180) -> str:
    """Describe an image with a local vision model in Ollama (blocking)."""
    payload = {"model": model, "prompt": prompt, "images": [image_b64(image)], "stream": False,
               "options": {"temperature": 0.1, "num_predict": 220}}
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return re.sub(r"<think>[\s\S]*?</think>", "", r.json().get("response", "")).strip()
