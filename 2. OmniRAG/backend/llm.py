"""Answer generation with local open-source models served by Ollama.

Text-only models get images as their captions/OCR text; vision models
(llava, llama3.2-vision, qwen2.5vl, gemma3, moondream…) also receive the pixels.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from .config import LLM_MODEL, OLLAMA_URL, PREFERRED_MODELS, PREFERRED_VISION, VISION_MODEL

SYSTEM_PROMPT = """You are OmniRAG, an assistant that answers questions about the user's collection "{name}",
which contains documents, images, audio and video.
Answer using ONLY the numbered context items below.
- Cite the items you used inline, like [1] or [2][3], right after the relevant sentence.
- Items can be document passages, image descriptions/OCR, video frames or speech transcripts with timestamps.
  When citing audio or video, mention the timestamp (e.g. "at 02:15").
- If images are attached, they are the context items stated in the question; describe what you actually see.
- If an image has no description, it was retrieved because it visually matches the question: point the user to it
  (file and page/timestamp) as the likely match, without inventing details about what it shows.
- If the context doesn't contain the answer, say you couldn't find it in this collection. Never make up facts.
- Be clear and concise. Use short paragraphs, bullet lists and **bold** key terms where it helps."""

_caps: dict[str, list[str]] = {}
EMBEDDING_ONLY = ("embed", "bge", "minilm", "e5-")


async def _capabilities(client: httpx.AsyncClient, model: str) -> list[str]:
    if model not in _caps:
        try:
            r = await client.post(f"{OLLAMA_URL}/api/show", json={"model": model})
            data = r.json()
            caps = data.get("capabilities")
            if caps is None:  # older Ollama: infer from the model's projector / name
                caps = ["completion"] + (["vision"] if data.get("projector_info") or any(
                    v in model for v in PREFERRED_VISION) else [])
            _caps[model] = caps
        except (httpx.HTTPError, ValueError):
            return []
    return _caps[model]


def _pick(models: list[str], preferred: list[str], forced: str) -> str | None:
    if forced and forced in models:
        return forced
    for pref in preferred:
        for m in models:
            if m.startswith(pref):
                return m
    return models[0] if models else None


async def ollama_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            names = sorted(m["name"] for m in r.json().get("models", []))
            chat = [m for m in names if not any(t in m for t in EMBEDDING_ONLY)]
            vision = [m for m in chat if "vision" in await _capabilities(client, m)]
    except (httpx.HTTPError, ValueError):
        return {"online": False, "models": [], "vision_models": [], "default": None, "default_vision": None,
                "url": OLLAMA_URL}
    return {"online": True, "models": chat, "vision_models": vision,
            "default": _pick(chat, PREFERRED_MODELS, LLM_MODEL),
            "default_vision": _pick(vision, PREFERRED_VISION, VISION_MODEL), "url": OLLAMA_URL}


def describe(s: dict, i: int) -> str:
    """One numbered context item, phrased for the item's modality."""
    if s["kind"] == "transcript":
        head = f"[{i}] Transcript of {s['modality']} \"{s['file_name']}\" · {s['loc']}"
        return f"{head}\n{s['text']}"
    if s["kind"] in ("image", "figure", "frame"):
        what = {"image": "Image", "figure": "Figure from", "frame": "Video frame from"}[s["kind"]]
        where = f" · {s['loc']}" if s.get("loc") else ""
        parts = [f"[{i}] {what} \"{s['file_name']}\"{where}"]
        if s.get("caption"):
            parts.append(f"Description: {s['caption']}")
        if s.get("ocr"):
            parts.append(f"Text visible in the image: {s['ocr']}")
        if not s.get("caption") and not s.get("ocr"):
            parts.append("(No description available: this image was retrieved because it visually matches the question.)")
        return "\n".join(parts)
    where = f" · {s['loc']}" if s.get("loc") else ""
    return f"[{i}] Document \"{s['file_name']}\"{where}\n{s['text']}"


def build_messages(question: str, sources: list[dict], history: list[dict], name: str,
                   attached: list[tuple[str, str]], user_image: bool) -> list[dict]:
    """`attached` = [(label, base64)] images sent to a vision model alongside the text."""
    context = "\n\n".join(describe(s, i) for i, s in enumerate(sources, 1)) or "(no matching items)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(name=name)}]
    for turn in history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"][:2000]})
    notes = []
    if user_image:
        notes.append("The user attached an image to this question"
                     + (" (the first attached image)." if attached and attached[0][0] == "user" else
                        ". Items that look similar to it were retrieved below."))
    labels = [lbl for lbl, _ in attached if lbl != "user"]
    if labels:
        notes.append("Attached images, in order after any user image: " + ", ".join(labels) + ".")
    user = {"role": "user",
            "content": f"Context items:\n\n{context}\n\n---\n{' '.join(notes)}\nQuestion: {question}".strip()}
    if attached:
        user["images"] = [b64 for _, b64 in attached]
    messages.append(user)
    return messages


async def stream_answer(model: str, messages: list[dict]) -> AsyncIterator[str]:
    payload = {"model": model, "messages": messages, "stream": True,
               "options": {"temperature": 0.2, "num_ctx": 8192}}
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="ignore")
                raise RuntimeError(f"Ollama error {r.status_code}: {body[:200]}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break


def extractive_answer(sources: list[dict]) -> str:
    """Used when Ollama isn't reachable, so the app still returns something useful."""
    if not sources:
        return "I couldn't find anything relevant in this collection."
    lines = ["> **Ollama isn't running**, so here are the most relevant items instead of a generated "
             "answer. Start it with `ollama serve` to get full answers.\n"]
    for i, s in enumerate(sources[:4], 1):
        text = s["text"] or "(matched visually)"
        snippet = text[:450].rsplit(" ", 1)[0] + ("…" if len(text) > 450 else "")
        where = f" · {s['loc']}" if s.get("loc") else ""
        lines.append(f"**{s['file_name']}{where}** [{i}]\n\n{snippet}\n")
    return "\n".join(lines)

