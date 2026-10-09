"""Answer generation with a local open-source LLM served by Ollama."""
from __future__ import annotations

import json
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx

from .config import LLM_MODEL, OLLAMA_URL, PREFERRED_MODELS

SYSTEM_PROMPT = """You are BrowseAI, an assistant that answers questions about the website {site}.
Answer using ONLY the numbered context excerpts taken from that website.
- Cite the excerpts you used inline, like [1] or [2][3], right after the relevant sentence.
- If the excerpts don't contain the answer, say you couldn't find it on {site} — do not guess.
- Be clear and concise. Use short paragraphs, bullet lists and **bold** key terms where it helps."""


async def ollama_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = sorted(m["name"] for m in r.json().get("models", []))
    except (httpx.HTTPError, ValueError):
        return {"online": False, "models": [], "default": None, "url": OLLAMA_URL}
    embedding_only = ("embed", "bge", "minilm", "e5-")
    chat_models = [m for m in models if not any(t in m for t in embedding_only)]
    return {"online": True, "models": chat_models, "default": pick_default(chat_models), "url": OLLAMA_URL}


def pick_default(models: list[str]) -> str | None:
    if LLM_MODEL:
        return LLM_MODEL
    for pref in PREFERRED_MODELS:
        for m in models:
            if m.startswith(pref):
                return m
    return models[0] if models else None


def build_messages(question: str, sources: list[dict], history: list[dict], site_url: str) -> list[dict]:
    site = urlparse(site_url).netloc or site_url
    context = "\n\n".join(
        f"[{i}] {s['title']}{' › ' + s['section'] if s['section'] else ''}\nURL: {s['url']}\n{s['text']}"
        for i, s in enumerate(sources, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(site=site)}]
    for turn in history[-6:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"][:2000]})
    messages.append({"role": "user", "content": f"Context excerpts:\n\n{context}\n\n---\nQuestion: {question}"})
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
        return "I couldn't find anything relevant on this website."
    lines = ["> **Ollama isn't running**, so here are the most relevant passages from the site "
             "instead of a generated answer. Start it with `ollama serve` to get full answers.\n"]
    for i, s in enumerate(sources[:3], 1):
        snippet = s["text"][:500].rsplit(" ", 1)[0] + ("…" if len(s["text"]) > 500 else "")
        lines.append(f"**{s['title']}** [{i}]\n\n{snippet}\n")
    return "\n".join(lines)
