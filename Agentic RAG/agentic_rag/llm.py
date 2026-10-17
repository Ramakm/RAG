"""LLM factory. Nodes call `llm.get_llm(...)` through this module so tests can swap in a fake."""
import os

from langchain_anthropic import ChatAnthropic


class ConfigError(RuntimeError):
    """Misconfiguration (e.g. missing API key). Nodes re-raise this instead of degrading gracefully."""


def get_llm(model: str) -> ChatAnthropic:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ConfigError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatAnthropic(model=model, max_tokens=2048)
