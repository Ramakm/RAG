import os

# Open-source models served locally through Hugging Face transformers (no Ollama, no API key).
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
