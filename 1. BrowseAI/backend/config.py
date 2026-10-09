"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = Path(os.getenv("BROWSEAI_DATA_DIR", BASE_DIR / "data"))
KB_DIR = DATA_DIR / "knowledge_bases"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Empty means "pick the best model that is installed in Ollama".
LLM_MODEL = os.getenv("BROWSEAI_LLM_MODEL", "")
PREFERRED_MODELS = ["llama3.2", "llama3.1", "qwen2.5", "mistral", "gemma3", "gemma2", "phi4", "phi3", "llama3"]

EMBED_MODEL = os.getenv("BROWSEAI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

USER_AGENT = "BrowseAI/1.0 (+https://github.com/; RAG website indexer)"
CHUNK_SIZE = int(os.getenv("BROWSEAI_CHUNK_SIZE", 1000))      # characters
CHUNK_OVERLAP = int(os.getenv("BROWSEAI_CHUNK_OVERLAP", 150))  # characters

KB_DIR.mkdir(parents=True, exist_ok=True)
