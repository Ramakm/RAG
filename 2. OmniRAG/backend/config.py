"""Central configuration. Every value can be overridden with an environment variable."""
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = Path(os.getenv("OMNIRAG_DATA_DIR", BASE_DIR / "data"))
KB_DIR = DATA_DIR / "collections"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# Empty means "pick the best model that is installed in Ollama".
LLM_MODEL = os.getenv("OMNIRAG_LLM_MODEL", "")
VISION_MODEL = os.getenv("OMNIRAG_VISION_MODEL", "")
PREFERRED_MODELS = ["llama3.2", "llama3.1", "qwen2.5", "qwen3", "gemma3", "mistral", "gemma2", "phi4", "llama3"]
PREFERRED_VISION = ["qwen2.5vl", "llama3.2-vision", "gemma3", "minicpm-v", "llava", "moondream", "llava-phi3", "bakllava"]

TEXT_EMBED_MODEL = os.getenv("OMNIRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CLIP_MODEL = os.getenv("OMNIRAG_CLIP_MODEL", "clip-ViT-B-32")
WHISPER_MODEL = os.getenv("OMNIRAG_WHISPER_MODEL", "base")
WHISPER_SIZES = ("tiny", "base", "small", "medium")

CHUNK_SIZE = int(os.getenv("OMNIRAG_CHUNK_SIZE", 900))       # characters
CHUNK_OVERLAP = int(os.getenv("OMNIRAG_CHUNK_OVERLAP", 150))  # characters
TRANSCRIPT_WINDOW = 45.0                                       # seconds of speech per transcript chunk
MAX_UPLOAD_MB = int(os.getenv("OMNIRAG_MAX_UPLOAD_MB", 2048))
MAX_CAPTIONS = int(os.getenv("OMNIRAG_MAX_CAPTIONS", 60))      # vision-model descriptions per build
MAX_FRAMES = int(os.getenv("OMNIRAG_MAX_FRAMES", 120))         # sampled frames per video

TESSERACT = shutil.which("tesseract")

DOC_EXT = {".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".log"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def modality_of(name: str) -> str | None:
    ext = Path(name).suffix.lower()
    for modality, exts in (("document", DOC_EXT), ("image", IMAGE_EXT), ("audio", AUDIO_EXT), ("video", VIDEO_EXT)):
        if ext in exts:
            return modality
    return None


KB_DIR.mkdir(parents=True, exist_ok=True)
