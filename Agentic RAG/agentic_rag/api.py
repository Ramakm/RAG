"""FastAPI service.  Run: uvicorn agentic_rag.api:app --port 8000"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import agent, vectorstore
from .config import settings
from .ingestion.pipeline import ingest

app = FastAPI(title="Agentic RAG - Fan Manufacturing", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "db_exists": settings.db_path.exists(),
        "chunks_indexed": vectorstore.count(),
        "models": {"strong": settings.strong_model, "cheap": settings.cheap_model},
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not settings.db_path.exists():
        raise HTTPException(503, "No database found. Run `python -m agentic_rag.cli seed` (demo) or point DATA_DIR at your data.")
    return agent.ask(req.question)


@app.post("/ingest")
def run_ingest(force: bool = False) -> dict:
    return ingest(force=force)
