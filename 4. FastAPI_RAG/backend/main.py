import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .ingest import chunk_text, extract_text
from .llm import LocalLLM
from .store import VectorStore

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = VectorStore()
    state["llm"] = LocalLLM()
    yield


app = FastAPI(title="FastAPI RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    return [{"name": n, "chunks": c} for n, c in state["store"].sources().items()]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    try:
        text = extract_text(file.filename, data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No extractable text found in this file.")
    return {"name": file.filename, "chunks": state["store"].add(file.filename, chunks)}


@app.delete("/documents/{name}")
def delete_document(name: str):
    state["store"].remove(name)
    return {"deleted": name}


@app.post("/ask")
def ask(req: AskRequest):
    """Server-sent events: one `sources` event, then `token` events, then `done`."""
    contexts = state["store"].search(req.question)
    if not contexts:
        raise HTTPException(400, "Upload a document first.")

    def events():
        sources = [{"source": c["source"], "text": c["text"], "score": c["score"]} for c in contexts]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        for token in state["llm"].stream(req.question, contexts):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
