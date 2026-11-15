"""FastAPI server: chat with your files (RAG), threads saved in MongoDB, traced in LangSmith."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import config  # noqa: F401  (loads .env and configures LangSmith before other imports)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import chat
import rag
import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("app")

def _ingest(name: str, data: bytes) -> dict:
    """Index a file, replacing any earlier version with the same name. Identical content is skipped."""
    if not data:
        raise ValueError("File is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    sha = hashlib.sha256(data + rag.INDEX_VERSION.encode()).hexdigest()
    existing = store.find_files_by_name(name)
    same = next((f for f in existing if f["sha256"] == sha), None)
    if same and len(existing) == 1:
        return {k: v for k, v in same.items() if k != "sha256"}
    file_id, chunks = rag.index_file(name, data)  # index first, so a bad file never removes a good one
    for old in existing:
        store.delete_file(old["file_id"])
        rag.delete_file(old["file_id"])
    return store.add_file(file_id, name, len(data), chunks, sha)


def _sync_knowledge_folder() -> None:
    for path in sorted(config.KNOWLEDGE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in rag.SUPPORTED_EXTENSIONS:
            continue
        try:
            info = _ingest(path.name, path.read_bytes())
            log.info("knowledge/%s ready (%s chunks)", path.name, info["chunks"])
        except Exception:
            log.exception("could not index knowledge/%s", path.name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(_sync_knowledge_folder)
    # Load the embedding model now, so the first question does not pay for it.
    await asyncio.to_thread(rag.embed, ["warm up"])
    yield


app = FastAPI(title="RAG Chat", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None


class ThreadPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    pinned: bool | None = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------- pages / health ----------

@app.get("/")
async def index():
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    try:
        store.db.command("ping")
        mongo = "ok"
    except Exception as exc:
        mongo = f"error: {exc}"
    return {
        "status": "ok",
        "mongo": mongo,
        "model": config.GROQ_LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
        "chunks": rag.chunk_count(),
        "langsmith_tracing": config.TRACING_ENABLED,
        "langsmith_project": config.LANGSMITH_PROJECT,
    }


# ---------- files ----------

@app.get("/api/files")
async def list_files():
    return store.list_files()


@app.post("/api/files")
async def upload_files(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "file").name
        data = await f.read()
        try:
            results.append({"ok": True, **await asyncio.to_thread(_ingest, name, data)})
        except ValueError as exc:
            results.append({"ok": False, "name": name, "error": str(exc)})
        except Exception as exc:
            log.exception("indexing failed for %s", name)
            results.append({"ok": False, "name": name, "error": f"Could not read this file: {exc}"})
    return results


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    if store.delete_file(file_id) is None:
        raise HTTPException(404, "file not found")
    await asyncio.to_thread(rag.delete_file, file_id)
    return {"ok": True}


# ---------- threads ----------

@app.get("/api/threads")
async def list_threads():
    return store.list_threads()


@app.post("/api/threads")
async def create_thread():
    return store.create_thread()


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str):
    thread = store.get_thread(thread_id)
    if thread is None:
        raise HTTPException(404, "thread not found")
    return {**thread, "messages": store.get_messages(thread_id)}


@app.patch("/api/threads/{thread_id}")
async def rename_thread(thread_id: str, body: ThreadPatch):
    if store.get_thread(thread_id) is None:
        raise HTTPException(404, "thread not found")
    if body.title is not None and body.title.strip():
        store.rename_thread(thread_id, body.title.strip())
    if body.pinned is not None:
        store.pin_thread(thread_id, body.pinned)
    return store.get_thread(thread_id)


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if not store.delete_thread(thread_id):
        raise HTTPException(404, "thread not found")
    return {"ok": True}


# ---------- chat ----------

@app.post("/api/chat")
async def chat_turn(body: ChatIn):
    """Server-sent events: meta -> sources -> token... -> done (or error)."""
    started = time.perf_counter()
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "empty message")

    if body.thread_id:
        thread = store.get_thread(body.thread_id)
        if thread is None:
            raise HTTPException(404, "thread not found")
    else:
        thread = store.create_thread()
    thread_id = thread["thread_id"]

    history = store.get_messages(thread_id, limit=config.HISTORY_MESSAGES)
    first_turn = thread["message_count"] == 0
    if first_turn:  # temporary heading until the generated one arrives with the answer
        title = question[:60] + ("…" if len(question) > 60 else "")
        store.rename_thread(thread_id, title)
        thread["title"] = title
    store.add_message(thread_id, "user", question)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, data: dict | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def worker() -> None:
        # Runs to completion even if the browser disconnects, so the reply is always saved.
        try:
            result = chat.run_turn(
                thread_id, question, history, emit, first_turn,
                # LangSmith groups runs that share this metadata into one Thread.
                langsmith_extra={"metadata": {"thread_id": thread_id, "session_id": thread_id}},
            )
            if result.get("title"):
                store.rename_thread(thread_id, result["title"])
            saved = store.add_message(thread_id, "assistant", result["answer"], sources=result["sources"],
                                      usage=result["usage"], run_id=result["run_id"],
                                      duration_ms=int((time.perf_counter() - started) * 1000))
            emit("done", {"message": saved, "thread": store.get_thread(thread_id)})
        except Exception as exc:
            log.exception("chat turn failed")
            emit("error", {"message": str(exc)})
        finally:
            emit("__end__", None)

    loop.run_in_executor(None, worker)

    async def stream():
        yield _sse("meta", {"thread_id": thread_id, "title": thread["title"]})
        while True:
            kind, data = await queue.get()
            if kind == "__end__":
                break
            yield _sse(kind, data)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
