"""FastAPI server: chat with your files (RAG), threads saved in MongoDB, traced in LangSmith."""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import config  # noqa: F401  (loads .env and configures LangSmith before other imports)

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field

import chat
import guardrails
import live_tools
import mcp_client
import rag
import store
import tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("app")


def _visitor(request: Request | WebSocket) -> str:
    """The visitor's IP. Behind the Caddy proxy (a private Docker address) use X-Forwarded-For."""
    host = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for", "")
    try:
        behind_proxy = ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        behind_proxy = False
    if forwarded and behind_proxy:
        return forwarded.split(",")[0].strip() or host
    return host


def _rate_limit(request: Request, bucket: str, per_minute: int) -> None:
    ok, wait = guardrails.limiter.allow(bucket, _visitor(request), per_minute)
    if not ok:
        store.log_guardrail("rate_limited", bucket, visitor=_visitor(request), detail=request.url.path)
        raise HTTPException(429, f"Too many requests. Please wait about {wait} seconds and try again.")

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
    # Start the Salesforce MCP server (if configured) and hand its tools to the chat.
    sf_tools = await asyncio.to_thread(mcp_client.load_mcp_tools)
    log.info("salesforce tools: %s", ", ".join(t.name for t in sf_tools) or "none (not configured or failed)")
    # Live data tools: weather, share prices, web search.
    extra = live_tools.load_live_tools()
    log.info("live data tools: %s", ", ".join(t.name for t in extra) or "none")
    chat.TOOLS = sf_tools + extra
    if config.SARVAM_API_KEY:
        # Load live voice (Pipecat, WebRTC) in the background so the first live session starts quickly.
        asyncio.get_running_loop().run_in_executor(None, _live_voice)
    yield


app = FastAPI(title="RAG Chat", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    voice: bool = False  # asked with the mic: reply is short, spoken-style, Hindi for Hindi questions


class TtsIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class ThreadPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    pinned: bool | None = None


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "rate_limit_exceeded" in text or "Error code: 429" in text:
        wait = re.search(r"try again in ([0-9hms.]+)", text)
        when = f" Try again in about {wait.group(1).split('.')[0]}s." if wait else " Try again later."
        if "tokens per day" in text:
            return "Groq's daily token limit for the AI model is used up." + when
        return "Groq is rate-limiting requests right now." + when
    return text


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
        "live_voice": bool(config.SARVAM_API_KEY),
        "langsmith_project": config.LANGSMITH_PROJECT,
        "guardrails": {
            "enabled": config.GUARDRAILS_ENABLED,
            "llm_check": config.GUARDRAILS_ENABLED and config.GUARDRAILS_LLM_CHECK,
            "confirm_writes": config.GUARDRAILS_ENABLED and config.GUARDRAILS_CONFIRM_WRITES,
            "output_filter": config.GUARDRAILS_ENABLED and config.GUARDRAILS_OUTPUT_FILTER,
            "messages_per_minute": config.GUARDRAILS_MSGS_PER_MINUTE,
        },
        "live_tools": [t.name for t in chat.TOOLS if not t.name.startswith("sf_")],
        "salesforce": {
            "connected": any(t.name.startswith("sf_") for t in chat.TOOLS),
            "auth": "connected app" if mcp_client.connected_app_configured()
                    else (f"cli alias {config.SF_ORG_ALIAS}" if config.SF_ORG_ALIAS else "not configured"),
            "tools": [t.name for t in chat.TOOLS if t.name.startswith("sf_")],
        },
    }


# ---------- files ----------

@app.get("/api/files")
async def list_files():
    return store.list_files()


@app.post("/api/files")
async def upload_files(request: Request, files: list[UploadFile] = File(...)):
    _rate_limit(request, "upload", config.GUARDRAILS_UPLOADS_PER_MINUTE)
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


# ---------- voice input ----------

_stt_client: Groq | None = None


# Phrases Whisper is known to invent from silence or background noise.
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "thank you so much for watching",
    "please subscribe", "subscribe", "bye", "you", "okay", "ok", "so", "uh", "um", "hmm",
    "subtitles by the amara.org community", "धन्यवाद",
}


_SOUND_TAG_RE = re.compile(r"[\[(*♪][^\])*♪]{0,40}[\])*♪]")  # [MUSIC], (siren blares), *cough*
_CAPS_TAIL_RE = re.compile(r"(?:\s+[A-Z]{3,}){1,4}\s*$")  # "... right now? SIREN BLARES"


def _strip_sound_tags(text: str) -> str:
    text = _SOUND_TAG_RE.sub(" ", text)
    tail = _CAPS_TAIL_RE.search(text)
    # Only drop a trailing ALL-CAPS run when it follows finished speech and has no digits/IDs.
    if tail and re.search(r"[.?!।]\s*$", text[: tail.start()]):
        text = text[: tail.start()]
    return re.sub(r"\s+", " ", text).strip()


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _is_hallucination(text: str) -> bool:
    return text.lower().strip(" .!?,।") in _HALLUCINATIONS


def _confident_text(resp) -> str:
    """Only the segments Whisper is confident contain real speech (drops silence and noise)."""
    segments = _field(resp, "segments") or []
    if not segments:
        return (_field(resp, "text", "") or "").strip()
    kept = [
        _field(s, "text", "") for s in segments
        if (_field(s, "no_speech_prob", 0) or 0) < config.STT_MAX_NO_SPEECH
        and (_field(s, "avg_logprob", 0) or 0) > config.STT_MIN_LOGPROB
    ]
    return " ".join(t.strip() for t in kept if t and t.strip())


def _transcribe(filename: str, data: bytes) -> str:
    global _stt_client
    if _stt_client is None:
        _stt_client = Groq(api_key=config.GROQ_API_KEY)
    kwargs = {"file": (filename, data), "model": config.GROQ_STT_MODEL, "temperature": 0.0,
              "response_format": "verbose_json"}
    if config.STT_PROMPT:
        kwargs["prompt"] = config.STT_PROMPT  # vocabulary hint: order IDs, couriers, store name
    if config.STT_LANGUAGE:
        kwargs["language"] = config.STT_LANGUAGE
    resp = _stt_client.audio.transcriptions.create(**kwargs)
    text = _confident_text(resp)
    # Auto-detect is for Hindi + English. Whisper often labels Hindi speech as Urdu (and writes it
    # in Urdu script); real speech in any other language is transcribed again as Hindi.
    # Only when the first pass found confident speech: forcing Hindi on noise invents words.
    detected = (_field(resp, "language", "") or "").lower()
    if text and not config.STT_LANGUAGE and detected and detected not in {"english", "en", "hindi", "hi"}:
        # Noise often comes back as a hallucination ("Thank you for watching.") in a random language;
        # that is not speech, so do not force it into Hindi (which invents a Hindi word).
        no_speech = max((_field(s, "no_speech_prob", 0) or 0) for s in (_field(resp, "segments") or [{}]))
        if _is_hallucination(text) or no_speech >= 0.3:
            return ""
        log.info("whisper detected %r; transcribing again as Hindi", detected)
        text = _confident_text(_stt_client.audio.transcriptions.create(**{**kwargs, "language": "hi"}))
    text = _strip_sound_tags(text)

    normalized = text.lower().strip(" .!?,।")
    if not normalized or _is_hallucination(text):
        return ""
    if config.STT_PROMPT:
        # Whisper sometimes "hears" the vocabulary hint in noise. Drop output made mostly of hint words.
        hint = set(re.findall(r"\w+", config.STT_PROMPT.lower()))
        words = re.findall(r"\w+", normalized)
        if words and sum(w in hint for w in words) / len(words) >= 0.6:
            return ""
    return text


@app.post("/api/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    _rate_limit(request, "voice", config.GUARDRAILS_VOICE_PER_MINUTE)
    """Speech to text for the microphone button (Groq Whisper)."""
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio")
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "recording is too long")
    try:
        text = await asyncio.to_thread(_transcribe, audio.filename or "speech.webm", data)
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(502, f"Could not transcribe the recording: {exc}")
    return {"text": text}


@app.post("/api/tts")
async def text_to_speech(request: Request, body: TtsIn):
    _rate_limit(request, "tts", config.GUARDRAILS_TTS_PER_MINUTE)
    """MP3 of an answer, read by a Hindi voice (Devanagari text) or an Indian-English voice."""
    try:
        audio, voice = await tts.synthesize(body.text)
    except Exception as exc:
        log.exception("tts failed")
        raise HTTPException(502, f"Could not create the spoken reply: {exc}")
    if not audio:
        raise HTTPException(422, "nothing to speak")
    return Response(audio, media_type="audio/mpeg", headers={"X-Voice": voice, "Cache-Control": "no-store"})


@app.get("/api/tts/stream")
async def text_to_speech_stream(request: Request, text: str = Query(..., min_length=1, max_length=4000)):
    _rate_limit(request, "tts", config.GUARDRAILS_TTS_PER_MINUTE)
    """Streaming MP3 for one sentence: the browser plays it while it is still arriving.
    GET so an <audio> element can load it directly."""
    if not tts.clean_for_speech(text):
        return Response(status_code=204)  # nothing speakable (only markdown, a citation…)

    async def body():
        try:
            async for chunk in tts.stream(text):
                yield chunk
        except Exception:
            log.exception("tts stream failed")

    return StreamingResponse(body(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


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
async def chat_turn(body: ChatIn, request: Request):
    """Server-sent events: meta -> sources -> token... -> done (or error).
    Also 'tool' (Salesforce step), 'confirm' (change waiting for yes/no), 'redact' (answer replaced)."""
    started = time.perf_counter()
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "empty message")
    _rate_limit(request, "chat", config.GUARDRAILS_MSGS_PER_MINUTE)
    visitor = _visitor(request)

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
                thread_id, question, history, emit, first_turn, body.voice, visitor,
                # LangSmith groups runs that share this metadata into one Thread.
                langsmith_extra={"metadata": {"thread_id": thread_id, "session_id": thread_id}},
            )
            if result.get("title"):
                store.rename_thread(thread_id, result["title"])
            saved = store.add_message(thread_id, "assistant", result["answer"], sources=result["sources"],
                                      usage=result["usage"], run_id=result["run_id"],
                                      duration_ms=int((time.perf_counter() - started) * 1000),
                                      tools=result.get("tools"), guardrail=result.get("guardrail"),
                                      pending_action=result.get("pending_action"))
            emit("done", {"message": saved, "thread": store.get_thread(thread_id)})
        except Exception as exc:
            log.exception("chat turn failed")
            emit("error", {"message": _friendly_error(exc)})
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


# ---------- live voice (Sarvam, real time, interruptible) ----------

class LiveOffer(BaseModel):
    sdp: str = Field(min_length=1, max_length=20000)
    type: str = Field(pattern="^offer$")
    thread_id: str | None = None


def _live_voice():
    import live_voice  # loaded on first use: Pipecat and the voice-activity model are large
    return live_voice


@app.get("/api/live/config")
async def live_config(request: Request):
    if not config.SARVAM_API_KEY:
        raise HTTPException(503, "Live voice is not set up (no SARVAM_API_KEY).")
    return {"ice_servers": _live_voice().ice_servers_for_browser(_visitor(request))}


@app.post("/api/live/offer")
async def live_offer(body: LiveOffer, request: Request):
    """WebRTC handshake: the page's offer in, the server's answer out; the live session then runs."""
    if not config.SARVAM_API_KEY:
        raise HTTPException(503, "Live voice is not set up (no SARVAM_API_KEY).")
    _rate_limit(request, "live", config.GUARDRAILS_MSGS_PER_MINUTE)
    try:
        return await _live_voice().start_webrtc_session(body.sdp, body.type, body.thread_id, _visitor(request))
    except Exception as exc:
        log.exception("live voice WebRTC offer failed")
        raise HTTPException(500, "Could not start live voice.") from exc


@app.websocket("/ws/live")
async def live_voice_session(websocket: WebSocket, thread_id: str | None = None):
    """Fallback when WebRTC cannot connect: raw audio over a WebSocket."""
    await websocket.accept()
    if not config.SARVAM_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Live voice is not set up (no SARVAM_API_KEY)."}))
        await websocket.close()
        return
    visitor = _visitor(websocket)
    ok, wait = guardrails.limiter.allow("live", visitor, config.GUARDRAILS_MSGS_PER_MINUTE)
    if not ok:
        await websocket.send_text(json.dumps({"type": "error", "message": f"Too many requests. Wait about {wait}s."}))
        await websocket.close()
        return
    await _live_voice().run_websocket_session(websocket, thread_id, visitor)


if __name__ == "__main__":
    # Local run: python app.py  (open http://127.0.0.1:8000)
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
