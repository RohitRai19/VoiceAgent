"""FastAPI server: voice in, voice out."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

load_dotenv()

import brain  # noqa: E402
import db_tools  # noqa: E402
import voice  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("app")

app = FastAPI(title="Voice DB Assistant")

STATIC_DIR = Path(__file__).parent / "static"
HISTORY_TURNS = 6  # user+assistant pairs kept per session
_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    try:
        db_tools.db.command("ping")
        mongo = "ok"
    except Exception as exc:
        mongo = f"error: {exc}"
    return {"status": "ok", "mongo": mongo, "model": brain.GROQ_LLM_MODEL}


@app.post("/voice/reset")
async def reset(session_id: str = Form(...)):
    _history.pop(session_id, None)
    return {"ok": True}


@app.post("/voice/turn")
async def voice_turn(audio: UploadFile = File(...), session_id: str = Form("default")):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "empty audio")

    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. STT (blocking SDK -> threadpool)
    filename = audio.filename or "audio.webm"
    try:
        transcript, language = await asyncio.to_thread(voice.transcribe, audio_bytes, filename)
    except Exception as exc:
        log.exception("stt failed")
        raise HTTPException(502, f"transcription failed: {exc}")
    timings["stt"] = round(time.perf_counter() - t0, 3)

    if not transcript:
        return JSONResponse({
            "transcript": "", "language": language, "reply": "",
            "voice": None, "audio_b64": "", "timings": timings,
        })

    # 2. LLM + tools
    t1 = time.perf_counter()
    history = list(_history[session_id])
    try:
        reply = await asyncio.to_thread(brain.answer, transcript, language, history)
    except Exception as exc:
        log.exception("llm failed")
        raise HTTPException(502, f"llm failed: {exc}")
    timings["llm"] = round(time.perf_counter() - t1, 3)

    _history[session_id].append({"role": "user", "content": transcript})
    _history[session_id].append({"role": "assistant", "content": reply})

    # 3. TTS
    t2 = time.perf_counter()
    chosen_voice = voice.pick_voice(language, reply)
    try:
        mp3 = await voice.synthesize(reply, chosen_voice)
    except Exception as exc:
        log.exception("tts failed")
        mp3 = b""
    timings["tts"] = round(time.perf_counter() - t2, 3)
    timings["total"] = round(time.perf_counter() - t0, 3)

    log.info(json.dumps({
        "event": "turn", "session": session_id, "language": language, "voice": chosen_voice,
        "audio_in_bytes": len(audio_bytes), "audio_out_bytes": len(mp3),
        "transcript": transcript, "reply": reply, **timings,
    }, ensure_ascii=False))

    return {
        "transcript": transcript,
        "language": language,
        "reply": reply,
        "voice": chosen_voice,
        "audio_b64": base64.b64encode(mp3).decode("ascii"),
        "timings": timings,
    }
