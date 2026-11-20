"""Live voice mode: a real-time, interruptible voice conversation.

Audio travels over WebRTC (the browser's echo cancellation then removes the bot's own voice
from the mic, so speakers work without headphones). A WebSocket path is kept as a fallback
for networks where WebRTC cannot connect. On top, an echo guard ignores speech that repeats
what the bot just said, and needs a few words before it lets the user interrupt.

Pipeline (one per browser connection, built with Pipecat):

    browser mic (16 kHz PCM) -> Sarvam live speech-to-text (Saaras v4, server-side end-of-speech)
      -> user turn -> Brain (the same chat.run_turn as the text chat: RAG, Salesforce, tools,
         guardrails, Mongo history, LangSmith) -> Sarvam streaming voice (Bulbul v4 Flash)
      -> browser speaker (24 kHz PCM)

When the user starts talking while the bot is speaking, Pipecat sends an InterruptionFrame:
the voice stops, the browser drops the audio it still has queued, and the old answer is muted.

WebRTC: the mic and the bot's voice are normal WebRTC audio tracks; the JSON UI events below go
over a data channel (the browser also sends "ping" every second, which tells Pipecat it is there).

Wire format on the WebSocket fallback:
    browser -> server: binary = 16-bit mono PCM at 16 kHz
    server -> browser: binary = 16-bit mono PCM at 24 kHz; text = JSON UI events
        {"type": "ready"}                         session is listening
        {"type": "partial", "text": ...}          live words while the user speaks
        {"type": "user", "text": ...}             the finished question
        {"type": "thread", "thread": {...}}       the chat thread this session writes to
        {"type": "tool", "label": ...}            a tool step (Salesforce, weather, ...)
        {"type": "token", "text": ...}            answer text as it is generated
        {"type": "done", "message": {...}, "thread": {...}}
        {"type": "interrupt"}                     stop playback now
        {"type": "error", "message": ...}
        {"type": "end", "reason": "idle" | "max" | ...}
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import re
import time
from collections import deque

from fastapi import WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.sarvam.stt import SarvamRealtimeSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_start.min_words_user_turn_start_strategy import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

import chat
import config
import store
import tts

log = logging.getLogger("live")

# Where the answer can be cut into speakable pieces: a sentence end or a line break.
_SPEAK_BREAK = re.compile(r"[.!?।](?=\s)|\n")

IN_RATE = 16000   # browser mic -> Sarvam speech-to-text
OUT_RATE = 24000  # Sarvam voice -> browser speaker


def _ui(message: dict) -> OutputTransportMessageUrgentFrame:
    return OutputTransportMessageUrgentFrame(message=message)


# ---------- echo guard ----------

_WORD = re.compile(r"[^\s.,!?।;:\"'()\[\]\-–—]+")  # split on spaces/punctuation; keeps Hindi vowel signs
_ID_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])[-‐‑–](?=[A-Za-z0-9])")  # "ORD‑10002" and "ORD10002" are the same
# Small words that say nothing about who is talking; the decision rests on the other words.
_SMALL_WORDS = set("""a an the is are was were be to of in on at for and or it its this that my your our
i you we he she they me us do does did can could will would please so ok okay what which
है हैं था थी थे का की के को में से पर और या यह वह मेरा मेरी मेरे आपका आपकी आपके क्या भी तो ही
hai hain ka ki ke ko mein se par aur ya yeh woh mera meri mere aapka kya bhi""".split())


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(_ID_HYPHEN.sub("", text))]


class EchoMemory:
    """What the bot said recently. Speech that mostly repeats it is the bot hearing itself."""

    WINDOW_S = 30  # an answer is spoken a while after it is written, so keep it for a while

    AFTER_SPEECH_S = 2.5  # echo can still arrive from speech-to-text just after the bot stops

    def __init__(self):
        self._said: deque[tuple[float, list[str]]] = deque()
        self.bot_speaking = False
        self._stopped_at = 0.0

    def add(self, text: str) -> None:
        self._said.append((time.monotonic(), _words(text)))

    def bot_speaking_changed(self, speaking: bool) -> None:
        self.bot_speaking = speaking
        if not speaking:
            self._stopped_at = time.monotonic()

    def is_echo(self, heard: str) -> bool:
        words = _words(heard)
        now = time.monotonic()
        if not self.bot_speaking and now - self._stopped_at > self.AFTER_SPEECH_S:
            return False  # the bot is quiet: whatever is heard is the user
        while self._said and now - self._said[0][0] > self.WINDOW_S:
            self._said.popleft()
        if len(words) < 2 or not self._said:
            return False  # a short "yes" / "haan" / "stop" is the user, not an echo
        recent = {w for _, said in self._said for w in said}
        # Speech-to-text glues words and cuts them short ("the orderORD10"), so a word also counts
        # as repeated when it appears inside the bot's sentence with the spaces removed.
        joined = ["".join(said) for _, said in self._said]

        def repeated(w: str) -> bool:
            return w in recent or (len(w) >= 3 and any(w in j for j in joined))

        if sum(map(repeated, words)) / len(words) < 0.6:
            return False
        meaningful = [w for w in words if w not in _SMALL_WORDS]
        if not meaningful:
            return all(map(repeated, words))
        return sum(map(repeated, meaningful)) / len(meaningful) >= 0.6


class EchoGuardStartStrategy(MinWordsUserTurnStartStrategy):
    """Starts a user turn from the words heard: one word when the bot is quiet, at least
    LIVE_INTERRUPT_MIN_WORDS while it speaks, and never for words that echo the bot."""

    def __init__(self, echo: EchoMemory, **kwargs):
        super().__init__(**kwargs)
        self._echo = echo

    async def process_frame(self, frame: Frame):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._echo.bot_speaking_changed(True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._echo.bot_speaking_changed(False)
        await super().process_frame(frame)

    async def _handle_transcription(self, frame):
        if self._echo.is_echo(frame.text):
            log.info("echo ignored: %s", frame.text[:80])
            await self.trigger_reset_aggregation()
            return
        await super()._handle_transcription(frame)


class BrowserSerializer(FrameSerializer):
    """Raw PCM audio both ways; JSON text for UI events. Sample rates match the pipeline's."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            return json.dumps({"type": "interrupt"})
        if isinstance(frame, AudioRawFrame):
            return frame.audio or None
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message, ensure_ascii=False)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)) and data:
            return InputAudioRawFrame(audio=bytes(data), num_channels=1, sample_rate=self._sample_rate)
        return None  # the browser ends a session by closing the socket


class Tap(FrameProcessor):
    """Sits right after speech-to-text: shows the live words in the browser."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterimTranscriptionFrame) and frame.text.strip():
            await self.push_frame(_ui({"type": "partial", "text": frame.text}))
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self.push_frame(_ui({"type": "speaking", "on": True}))
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(_ui({"type": "speaking", "on": False}))
        await self.push_frame(frame, direction)


class Brain(FrameProcessor):
    """Answers each finished user turn with chat.run_turn and streams the text to the voice."""

    def __init__(self, thread_id: str | None, visitor: str, echo: EchoMemory):
        super().__init__()
        self._thread_id = thread_id
        self._visitor = visitor
        self._echo = echo
        self._turn = 0           # bumped by every new turn and every interruption
        self._lock = asyncio.Lock()  # one answer at a time per session (same thread history)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            question = _last_user_text(frame.context)
            frame.context.set_messages([])  # history lives in Mongo, not in the Pipecat context
            if question:
                self._turn += 1
                self.create_task(self._answer(question, self._turn))
            return
        if isinstance(frame, InterruptionFrame):
            self._turn += 1  # mutes the answer that is still being generated
            await self.push_frame(frame, direction)
            await self.push_frame(_ui({"type": "interrupt"}))  # WebRTC: lets the page update its status
            return
        await self.push_frame(frame, direction)

    async def _answer(self, question: str, turn: int):
        async with self._lock:
            if turn != self._turn:
                return  # the user already moved on
            await self._run(question, turn)

    async def _run(self, question: str, turn: int):
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(kind: str, data: dict | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

        thread, history, first_turn = await asyncio.to_thread(self._prepare, question)
        await self.push_frame(_ui({"type": "user", "text": question}))
        await self.push_frame(_ui({"type": "thread", "thread": thread}))

        lang = chat._reply_language(question, voice=True)
        voice_lang = Language.HI_IN if lang == "hindi" else Language.EN_IN
        await self.push_frame(TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(language=voice_lang)))

        thread_id = thread["thread_id"]

        def worker() -> None:
            # Runs to the end even after an interruption, so the full answer is still saved.
            try:
                result = chat.run_turn(
                    thread_id, question, history, emit, first_turn, True, self._visitor,
                    langsmith_extra={"metadata": {"thread_id": thread_id, "session_id": thread_id,
                                                  "mode": "live_voice"}},
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
                log.exception("live turn failed")
                emit("error", {"message": _friendly(exc)})
            finally:
                emit("__end__", None)

        loop.run_in_executor(None, worker)

        speaking = False
        pending = ""  # answer text not yet handed to the voice

        async def say(piece: str):
            nonlocal speaking
            text = tts.clean_for_speech(piece).strip()
            if not text:
                return  # markdown-only lines ("**Items:**") have nothing to say
            if text[-1] not in ".!?।:;,":
                text += "."  # a list line has no full stop; the voice needs one to start speaking
            if not speaking:
                await self.push_frame(LLMFullResponseStartFrame())
                speaking = True
            self._echo.add(piece)  # as written ("ORD-10002"), which is how speech-to-text hears it back
            self._echo.add(text)   # as spoken
            await self.push_frame(LLMTextFrame(text + " "))

        while True:
            kind, data = await queue.get()
            if kind == "__end__":
                break
            live = turn == self._turn
            if kind == "token":
                if live:
                    await self.push_frame(_ui({"type": "token", "text": data["text"]}))
                    pending += data["text"]
                    while (m := _SPEAK_BREAK.search(pending)):
                        piece, pending = pending[:m.end()], pending[m.end():]
                        await say(piece)
            elif kind == "tool":
                await self.push_frame(_ui({"type": "tool", "label": data.get("label", "")}))
            elif kind in ("redact", "relabel", "done", "error"):
                await self.push_frame(_ui({"type": kind, **(data or {})}))
                if kind == "redact" and live:
                    pending = ""  # the guardrail replaced the answer: say the replacement instead
                    await say(data.get("text", ""))
                if kind == "error" and live:
                    await say("Sorry, something went wrong. Please try again.")
        if turn == self._turn:
            if pending.strip():
                await say(pending)
            if speaking:
                await self.push_frame(LLMFullResponseEndFrame())

    def _prepare(self, question: str):
        thread = store.get_thread(self._thread_id) if self._thread_id else None
        if thread is None:
            thread = store.create_thread()
            self._thread_id = thread["thread_id"]
        history = store.get_messages(self._thread_id, limit=config.HISTORY_MESSAGES)
        first_turn = thread["message_count"] == 0
        if first_turn:
            store.rename_thread(self._thread_id, question[:60] + ("…" if len(question) > 60 else ""))
        store.add_message(self._thread_id, "user", question)
        return store.get_thread(self._thread_id), history, first_turn


def _last_user_text(context: LLMContext) -> str:
    for msg in reversed(context.get_messages()):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return (content or "").strip()
    return ""


def _friendly(exc: Exception) -> str:
    text = str(exc)
    if "rate_limit_exceeded" in text or "Error code: 429" in text:
        return "The AI model is rate-limited right now. Try again in a little while."
    return "Something went wrong answering that."


def _on_local_network(visitor: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(visitor or "")
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def ice_servers_for_browser(visitor: str | None = None) -> list[dict]:
    """The STUN/TURN servers the page's RTCPeerConnection uses (the same ones as the server).

    STUN is skipped for a visitor on this machine or the local network: direct addresses work
    there, and STUN only adds delay (the server waits ~5 s for STUN on adapters with no route
    out, such as the WSL / Hyper-V virtual adapter on Windows)."""
    use_stun = config.WEBRTC_STUN_URL and not _on_local_network(visitor)
    servers = [{"urls": config.WEBRTC_STUN_URL}] if use_stun else []
    if config.WEBRTC_TURN_URL:
        servers.append({"urls": config.WEBRTC_TURN_URL, "username": config.WEBRTC_TURN_USERNAME,
                        "credential": config.WEBRTC_TURN_PASSWORD})
    return servers


def _ice_servers_for_server(visitor: str) -> list[IceServer]:
    return [IceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential"))
            for s in ice_servers_for_browser(visitor)]


_sessions: set[asyncio.Task] = set()  # running WebRTC sessions (kept so they are not garbage-collected)


async def start_webrtc_session(sdp: str, sdp_type: str, thread_id: str | None, visitor: str) -> dict:
    """Answer the browser's WebRTC offer and start the live pipeline in the background."""
    connection = SmallWebRTCConnection(ice_servers=_ice_servers_for_server(visitor))
    await connection.initialize(sdp=sdp, type=sdp_type)
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    task = asyncio.create_task(_run_pipeline(transport, thread_id, visitor, "webrtc"))
    _sessions.add(task)
    task.add_done_callback(_sessions.discard)
    answer = connection.get_answer()
    if not answer:
        task.cancel()
        raise RuntimeError("WebRTC connection produced no answer")
    return answer


async def run_websocket_session(websocket: WebSocket, thread_id: str | None, visitor: str) -> None:
    """Fallback for networks where WebRTC cannot connect: raw PCM over the WebSocket."""
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=BrowserSerializer(),
        ),
    )
    await _run_pipeline(transport, thread_id, visitor, "websocket")


async def _run_pipeline(transport: BaseTransport, thread_id: str | None, visitor: str, kind: str) -> None:
    """Run one live voice session until the browser leaves, it goes idle, or LIVE_MAX_S passes."""
    stt = SarvamRealtimeSTTService(
        api_key=config.SARVAM_API_KEY,
        endpointing="vad",       # Sarvam decides when the user has finished speaking
        sample_rate=IN_RATE,
        settings=SarvamRealtimeSTTService.Settings(
            model=config.SARVAM_STT_MODEL,
            language=None,       # auto-detect Hindi / English
            stream_type="fast",
            mode="codemix",      # Hinglish: English words in Latin letters, Hindi in Devanagari
        ),
    )
    voice = SarvamTTSService(
        api_key=config.SARVAM_API_KEY,
        sample_rate=OUT_RATE,
        settings=SarvamTTSService.Settings(
            model=config.SARVAM_TTS_MODEL,
            voice=config.SARVAM_TTS_VOICE,
            language=Language.EN_IN,
        ),
    )

    echo = EchoMemory()
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # A turn starts from the words heard (echo guard); Sarvam still decides when it ends.
            user_turn_strategies=UserTurnStrategies(
                start=[EchoGuardStartStrategy(echo, min_words=config.LIVE_INTERRUPT_MIN_WORDS)],
                stop=[ExternalUserTurnStopStrategy()],
            ),
        ),
    )
    pipeline = Pipeline([
        transport.input(),
        stt,
        Tap(),
        aggregators.user(),
        Brain(thread_id, visitor, echo),
        voice,
        transport.output(),
    ])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_in_sample_rate=IN_RATE, audio_out_sample_rate=OUT_RATE),
        idle_timeout_secs=config.LIVE_IDLE_S,   # no speech from either side -> stop (saves credit)
        cancel_on_idle_timeout=False,
        enable_rtvi=False,
    )
    ended = {"reason": "closed"}

    async def finish(reason: str):
        if ended["reason"] == "closed":
            ended["reason"] = reason
            await task.queue_frames([_ui({"type": "end", "reason": reason}), EndFrame()])

    @transport.event_handler("on_client_connected")
    async def _connected(_transport, _client):
        await task.queue_frame(_ui({"type": "ready", "transport": kind}))

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_transport, _client):
        await task.cancel()

    @task.event_handler("on_idle_timeout")
    async def _idle(_task):
        await finish("idle")

    async def cap():
        await asyncio.sleep(config.LIVE_MAX_S)
        await finish("max")

    capper = asyncio.create_task(cap())
    started = time.perf_counter()
    try:
        await PipelineRunner(handle_sigint=False).run(task)
    except Exception:
        log.exception("live session failed")
    finally:
        capper.cancel()
        log.info("live %s session ended (%s) after %.0fs", kind, ended["reason"], time.perf_counter() - started)
