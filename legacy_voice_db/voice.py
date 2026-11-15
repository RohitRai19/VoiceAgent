"""Speech-to-text (Groq Whisper) and text-to-speech (edge-tts)."""
from __future__ import annotations

import io
import os
import re

import edge_tts
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")

_client: Groq | None = None


def _groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# ---------- STT ----------

def transcribe(audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, str]:
    """Return (text, language_name). language_name is Whisper's full name, e.g. 'hindi'."""
    resp = _groq().audio.transcriptions.create(
        file=(filename, io.BytesIO(audio_bytes)),
        model=GROQ_STT_MODEL,
        response_format="verbose_json",
        temperature=0.0,
    )
    text = (getattr(resp, "text", "") or "").strip()
    language = (getattr(resp, "language", None) or "english").lower()
    return text, language


# ---------- voice selection ----------

VOICE_BY_LANGUAGE = {
    "hindi": "hi-IN-SwaraNeural",
    "english": "en-IN-NeerjaNeural",
    "tamil": "ta-IN-PallaviNeural",
    "telugu": "te-IN-ShrutiNeural",
    "bengali": "bn-IN-TanishaaNeural",
    "marathi": "mr-IN-AarohiNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "kannada": "kn-IN-SapnaNeural",
    "malayalam": "ml-IN-SobhanaNeural",
    "urdu": "ur-IN-GulNeural",
}
DEFAULT_VOICE = "en-IN-NeerjaNeural"

# Unicode script ranges -> language key. Checked against the reply text.
SCRIPT_RANGES = [
    ("devanagari", re.compile(r"[ऀ-ॿ]")),
    ("bengali", re.compile(r"[ঀ-৿]")),
    ("gujarati", re.compile(r"[઀-૿]")),
    ("tamil", re.compile(r"[஀-௿]")),
    ("telugu", re.compile(r"[ఀ-౿]")),
    ("kannada", re.compile(r"[ಀ-೿]")),
    ("malayalam", re.compile(r"[ഀ-ൿ]")),
    ("arabic", re.compile(r"[؀-ۿݐ-ݿ]")),
]


def detect_script(text: str) -> str | None:
    """Return the dominant non-Latin script name in `text`, or None if it is Latin/other."""
    counts = {name: len(rx.findall(text)) for name, rx in SCRIPT_RANGES}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def pick_voice(language: str | None, reply_text: str) -> str:
    """Choose the edge-tts voice. The reply's script overrides Whisper's language guess."""
    lang = (language or "").lower()
    script = detect_script(reply_text or "")

    if script == "devanagari":
        # Marathi and Hindi share Devanagari; keep Marathi if Whisper said so.
        return VOICE_BY_LANGUAGE["marathi"] if lang == "marathi" else VOICE_BY_LANGUAGE["hindi"]
    if script == "arabic":
        return VOICE_BY_LANGUAGE["urdu"]
    if script in VOICE_BY_LANGUAGE:
        return VOICE_BY_LANGUAGE[script]

    # Latin-script reply (English or Hinglish/romanised). An Indian-English voice reads
    # romanised Hindi acceptably; the Hindi voice reads Latin text poorly.
    if script is None and reply_text.strip():
        return VOICE_BY_LANGUAGE["english"]

    return VOICE_BY_LANGUAGE.get(lang, DEFAULT_VOICE)


# ---------- TTS ----------

async def synthesize(text: str, voice: str) -> bytes:
    """Return MP3 bytes for `text` spoken by `voice`."""
    if not text.strip():
        return b""
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)
