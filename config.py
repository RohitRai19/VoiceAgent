"""Settings loaded from .env. Import this before anything that reads the environment."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
STATIC_DIR = BASE_DIR / "static"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
# Used automatically when the main model fails (e.g. its daily token limit is used up).
# It has its own separate Groq quota. Empty = no fallback.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
# Salesforce (read by salesforce_case_server.py / mcp_client.py from the environment).
SF_ORG_ALIAS = os.getenv("SF_ORG_ALIAS", "").strip()
# Spoken replies (edge-tts). Hindi voice for Devanagari answers, Indian-English otherwise.
TTS_VOICE_HI = os.getenv("TTS_VOICE_HI", "hi-IN-SwaraNeural")
TTS_VOICE_EN = os.getenv("TTS_VOICE_EN", "en-IN-NeerjaNeural")
TTS_RATE = os.getenv("TTS_RATE", "+0%")  # e.g. "+10%" to speak faster
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "2500"))
# Speech-to-text model for the microphone button.
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
# Optional vocabulary hint for Whisper. Off by default: in testing it made Whisper "hear" the
# hint itself in background noise and mangled spoken order numbers (10013 -> 100013).
STT_PROMPT = os.getenv("STT_PROMPT", "").strip()
# Empty = auto-detect (works for English, Hindi and Hinglish). Set e.g. "en" or "hi" to force one.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "").strip()
# Segments above this no-speech probability, or below this confidence, are treated as noise.
STT_MAX_NO_SPEECH = float(os.getenv("STT_MAX_NO_SPEECH", "0.6"))
STT_MIN_LOGPROB = float(os.getenv("STT_MIN_LOGPROB", "-1.0"))
# Seconds without data from Groq before a request is abandoned (the fallback model then answers).
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))
# Small fast model that writes the chat headings (falls back to GROQ_LLM_MODEL if unavailable).
TITLE_MODEL = os.getenv("TITLE_MODEL", "openai/gpt-oss-20b")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "voicedb")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
# Overview tables built from JSON arrays are sent with every question while their
# total size stays under this many characters (0 = never; they are then only searched).
PIN_OVERVIEW_CHARS = int(os.getenv("PIN_OVERVIEW_CHARS", "16000"))
# Top-level JSON keys that are not knowledge (e.g. lists of example questions).
JSON_SKIP_KEYS = {k.strip() for k in os.getenv("JSON_SKIP_KEYS", "sample_questions_the_agent_can_answer").split(",") if k.strip()}
# Files in this folder are indexed automatically when the server starts.
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_DIR.mkdir(exist_ok=True)
HISTORY_MESSAGES =int(os.getenv("HISTORY_MESSAGES", "10"))  # prior messages sent to the LLM
# true: questions unrelated to the files are answered from the model's own knowledge
# (labelled as such). false: every answer must come from the files.
ALLOW_GENERAL_KNOWLEDGE = os.getenv("ALLOW_GENERAL_KNOWLEDGE", "true").lower() == "true"
# Includes the model's hidden reasoning tokens, so long calculations need headroom.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "6000"))
MAX_UPLOAD_MB =int(os.getenv("MAX_UPLOAD_MB", "25"))

# LangSmith: tracing is only switched on when a key is present, so the app
# still runs (untraced) before the key is added.
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "voiceagent-rag-chat")
TRACING_ENABLED = bool(LANGSMITH_API_KEY) and os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
os.environ["LANGSMITH_TRACING"] = "true" if TRACING_ENABLED else "false"
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
