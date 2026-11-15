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
