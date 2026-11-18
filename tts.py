"""Text to speech for spoken replies (Microsoft neural voices via edge-tts, no API key)."""
from __future__ import annotations

import re

import edge_tts

import config

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_GENERAL_LINE = re.compile(r"^\s*[_*]*\s*General knowledge, not from your files\.?\s*[_*]*\s*", re.IGNORECASE)
_TABLE_RULE = re.compile(r"^\|?[\s:\-|]+\|?$")


# IDs such as ORD-10003, SKU-ELEC-003, ORD10013, DLV8812345671.
_ID_RE = re.compile(r"\b([A-Z]{2,5})(?:-([A-Z]{2,5}))?-?(\d{3,})\b")


def _spell_id(m: re.Match) -> str:
    """Read an ID like a support agent would: letters one by one, then digits one by one."""
    words = [" ".join(m.group(1))]
    if m.group(2):
        words.append(" ".join(m.group(2)))
    return ", ".join(words) + ", " + " ".join(m.group(3))


def pick_voice(text: str) -> str:
    """Hindi voice for Devanagari text, Indian-English voice otherwise."""
    return config.TTS_VOICE_HI if _DEVANAGARI.search(text) else config.TTS_VOICE_EN


def clean_for_speech(text: str) -> str:
    """Turn a markdown chat answer into plain sentences a voice can read."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks are not read aloud
    t = _GENERAL_LINE.sub("", t)
    t = re.sub(r"\s*\[\d+\]", "", t)  # citations
    t = re.sub(r"https?://\S+", "", t)
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s or (_TABLE_RULE.match(s) and "-" in s):
            continue
        if s.startswith("|"):  # table row -> "a, b, c"
            s = ", ".join(c.strip() for c in s.strip("|").split("|") if c.strip())
        s = re.sub(r"^#{1,6}\s*", "", s)
        s = re.sub(r"^(?:[-*•]|\d+\.)\s+", "", s)
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"[*_`#>~]", "", t)
    # Odd spaces/hyphens the model writes (ORD‑10003, 2026‑11‑16) confuse the voice.
    t = t.replace("\u2011", "-").replace("\u2010", "-").replace("\u202f", " ").replace("\u00a0", " ")
    t = _ID_RE.sub(_spell_id, t)  # ORD-10003 -> "O R D, 1 0 0 0 3"
    # Long numbers (case 00001069, tracking numbers) digit by digit, without leading zeros.
    t = re.sub(r"(?<![\d,])\d{5,}(?![\d,])", lambda m: " ".join(m.group(0).lstrip("0") or "0"), t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t[: config.TTS_MAX_CHARS]


async def stream(text: str):
    """Yield MP3 chunks as the voice service produces them, so playback can start right away."""
    spoken = clean_for_speech(text)
    if not spoken:
        return
    async for chunk in edge_tts.Communicate(spoken, pick_voice(spoken), rate=config.TTS_RATE).stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def synthesize(text: str) -> tuple[bytes, str]:
    """MP3 bytes for `text` and the voice used. Empty bytes if there is nothing to say."""
    spoken = clean_for_speech(text)
    voice = pick_voice(spoken)
    if not spoken:
        return b"", voice
    chunks: list[bytes] = []
    async for chunk in edge_tts.Communicate(spoken, voice, rate=config.TTS_RATE).stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks), voice
