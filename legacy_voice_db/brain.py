"""LLM tool-calling loop over the MongoDB tools using the Groq SDK."""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq

import db_tools

load_dotenv()

log = logging.getLogger("brain")

GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
MAX_ROUNDS = 6

_client: Groq | None = None


def _groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


SYSTEM_PROMPT = """You are a voice assistant that answers questions about a company's MongoDB database. The user is speaking to you and will HEAR your reply read aloud by a text-to-speech voice.

LANGUAGE RULES (most important):
- Reply in exactly the language AND script the user used. If they spoke Hindi in Devanagari, reply in Devanagari Hindi. If they spoke Tamil, reply in Tamil script. If they spoke Hinglish (Hindi words written in Latin letters), reply in Hinglish using Latin letters, not Devanagari. If they spoke English, reply in English.
- Whisper detected the spoken language as: {language}. Use that as a hint, but the transcript's own script and wording decide the reply language.
- Never translate names of people, products or cities into another script unless the user did.

STYLE RULES:
- 1 to 3 short spoken sentences. This is speech, not text.
- No markdown, no bullet points, no numbered lists, no emojis, no URLs, no code, no JSON.
- Say numbers the way a person would say them aloud (for example "twelve hundred rupees", "paanch orders", "pending").
- Be direct. Do not say "according to the database" every time.

DATA RULES:
- Answer ONLY using the database tools. Never invent, guess or embellish records, names, prices or statuses.
- Always call a tool before answering a factual question. If a name is given, search with a case-insensitive $regex on the relevant name field. Try partial names (first name only) if a full-name match returns nothing.
- Join across collections yourself: for example find the customer first, then query orders with their customer id or name.
- If nothing matches, say so plainly in the user's language and offer what you can look up instead.
- If the question is not about the database, briefly say you can only answer questions about the data.

LIVE DATABASE SCHEMA (collection: count, fields and one example document):
{schema}
"""


def _build_system(language: str) -> str:
    return SYSTEM_PROMPT.format(language=language or "unknown", schema=db_tools.schema_summary())


def answer(transcript: str, language: str, history: list[dict] | None = None) -> str:
    """Run the tool-calling loop and return the final assistant text.

    `history` is a list of prior {"role": "user"/"assistant", "content": str} messages.
    """
    messages: list[dict] = [{"role": "system", "content": _build_system(language)}]
    for m in history or []:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": transcript})

    client = _groq()
    for round_no in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=messages,
            tools=db_tools.TOOL_SPECS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=600,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            text = (msg.content or "").strip()
            if text:
                return text
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result = db_tools.run_tool(tc.function.name, tc.function.arguments)
            log.info("tool %s(%s) -> %d chars", tc.function.name, tc.function.arguments, len(result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # Out of rounds or empty reply: force a final answer without tools.
    messages.append({
        "role": "user",
        "content": "Now give your final spoken answer in the user's language using only what the tools returned.",
    })
    resp = client.chat.completions.create(
        model=GROQ_LLM_MODEL, messages=messages, temperature=0.2, max_tokens=300
    )
    return (resp.choices[0].message.content or "").strip() or "Sorry, I could not find that in the database."


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How many orders are pending?"
    print(answer(q, "english"))
