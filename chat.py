"""One chat turn: rewrite follow-up -> retrieve chunks -> answer with Groq. Traced in LangSmith."""
from __future__ import annotations

import re
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

import config
import rag

_llm: ChatGroq | None = None
_CITATION_RE = re.compile(r"【(\d+)[^】]*】")


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=config.GROQ_LLM_MODEL, api_key=config.GROQ_API_KEY,
                        temperature=0.2, max_tokens=config.MAX_ANSWER_TOKENS)
    return _llm


ANSWER_PROMPT = """You are an assistant that answers questions using the user's own uploaded files.

Rules:
- Answer ONLY from the numbered context excerpts below. Do not use outside knowledge and do not guess.
- If the excerpts do not contain the answer, say you could not find it in the uploaded files, and mention what related information is available if any. Never invent an order, product, date, amount or status.
- Excerpts titled "Overview table" list EVERY record. Use them for counting, listing, filtering and totals, and check every row before answering. Use the detailed record excerpts for specifics such as tracking history.
- When you add or multiply numbers, show the figures you used so the result can be checked.
- If the files contain a "last updated" date, treat it as today when a question depends on the current date.
- Cite the excerpts you used with plain square brackets only, like [1] or [2]. No other citation format, no line numbers.
- Reply in the same language AND script the user wrote in. If they wrote Hindi in Latin letters (Hinglish), reply in Hinglish with Latin letters, not Devanagari.
- Be concise and use markdown when it helps (lists, tables, code).

Context excerpts:
{context}
"""

CONDENSE_PROMPT = """Rewrite the user's latest message as one standalone search query that makes sense without the conversation. Keep the user's language. Resolve words like "it", "that", "he", "this one" using the conversation. If the message is already standalone, return it unchanged. Output only the query, nothing else."""

NO_CONTEXT = "(no files have been uploaded yet, or nothing relevant was found)"


def _usage(msg) -> dict:
    u = getattr(msg, "usage_metadata", None) or {}
    return {"input_tokens": int(u.get("input_tokens", 0)), "output_tokens": int(u.get("output_tokens", 0)),
            "total_tokens": int(u.get("total_tokens", 0))}


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")}


def _history_messages(history: list[dict]) -> list:
    out = []
    for m in history:
        if m["role"] == "user":
            out.append(HumanMessage(m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(m["content"]))
    return out


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return NO_CONTEXT
    blocks = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        where = meta.get("source", "file") + (f", page {meta['page']}" if meta.get("page") else "")
        blocks.append(f"[{i}] ({where})\n{d['page_content']}")
    return "\n\n".join(blocks)


def _sources(docs: list[dict]) -> list[dict]:
    return [{
        "n": i,
        "source": d["metadata"].get("source", "file"),
        "page": d["metadata"].get("page"),
        "section": d["metadata"].get("section"),
        "match": d["metadata"].get("match"),
        "score": d["metadata"].get("score"),
        "snippet": d["page_content"][:400],
    } for i, d in enumerate(docs, 1)]


def _trace_inputs(inputs: dict) -> dict:
    # Keep the trace readable: drop the callback, show how much history was sent.
    return {"thread_id": inputs.get("thread_id"), "question": inputs.get("question"),
            "history_messages": len(inputs.get("history") or [])}


@traceable(run_type="chain", name="rag_chat_turn", process_inputs=_trace_inputs)
def run_turn(thread_id: str, question: str, history: list[dict],
             on_event: Callable[[str, dict], None] | None = None) -> dict:
    """Answer `question` for a thread. `on_event(kind, data)` receives 'sources' and 'token' events."""
    emit = on_event or (lambda kind, data: None)
    llm = _get_llm()
    past = _history_messages(history)
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # 1. Follow-ups ("and what about the second one?") need the history to be searchable.
    search_query = question
    if past:
        rewritten = llm.invoke([SystemMessage(CONDENSE_PROMPT), *past, HumanMessage(question)],
                               config={"run_name": "rewrite_question"})
        usage = _add_usage(usage, _usage(rewritten))
        search_query = (rewritten.content or "").strip() or question

    # 2. Retrieve
    docs = rag.retrieve(search_query)
    sources = _sources(docs)
    emit("sources", {"sources": sources, "search_query": search_query})

    # 3. Answer, streaming tokens as they arrive
    prompt = [SystemMessage(ANSWER_PROMPT.format(context=_format_context(docs))), *past, HumanMessage(question)]
    full = None
    for chunk in llm.stream(prompt, config={"run_name": "answer"}):
        full = chunk if full is None else full + chunk
        if chunk.content:
            emit("token", {"text": chunk.content})
    answer = ((full.content if full is not None else "") or "").strip()
    answer = _CITATION_RE.sub(r"[\1]", answer)  # the model sometimes writes 【2†L7-L9】 instead of [2]
    if full is not None:
        usage = _add_usage(usage, _usage(full))
    if not answer:
        answer = "Sorry, I could not generate an answer. Please try again."
        emit("token", {"text": answer})
    elif full is not None and (full.response_metadata or {}).get("finish_reason") == "length":
        note = "\n\n_(This answer was cut off at the length limit. Ask a narrower question, or raise MAX_ANSWER_TOKENS in .env.)_"
        answer += note
        emit("token", {"text": note})

    run = get_current_run_tree()
    return {
        "answer": answer,
        "sources": sources,
        "usage": usage,
        "search_query": search_query,
        "run_id": str(run.id) if run is not None else None,
    }
