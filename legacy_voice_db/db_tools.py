"""Read-only MongoDB tools exposed to the LLM as OpenAI-style function specs."""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "voicedb")

_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = _client[MONGODB_DB]

MAX_LIMIT = 20
FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$expr"}


# ---------- JSON helpers ----------

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, default=str)


# ---------- validation ----------

def _validate_filter(node: Any, depth: int = 0) -> None:
    if depth > 10:
        raise ValueError("filter is nested too deeply")
    if isinstance(node, dict):
        for key, val in node.items():
            if not isinstance(key, str):
                raise ValueError("filter keys must be strings")
            if key in FORBIDDEN_OPERATORS:
                raise ValueError(f"operator {key} is not allowed")
            _validate_filter(val, depth + 1)
    elif isinstance(node, list):
        for item in node:
            _validate_filter(item, depth + 1)


def _coerce_filter(flt: Any) -> dict:
    if flt is None or flt == "":
        return {}
    if isinstance(flt, str):
        flt = json.loads(flt)
    if not isinstance(flt, dict):
        raise ValueError("filter must be a JSON object")
    _validate_filter(flt)
    return flt


def _coerce_sort(sort: Any) -> list[tuple[str, int]] | None:
    if not sort:
        return None
    if isinstance(sort, str):
        sort = json.loads(sort)
    if isinstance(sort, dict):
        items = list(sort.items())
    elif isinstance(sort, list):
        items = [tuple(x) if isinstance(x, (list, tuple)) else (x, 1) for x in sort]
    else:
        raise ValueError('sort must be an object like {"field": -1}')
    out: list[tuple[str, int]] = []
    for field, direction in items:
        d = int(direction)
        if d not in (1, -1):
            raise ValueError("sort direction must be 1 or -1")
        out.append((str(field), d))
    return out or None


def _check_collection(name: Any) -> str:
    if not isinstance(name, str) or not name or name.startswith("system."):
        raise ValueError("invalid collection name")
    if name not in db.list_collection_names():
        raise ValueError(f"collection '{name}' does not exist")
    return name


# ---------- tools ----------

def list_collections() -> list[dict]:
    out = []
    for name in sorted(db.list_collection_names()):
        if name.startswith("system."):
            continue
        coll = db[name]
        out.append({
            "name": name,
            "count": coll.estimated_document_count(),
            "sample": coll.find_one(),
        })
    return out


def query(collection: str, filter: Any = None, limit: int = 10, sort: Any = None) -> dict:
    name = _check_collection(collection)
    flt = _coerce_filter(filter)
    try:
        lim = int(limit) if limit is not None else 10
    except (TypeError, ValueError):
        lim = 10
    lim = max(1, min(lim, MAX_LIMIT))
    cursor = db[name].find(flt)
    srt = _coerce_sort(sort)
    if srt:
        cursor = cursor.sort(srt)
    docs = list(cursor.limit(lim))
    return {"collection": name, "filter": flt, "returned": len(docs), "documents": docs}


def count(collection: str, filter: Any = None) -> dict:
    name = _check_collection(collection)
    flt = _coerce_filter(filter)
    return {"collection": name, "filter": flt, "count": db[name].count_documents(flt)}


def schema_summary(max_chars: int = 8000) -> str:
    """Compact text description of collections and their fields for the system prompt."""
    lines = []
    try:
        for c in list_collections():
            sample = c["sample"] or {}
            fields = ", ".join(f"{k} ({type(v).__name__})" for k, v in sample.items())
            lines.append(f"- {c['name']}: {c['count']} docs. fields: {fields}")
            lines.append(f"  example: {dumps(sample)}")
    except Exception as exc:  # database unreachable etc.
        lines.append(f"(schema unavailable: {exc})")
    text = "\n".join(lines)
    return text[:max_chars]


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_collections",
            "description": "List all collections with document counts and one sample document each.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Find documents in a collection using a MongoDB filter. Returns up to `limit` documents (max 20).",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection name."},
                    "filter": {
                        "type": "object",
                        "description": 'MongoDB filter document, e.g. {"status": "shipped"} or {"customer_name": {"$regex": "rahul", "$options": "i"}}. Use case-insensitive $regex for names.',
                    },
                    "limit": {"type": "integer", "description": "Max documents to return, 1-20. Default 10."},
                    "sort": {
                        "type": "object",
                        "description": 'Sort spec, e.g. {"price": -1}. Optional.',
                    },
                },
                "required": ["collection"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count",
            "description": "Count documents in a collection matching a MongoDB filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string", "description": "Collection name."},
                    "filter": {"type": "object", "description": "MongoDB filter document. Empty for all."},
                },
                "required": ["collection"],
            },
        },
    },
]

_TOOL_FUNCS = {
    "list_collections": lambda **kw: list_collections(),
    "query": lambda **kw: query(
        kw.get("collection"), kw.get("filter"), kw.get("limit", 10), kw.get("sort")
    ),
    "count": lambda **kw: count(kw.get("collection"), kw.get("filter")),
}


def run_tool(name: str, args: Any) -> str:
    """Execute a tool by name. Always returns a JSON string, never raises."""
    try:
        if isinstance(args, str):
            args = json.loads(args) if args.strip() else {}
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return dumps({"error": "arguments must be a JSON object"})
        func = _TOOL_FUNCS.get(name)
        if func is None:
            return dumps({"error": f"unknown tool '{name}'"})
        return dumps({"ok": True, "result": func(**args)})
    except Exception as exc:
        return dumps({"error": f"{type(exc).__name__}: {exc}"})
