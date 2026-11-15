"""MongoDB persistence: chat threads, their messages, and the list of indexed files."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient

import config

_client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
db = _client[config.MONGODB_DB]
threads = db["chat_threads"]
messages = db["chat_messages"]
files = db["rag_files"]

messages.create_index([("thread_id", ASCENDING), ("created_at", ASCENDING)])
threads.create_index([("updated_at", DESCENDING)])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:  # Mongo returns naive UTC datetimes
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


# ---------- threads ----------

def create_thread(title: str = "New chat") -> dict:
    doc = {"_id": str(uuid.uuid4()), "title": title, "created_at": _now(), "updated_at": _now(),
           "message_count": 0, "total_tokens": 0, "pinned": False}
    threads.insert_one(doc)
    return _thread_out(doc)


def get_thread(thread_id: str) -> dict | None:
    doc = threads.find_one({"_id": thread_id})
    return _thread_out(doc) if doc else None


def list_threads() -> list[dict]:
    # Pinned chats first, then most recently used.
    cur = threads.find().sort([("pinned", DESCENDING), ("updated_at", DESCENDING)]).limit(200)
    return [_thread_out(d) for d in cur]


def rename_thread(thread_id: str, title: str) -> None:
    threads.update_one({"_id": thread_id}, {"$set": {"title": title[:80]}})


def pin_thread(thread_id: str, pinned: bool) -> None:
    threads.update_one({"_id": thread_id}, {"$set": {"pinned": bool(pinned)}})


def delete_thread(thread_id: str) -> bool:
    messages.delete_many({"thread_id": thread_id})
    return threads.delete_one({"_id": thread_id}).deleted_count == 1


def _thread_out(doc: dict) -> dict:
    return {
        "thread_id": doc["_id"],
        "title": doc.get("title", "New chat"),
        "created_at": _iso(doc.get("created_at")),
        "updated_at": _iso(doc.get("updated_at")),
        "message_count": doc.get("message_count", 0),
        "total_tokens": doc.get("total_tokens", 0),
        "pinned": bool(doc.get("pinned", False)),
    }


# ---------- messages ----------

def add_message(thread_id: str, role: str, content: str, *, sources: list[dict] | None = None,
                usage: dict | None = None, run_id: str | None = None, duration_ms: int | None = None) -> dict:
    doc = {
        "_id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "usage": usage or {},
        "run_id": run_id,
        "duration_ms": duration_ms,
        "created_at": _now(),
    }
    messages.insert_one(doc)
    threads.update_one(
        {"_id": thread_id},
        {"$set": {"updated_at": doc["created_at"]},
         "$inc": {"message_count": 1, "total_tokens": int((usage or {}).get("total_tokens", 0))}},
    )
    return _message_out(doc)


def get_messages(thread_id: str, limit: int = 0) -> list[dict]:
    """Messages oldest-first. `limit` > 0 returns only the most recent `limit`."""
    cur = messages.find({"thread_id": thread_id}).sort("created_at", DESCENDING if limit else ASCENDING)
    if limit:
        return [_message_out(d) for d in reversed(list(cur.limit(limit)))]
    return [_message_out(d) for d in cur]


def _message_out(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        "thread_id": doc["thread_id"],
        "role": doc["role"],
        "content": doc["content"],
        "sources": doc.get("sources", []),
        "usage": doc.get("usage", {}),
        "run_id": doc.get("run_id"),
        "duration_ms": doc.get("duration_ms"),
        "created_at": _iso(doc.get("created_at")),
    }


# ---------- indexed files ----------

def add_file(file_id: str, name: str, size: int, chunks: int, sha256: str = "") -> dict:
    doc = {"_id": file_id, "name": name, "size": size, "chunks": chunks, "sha256": sha256, "uploaded_at": _now()}
    files.insert_one(doc)
    return _file_out(doc)


def find_files_by_name(name: str) -> list[dict]:
    return [{**_file_out(d), "sha256": d.get("sha256", "")} for d in files.find({"name": name})]


def list_files() -> list[dict]:
    return [_file_out(d) for d in files.find().sort("uploaded_at", DESCENDING)]


def delete_file(file_id: str) -> dict | None:
    doc = files.find_one_and_delete({"_id": file_id})
    return _file_out(doc) if doc else None


def _file_out(doc: dict) -> dict:
    return {"file_id": doc["_id"], "name": doc["name"], "size": doc["size"],
            "chunks": doc["chunks"], "uploaded_at": _iso(doc.get("uploaded_at"))}
