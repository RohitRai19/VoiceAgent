"""RAG: read files, split into chunks, embed locally (fastembed) and search with Chroma."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import threading
import uuid
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

import config

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".log", ".html", ".htm", ".xml", ".yaml", ".yml",
                   ".py", ".js", ".ts", ".java", ".sql", ".rst", ".ini", ".cfg"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx", ".csv"}

# Code-like tokens: contain a digit, e.g. ORD-10003, SKU-ELEC-001, DLV8812345671
_CODE_RE = re.compile(r"\b(?=[A-Za-z0-9\-_]*\d)[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b")
# Spoken / dictated IDs: "SKU ELEC 003" -> "SKU-ELEC-003", "ORD 10013" -> "ORD-10013".
# "ORD 10001", "sku elec 1001", and also "ORD10001" (live speech-to-text joins them); never "order 10001".
_SPOKEN_ID_RE = re.compile(r"\b(sku|ord)(?:\s*[- ]\s*(?:(?!ord\b)([a-z]{2,5})\s*[- ]\s*)?|(?=\d))(\d{3,})\b", re.IGNORECASE)
# Speech engines often space out digits and letters: "1 0 0 1 3", "o r d".
_SPACED_DIGITS_RE = re.compile(r"\b\d(?:\s\d){2,}\b")
_SPACED_ORD_RE = re.compile(r"\bo\.?\s*r\.?\s*d\.?(?=\s|-|\d|$)", re.IGNORECASE)


def normalize_ids(text: str) -> str:
    text = _SPACED_DIGITS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = _SPACED_ORD_RE.sub("ORD", text)
    return _SPOKEN_ID_RE.sub(lambda m: "-".join(p for p in m.groups() if p).upper(), text)

_lock = threading.Lock()
_embedder: TextEmbedding | None = None
_collection = None

_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)


def _get_embedder() -> TextEmbedding:
    """Loaded lazily; the first call downloads the model (a few hundred MB) once."""
    global _embedder
    with _lock:
        if _embedder is None:
            # In Docker the model is baked into FASTEMBED_CACHE_PATH; locally fastembed's default cache is used.
            cache = os.getenv("FASTEMBED_CACHE_PATH")
            _embedder = TextEmbedding(model_name=config.EMBED_MODEL, **({"cache_dir": cache} if cache else {}))
        return _embedder


def _get_collection():
    global _collection
    with _lock:
        if _collection is None:
            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            _collection = client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})
        return _collection


def embed(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_embedder().embed(texts)]


# ---------- reading files ----------

def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_sections(filename: str, data: bytes) -> list[tuple[str, dict]]:
    """Return [(text, metadata)] for a file. PDFs give one section per page."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(page.extract_text() or "", {"page": i + 1}) for i, page in enumerate(reader.pages)]
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return [("\n".join(parts), {})]
    if ext == ".csv":
        rows = list(csv.reader(io.StringIO(_decode(data))))
        if not rows:
            return []
        header, body = rows[0], rows[1:]
        # Repeat the header on every row so each chunk is understandable on its own.
        lines = ["; ".join(f"{h}: {v}" for h, v in zip(header, row)) for row in body]
        return [("\n".join(lines), {})]
    if ext == ".json":
        try:
            return _json_sections(json.loads(_decode(data)))
        except json.JSONDecodeError:
            return [(_decode(data), {})]  # not valid JSON: index it as plain text
    if ext in TEXT_EXTENSIONS:
        return [(_decode(data), {})]
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


# ---------- structured JSON ----------
# Character-splitting JSON cuts records in half and makes "list all ..." questions
# unanswerable from a handful of chunks. Instead, for every array of objects we store:
#   - one chunk per record (full detail, found by similarity or by an ID in the question)
#   - a compact overview table of the whole array (always given to the LLM when small
#     enough, so counting / listing / totals see every record)

INDEX_VERSION = "2"  # bump when chunking changes, so files are re-indexed on the next start
MAX_ID_HITS = 3
MAX_RECORD_CHARS = 3500
TABLE_PART_CHARS = 6000
SMALL_SECTION_CHARS = 1500
INLINE_LIST_CHARS = 120


def _is_scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v).replace("|", "/").replace("\n", " ").strip()


def _render(value, indent: int = 0) -> str:
    """Readable text for any JSON value (one fact per line)."""
    pad = "  " * indent
    if _is_scalar(value):
        return pad + (_cell(value) or "none")
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict) and all(_is_scalar(x) for x in item.values()):
                lines.append(pad + "- " + ", ".join(f"{k}: {_cell(x) or 'none'}" for k, x in item.items()))
            elif _is_scalar(item):
                lines.append(pad + "- " + (_cell(item) or "none"))
            else:
                lines.append(pad + "-\n" + _render(item, indent + 1))
        return "\n".join(lines)
    lines = []
    for k, v in value.items():
        if _is_scalar(v):
            lines.append(f"{pad}{k}: {_cell(v) or 'none'}")
        elif isinstance(v, dict) and all(_is_scalar(x) for x in v.values()):
            lines.append(f"{pad}{k}: " + ", ".join(f"{sk}={_cell(sv) or 'none'}" for sk, sv in v.items()))
        else:
            lines.append(f"{pad}{k}:\n" + _render(v, indent + 1))
    return "\n".join(lines)


def _inline_list(items: list) -> str:
    return "; ".join("/".join(_cell(x) for x in it.values()) if isinstance(it, dict) else _cell(it) for it in items)


def _table_parts(name: str, records: list[dict]) -> list[str]:
    """Compact table of an array of objects, split into parts of about TABLE_PART_CHARS."""
    columns: list[tuple[str, str, str | None]] = []  # (header, key, subkey)
    seen: set[str] = set()
    for rec in records:
        for key, val in rec.items():
            if isinstance(val, dict):
                for sub, sv in val.items():
                    if _is_scalar(sv) and f"{key}.{sub}" not in seen:
                        seen.add(f"{key}.{sub}")
                        columns.append((f"{key}.{sub}", key, sub))
            elif key not in seen:
                if isinstance(val, list):
                    lists = [r.get(key) for r in records if isinstance(r.get(key), list)]
                    if sum(len(_inline_list(x)) for x in lists) / max(len(lists), 1) > INLINE_LIST_CHARS:
                        seen.add(key)  # too long for a table (e.g. histories); kept in the record chunks
                        continue
                    first = next((x[0] for x in lists if x), None)
                    header = f"{key}({'/'.join(first)})" if isinstance(first, dict) else key
                    columns.append((header, key, None))
                else:
                    columns.append((key, key, None))
                seen.add(key)

    def row(rec: dict) -> str:
        cells = []
        for _, key, sub in columns:
            val = rec.get(key)
            if sub is not None:
                val = val.get(sub) if isinstance(val, dict) else None
            cells.append(_inline_list(val) if isinstance(val, list) else _cell(val))
        return " | ".join(cells)

    rows = [row(r) for r in records]
    groups: list[list[str]] = [[]]
    size = 0
    for r in rows:
        if groups[-1] and size + len(r) > TABLE_PART_CHARS:
            groups.append([])
            size = 0
        groups[-1].append(r)
        size += len(r) + 1
    parts = []
    for i, group in enumerate(groups, 1):
        part = f" (part {i} of {len(groups)}, {len(group)} rows)" if len(groups) > 1 else ""
        parts.append(
            f"Overview table of ALL {len(records)} {name}{part}. One row per record, an empty cell means none. "
            f"Use this table for counting, listing, filtering and totals.\n"
            f"Columns: {' | '.join(h for h, _, _ in columns)}\n" + "\n".join(group)
        )
    return parts


def _json_sections(obj) -> list[tuple[str, dict]]:
    if isinstance(obj, list):
        obj = {"records": obj}
    if not isinstance(obj, dict):
        return [(str(obj), {})]
    out: list[tuple[str, dict]] = []
    for key, val in obj.items():
        if key in config.JSON_SKIP_KEYS:
            continue
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            for part in _table_parts(key, val):
                out.append((part, {"kind": "overview", "table": True, "section": key, "_whole": True}))
            for rec in val:
                text = f"{key} record\n{_render(rec)}"
                out.append((text, {"kind": "record", "section": key, "_whole": len(text) <= MAX_RECORD_CHARS}))
        else:
            text = f"{key}\n{_render(val, 1)}"
            small = len(text) <= SMALL_SECTION_CHARS
            out.append((text, {"kind": "overview" if small else "text", "section": key, "_whole": small}))
    return out


# ---------- indexing ----------

def index_file(filename: str, data: bytes) -> tuple[str, int]:
    """Chunk, embed and store a file. Returns (file_id, number_of_chunks)."""
    global _vocab
    _vocab = None
    file_id = str(uuid.uuid4())
    texts: list[str] = []
    metas: list[dict] = []
    for text, meta in extract_sections(filename, data):
        meta = {"kind": "text", **meta}
        whole = meta.pop("_whole", False)
        for chunk in ([text] if whole else _splitter.split_text(text)):
            if chunk.strip():
                texts.append(chunk)
                metas.append({"file_id": file_id, "source": filename, "chunk": len(texts), **meta})
    if not texts:
        raise ValueError("No readable text found in this file (scanned/image-only PDFs are not supported).")

    col = _get_collection()
    for start in range(0, len(texts), 128):
        batch = texts[start:start + 128]
        col.add(
            ids=[f"{file_id}:{start + i}" for i in range(len(batch))],
            documents=batch,
            embeddings=embed(batch),
            metadatas=metas[start:start + 128],
        )
    return file_id, len(texts)


def delete_file(file_id: str) -> None:
    global _vocab
    _get_collection().delete(where={"file_id": file_id})
    _vocab = None


# ---------- "is this question about the files?" ----------
# Names, products, couriers, cities, statuses... taken from short values in the indexed JSON
# records. A question mentioning one of them is treated as a question about the files.
_vocab: set[str] | None = None
_VOCAB_STOP = {"none", "applicable", "online", "paid", "with", "from", "this", "that", "your", "have",
               "will", "after", "before", "original", "payment", "method", "customer", "address"}
_WORD_RE = re.compile(r"[a-zऀ-ॿ]{4,}")


def known_terms() -> set[str]:
    global _vocab
    if _vocab is None:
        words: set[str] = set()
        records = _get_collection().get(where={"kind": "record"}, include=["documents"])["documents"]
        for doc in records:
            for line in doc.splitlines():
                key, sep, value = line.partition(":")
                value = value.strip()
                if sep and 0 < len(value) <= 60 and not key.strip().startswith("-"):
                    words.update(_WORD_RE.findall(value.lower()))
        _vocab = words - _VOCAB_STOP
    return _vocab


def mentions_known_terms(text: str) -> bool:
    return bool(set(_WORD_RE.findall(text.lower())) & known_terms())


def chunk_count() -> int:
    return _get_collection().count()


# ---------- retrieval ----------

@traceable(run_type="retriever", name="retrieve_chunks")
def retrieve(query: str, k: int = config.TOP_K, pin_tables: bool = True) -> list[dict]:
    """Top-k chunks for `query`, in the document shape LangSmith renders for retriever runs."""
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []

    docs: list[dict] = []
    seen: set[str] = set()

    def add(doc_id: str, text: str, meta: dict | None, score: float | None, match: str) -> None:
        if doc_id in seen:
            return
        seen.add(doc_id)
        docs.append({"page_content": text, "type": "Document",
                     "metadata": {**(meta or {}), "score": score, "match": match}})

    ov = col.get(where={"kind": "overview"}, include=["documents", "metadatas"])
    pinned = 0 < sum(len(d) for d in ov["documents"]) <= config.PIN_OVERVIEW_CHARS
    where = {"kind": {"$ne": "overview"}} if pinned else None

    # 1. Exact IDs in the question (ORD-10003, SKU-ELEC-001, tracking numbers): embeddings
    #    cannot tell such codes apart, so fetch the chunks that literally contain them.
    #    A code found in many chunks (a date, a common number) is not an ID and is ignored.
    exact: list[tuple[str, str, dict]] = []
    # Spoken IDs arrive with spaces ("SKU ELEC 003", "ORD 10013"): join them into "SKU-ELEC-003".
    id_query = normalize_ids(query)
    for code in [t for t in dict.fromkeys(_CODE_RE.findall(id_query)) if len(t) >= 4][:5]:
        # Speech and quick typing drop the hyphen ("ORD10003"), so also try "ORD-10003"
        # and the bare number.
        upper = code.upper()
        hyphenated = re.sub(r"(?<=[A-Z])(?=\d)", "-", upper)
        number = max(re.findall(r"\d{5,}", code), key=len, default=None)
        for variant in dict.fromkeys(v for v in (code, upper, hyphenated, number) if v):
            hit = col.get(where_document={"$contains": variant}, where=where, limit=MAX_ID_HITS + 1,
                          include=["documents", "metadatas"])
            if 0 < len(hit["ids"]) <= MAX_ID_HITS:
                exact.extend(zip(hit["ids"], hit["documents"], hit["metadatas"]))

    # 2. Semantic search (fewer chunks when exact records or overview tables carry the answer).
    n_semantic = k if not pinned else max(2, k // 2)
    res = col.query(query_embeddings=embed([query]), n_results=min(k, total), where=where,
                    include=["documents", "metadatas", "distances"])
    semantic = list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))

    # 3. Overview chunks. Small sections (policies, metadata) always go in. Full tables are
    #    large, so they are only sent for questions without a specific ID (listing, counting,
    #    totals, lookups by name), and only for the sections the search actually matched.
    if pinned:
        sections = set() if (exact or not pin_tables) else {(m or {}).get("section") for _, _, m, _ in semantic}
        order = sorted(range(len(ov["ids"])), key=lambda i: (ov["metadatas"][i]["file_id"], ov["metadatas"][i]["chunk"]))
        for i in order:
            meta = ov["metadatas"][i]
            if not meta.get("table") or meta.get("section") in sections:
                add(ov["ids"][i], ov["documents"][i], meta, None, "overview")

    for doc_id, text, meta in exact:
        add(doc_id, text, meta, 1.0, "exact id")
    for doc_id, text, meta, dist in semantic[:n_semantic]:
        add(doc_id, text, meta, round(1 - float(dist), 4), "semantic")
    return docs
