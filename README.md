# Chat with your files (RAG)

A chat app that answers only from the files you upload.

- **RAG**: files are split into chunks, embedded locally (fastembed, no API key) and stored in Chroma (`data/chroma/`). Each question retrieves the best-matching chunks and Groq answers from them, with citations.
- **Chat history with thread IDs**: every chat is a thread in MongoDB (`chat_threads`, `chat_messages`). Each message stores its sources, token usage and LangSmith run ID.
- **LangSmith tracing**: every turn is one trace (`rag_chat_turn`) containing the question rewrite, the retrieval and the LLM call with token counts. Traces carry the thread ID, so LangSmith's **Threads** tab shows whole conversations.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then fill in GROQ_API_KEY and LANGSMITH_API_KEY
uvicorn app:app --reload --port 8000     # or double-click run.bat
```

Open http://localhost:8000, upload files in the left panel, then ask questions.
MongoDB must be running (`MONGODB_URI`). The first upload downloads the embedding model once.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Groq API key |
| `GROQ_LLM_MODEL` | `openai/gpt-oss-120b` | Chat model |
| `LANGSMITH_API_KEY` | empty | LangSmith key. Tracing is off until this is set |
| `LANGSMITH_TRACING` | `true` | Set `false` to switch tracing off |
| `LANGSMITH_PROJECT` | `voiceagent-rag-chat` | LangSmith project that receives the traces |
| `LANGSMITH_ENDPOINT` | US region | Set to `https://eu.api.smith.langchain.com` for EU accounts |
| `MONGODB_URI` / `MONGODB_DB` | `mongodb://localhost:27017` / `voicedb` | Where threads and messages are saved |
| `EMBED_MODEL` | multilingual MiniLM | fastembed model. Changing it requires deleting `data/chroma/` and re-uploading |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `HISTORY_MESSAGES`, `MAX_UPLOAD_MB` | 5, 1000, 150, 10, 25 | Tuning |

Supported files: PDF (text-based, not scanned), DOCX, CSV, TXT, MD, JSON, HTML and common code/config files.

### Knowledge folder

Any supported file placed in `knowledge/` is indexed automatically when the server starts. Editing a file there and restarting re-indexes it. `knowledge/rag_orders_knowledge_base.json` is the current knowledge base.

### How JSON files are indexed

For every array of objects in a JSON file (for example `orders`, `inventory`) the app stores:

- one chunk per record with full detail, found by meaning or by an exact ID in the question (`ORD-10003`, `SKU-ELEC-001`, a tracking number);
- a compact overview table of the whole array, sent to the model for listing, counting and totals so it sees every record.

Small sections such as `policies` are sent with every question. Top-level keys listed in `JSON_SKIP_KEYS` (default `sample_questions_the_agent_can_answer`) are not indexed. `PIN_OVERVIEW_CHARS` (default 16000) caps how much overview text may be sent; `MAX_ANSWER_TOKENS` (default 6000) caps the answer including the model's reasoning.

Questions about one order use about 1,800 tokens; listing questions over all orders use about 5,000. On Groq's free tier the per-minute token limit can make back-to-back listing questions wait 20 to 40 seconds.

## API

- `POST /api/chat` `{message, thread_id?}` -> server-sent events `meta`, `sources`, `token`, `done` / `error`. Omit `thread_id` to start a new thread.
- `GET /api/threads`, `POST /api/threads`, `GET|PATCH|DELETE /api/threads/{thread_id}`
- `GET /api/files`, `POST /api/files` (multipart `files`), `DELETE /api/files/{file_id}`
- `GET /health`

## Files

- `app.py` FastAPI routes, `chat.py` the traced chat turn, `rag.py` indexing and retrieval, `store.py` MongoDB, `config.py` settings, `static/index.html` the UI.
- `legacy_voice_db/` the earlier voice assistant over MongoDB, kept for reference and not used by this app.
