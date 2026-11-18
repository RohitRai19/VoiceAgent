# Chat with your files (RAG)

A chat app that answers from the files you upload, can work with Salesforce support Cases, and talks back in Hindi or English.

- **RAG**: files are split into chunks, embedded locally (fastembed, no API key) and stored in Chroma (`data/chroma/`). Each question retrieves the best-matching chunks and Groq answers from them, with citations.
- **Chat history with thread IDs**: every chat is a thread in MongoDB (`chat_threads`, `chat_messages`). Each message stores its sources, token usage and LangSmith run ID.
- **LangSmith tracing**: every turn is one trace (`rag_chat_turn`) containing the question rewrite, the retrieval and the LLM call with token counts. Traces carry the thread ID, so LangSmith's **Threads** tab shows whole conversations.
- **Voice in Hindi and English**: ask with the mic; the answer is written in the chat and spoken aloud live, sentence by sentence (Hindi voice for Hindi/Hinglish questions).
- **Salesforce (MCP)**: an MCP server with tools for support Cases and read-only SOQL, used by the chat when Salesforce keys are set.
- General questions are answered from the model's own knowledge (labelled), with a fallback model when Groq's limits are reached.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then fill in GROQ_API_KEY and LANGSMITH_API_KEY
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000, upload files in the left panel, then ask questions.
MongoDB must be running (`MONGODB_URI`). The first upload downloads the embedding model once.

### Docker

The image contains the app and the embedding model; keys are never baked in, they come from `.env` at runtime. `docker-compose.yml` runs the app together with its own MongoDB.

```powershell
docker compose up -d --build        # build and start app + MongoDB -> http://localhost:8080
docker compose logs -f app          # watch it start
docker compose down                 # stop (data is kept in the app-data / mongo-data volumes)
```

To use your existing local MongoDB instead of the bundled one, set `MONGODB_URI=mongodb://host.docker.internal:27017` in the `app` service.

Publish to Docker Hub:

```powershell
docker login
docker tag voiceagent:latest <dockerhub-username>/voiceagent:latest
docker push <dockerhub-username>/voiceagent:latest
```

Run the published image anywhere with `IMAGE=<dockerhub-username>/voiceagent:latest docker compose up -d` (a `.env` file with your keys must be next to `docker-compose.yml`).

### Deploy on a server (e.g. AWS EC2)

`docker-compose.prod.yml` runs **Caddy** (automatic HTTPS), the app and MongoDB. Only Caddy is reachable from the internet. HTTPS is required for the microphone to work in browsers.

1. Server: Ubuntu 24.04, x86_64, at least 2 GB RAM (e.g. EC2 `t3.small`), ports 22 (your IP), 80 and 443 open.
2. `git clone` this repo on the server and create `.env` there (same keys as locally) plus:
   ```
   DOMAIN=<your domain, e.g. your-name.duckdns.org>
   ```
3. `docker compose -f docker-compose.prod.yml up -d`, then open `https://<DOMAIN>`.

Update later: `docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d`.

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
| `GROQ_FALLBACK_MODEL` | `openai/gpt-oss-20b` | Used automatically when the main model fails, e.g. its daily token limit is used up. Separate Groq quota. Empty = off |
| `TITLE_MODEL` | `openai/gpt-oss-20b` | Writes the chat headings |
| `SF_LOGIN_URL`, `SF_CLIENT_ID`, `SF_CLIENT_SECRET` | empty | Salesforce Connected App (client credentials). When all three are set, Salesforce is used through them |
| `SF_ORG_ALIAS` | empty | Otherwise, a Salesforce CLI login alias for this app (`sf org login web -a <alias>`). All Salesforce settings empty = Salesforce off |
| `TTS_VOICE_HI` / `TTS_VOICE_EN` | `hi-IN-SwaraNeural` / `en-IN-NeerjaNeural` | Voices for spoken replies (Hindi answers / English answers). Any edge-tts voice name works, e.g. `hi-IN-MadhurNeural` for a male Hindi voice |
| `TTS_RATE` | `+0%` | Speaking speed, e.g. `+10%` |
| `ALLOW_GENERAL_KNOWLEDGE` | `true` | `true`: questions unrelated to the files are answered from the model's own knowledge, labelled "General knowledge, not from your files". `false`: every answer must come from the files |
| `EMBED_MODEL` | multilingual MiniLM | fastembed model. Changing it requires deleting `data/chroma/` and re-uploading |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `HISTORY_MESSAGES`, `MAX_UPLOAD_MB` | 4, 1000, 150, 10, 25 | Tuning |

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
- `POST /api/transcribe` (multipart `audio`) -> `{text}`; Groq Whisper (`GROQ_STT_MODEL`), used by the mic button only in browsers without built-in speech recognition

### Salesforce (MCP)

`salesforce_case_server.py` is this app's MCP server, with tools to create, read, search and update support Cases, add comments, list picklist values and run read-only SOQL. `mcp_client.py` starts it when the app starts and gives its tools to the chat model, so you can ask things like "show my latest cases", "what is the status of case 1069" or "open a case for order ORD-10004, the pan arrived dented". The model only creates or changes records when you ask it to; the Salesforce tools used are shown under each answer and traced in LangSmith.

It can also be used on its own from any MCP client (for example Claude Desktop): `python salesforce_case_server.py`.

### Voice input

**Spoken replies.** A question asked with the mic gets a short, phone-style answer: in Hindi (Devanagari) when the question is Hindi or Hinglish, otherwise in English. The answer is shown as text in the chat and read aloud live: each sentence is spoken as soon as the model has written it, while the rest is still being generated (Microsoft neural voices through `edge-tts`, no API key; the answer text is sent to Microsoft's speech service). IDs are read letter by letter and digit by digit ("O R D, 1 0 0 0 3"). Every answer also has a **Listen** button. `GET /api/tts/stream?text=...` streams the MP3 for one sentence; `POST /api/tts {text}` returns a whole MP3.

The mic button uses the browser's built-in speech recognition in Chrome and Edge: words appear in the box as you speak, and the question is sent after a 1.5 second pause (click the mic to send right away, Esc to cancel). The arrow next to the mic picks the speech language (Hindi by default, English India, English US). Other browsers record the audio and transcribe it with Groq Whisper instead, with browser noise suppression and a filter that drops low-confidence output from silence or background noise (`STT_MAX_NO_SPEECH`, `STT_MIN_LOGPROB`). Spoken IDs such as "o r d 1 0 0 1 3" or "SKU ELEC 003" are matched to `ORD-10013` / `SKU-ELEC-003`.
- `GET /health`

## Files

- `app.py` FastAPI routes, `chat.py` the traced chat turn, `rag.py` indexing and retrieval, `store.py` MongoDB, `config.py` settings, `static/index.html` the UI.
- `legacy_voice_db/` the earlier voice assistant over MongoDB, kept for reference and not used by this app.
