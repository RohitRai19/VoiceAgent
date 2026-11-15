# Voice DB Assistant

A voice-only, multilingual assistant that answers questions about a MongoDB database.
Hold the mic button, ask in Hindi, Tamil, Telugu, English, Hinglish or any other language,
and hear the answer spoken back in the same language and script.

Pipeline: browser (MediaRecorder webm/opus) -> Groq Whisper (STT + language detection)
-> Groq LLM with MongoDB tools (read-only) -> edge-tts (Microsoft neural Indian voices) -> MP3 back to the browser.

## Requirements

- Python 3.11
- MongoDB running locally (or set `MONGODB_URI`)
- A Groq API key

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your GROQ_API_KEY in .env
python seed.py              # creates sample customers / products / orders
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000, allow microphone access, then hold the round button (or the spacebar) while you speak and release to send.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Groq API key |
| `GROQ_LLM_MODEL` | `openai/gpt-oss-120b` | Chat model with tool calling |
| `GROQ_STT_MODEL` | `whisper-large-v3` | Speech-to-text model |
| `MONGODB_URI` | `mongodb://localhost:27017` | Mongo connection string |
| `MONGODB_DB` | `voicedb` | Database name |

## Endpoints

- `GET /` serves the page
- `POST /voice/turn` multipart `audio` + `session_id` -> `{transcript, language, reply, voice, audio_b64, timings}`
- `POST /voice/reset` form `session_id` clears that session's history
- `GET /health` pings Mongo

Each turn logs one JSON line with `stt`, `llm`, `tts` and `total` seconds.

## Sample data

`seed.py` drops and recreates three collections with 5 documents each:

- `customers`: customer_id, name, city, phone, email, joined
- `products`: product_id, name, category, price, stock (one product has stock 0)
- `orders`: order_id, customer_id, customer_name, product_id, product_name, quantity, amount, status (pending / shipped / delivered / cancelled), ordered_at, ship_to

Point it at your own database by changing `MONGODB_URI` / `MONGODB_DB`. The LLM discovers collections and field names from a live schema summary, so no code changes are needed.

## Example questions

**Hindi**
- राहुल शर्मा का ऑर्डर कहाँ तक पहुँचा?
- कितने ऑर्डर अभी पेंडिंग हैं?
- कौन सा प्रोडक्ट स्टॉक में नहीं है?

**Tamil**
- பிரியா ஐயர் எந்த ஊரில் இருக்கிறார்?
- எத்தனை ஆர்டர்கள் டெலிவரி ஆகிவிட்டன?
- மிக விலை உயர்ந்த பொருள் எது?

**Hinglish**
- Rahul Sharma ka order kahan tak pahuncha?
- Kitne orders cancel hue hain aur kyun?
- Masala chai ka price kya hai?

**English**
- Which products are out of stock?
- How many customers do we have in total?
- Who placed the most recent order?

Follow-up questions work within a session, for example "and when will it arrive?" after asking about an order.

## Safety

The LLM can only call `list_collections`, `query` and `count`. Filters are validated and `$where`, `$function`, `$accumulator` and `$expr` are rejected. Query results are capped at 20 documents.
