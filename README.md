# SHL Assessment Recommender

Conversational API that helps hiring managers find the right SHL assessments through dialogue.

Built for the SHL Labs AI Intern take-home assignment.

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/sand33p312/shl_recommender.git
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key (free at https://aistudio.google.com/app/apikey)
cp .env.example .env
# edit .env → GEMINI_API_KEY=...

# 4. Run
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API

### `GET /health`
```bash
curl http://localhost:8000/health
```
Returns `{"status": "ok"}`

### `POST /chat`

**Vague query (will ask clarifying question):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'
```

**Clear query (returns recommendations):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a mid-level Java developer"}]}'
```

**Multi-turn refinement:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}, {"role": "assistant", "content": "What level?"}, {"role": "user", "content": "Senior, also add personality test"}]}'
```

**Compare two assessments:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Compare OPQ32r and Numerical Reasoning"}]}'
```

- `recommendations` is `[]` while clarifying or refusing.
- `end_of_conversation` is `true` only when the user says they are done.
- The service is **stateless** — pass the full history on every call.

---

## Project Structure

```
shl_recommender/
├── catalog/
│   ├── catalog.json         # 192 SHL Individual Test Solutions
│   └── retriever.py         # TF-IDF cosine similarity search
├── agent/
│   └── agent.py             # Conversational agent + Gemini integration
├── api/
│   └── main.py              # FastAPI service
├── tests/
│   └── test_all.py          # 32 tests (retriever, behavior probes, schema, recall)
├── requirements.txt
├── Procfile                 # Render / Heroku start command
├── render.yaml              # One-click Render deployment
└── .env.example
```

---

## Running Tests

```bash
python tests/test_all.py
```

All 32 tests should pass in ~1 second (LLM calls are mocked).

---

## Deployment (Render)

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo.
3. Render will detect `render.yaml` automatically.
4. Set the `GEMINI_API_KEY` environment variable in the Render dashboard.
5. Deploy. The `/health` endpoint will confirm readiness.

---

## Design Decisions

See `approach_document.pdf` (or `approach_document.md`) for the full 2-page writeup.

| Decision | Choice | Rationale |
|---|---|---|
| LLM | gemini-flash-latest | Free tier, ~1–2s latency, 1M context window |
| Retrieval | TF-IDF (numpy) | No heavy deps, deterministic, debuggable |
| Framework | Raw FastAPI | Shows genuine understanding over LangChain abstraction |
| State | Fully stateless | Caller owns history; simpler ops |
| Hallucination guard | URL whitelist + name fuzzy-match | Every returned URL verified against catalog |
