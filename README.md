---
title: SHL Recommender
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SHL Assessment Recommender

Conversational API that helps hiring managers find the right SHL assessments through dialogue.

Built for the SHL Labs AI Intern take-home assignment.

---

## Live API

Base URL: `https://sk132-shl-recommender.hf.space`

- Health: `https://sk132-shl-recommender.hf.space/health`
- Chat: `https://sk132-shl-recommender.hf.space/chat`

---

## API

### `GET /health`
```bash
curl https://sk132-shl-recommender.hf.space/health
```
Returns `{"status": "ok"}`

### `POST /chat`

**Vague query (will ask clarifying question):**
```bash
curl -X POST https://sk132-shl-recommender.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment"}]}'
```

**Clear query (returns recommendations):**
```bash
curl -X POST https://sk132-shl-recommender.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a mid-level Java developer"}]}'
```

**Multi-turn refinement:**
```bash
curl -X POST https://sk132-shl-recommender.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}, {"role": "assistant", "content": "What level?"}, {"role": "user", "content": "Senior, also add personality test"}]}'
```

**Compare two assessments:**
```bash
curl -X POST https://sk132-shl-recommender.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Compare OPQ32r and Numerical Reasoning"}]}'
```

**Off-topic refusal:**
```bash
curl -X POST https://sk132-shl-recommender.hf.space/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the weather in Delhi?"}]}'
```

---

## Response Schema

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` while clarifying or refusing
- `end_of_conversation` is `true` only when user says they are done
- Service is **stateless** — pass full history on every call

---

## Quick Start (Local)

```bash
git clone https://github.com/sand33p312/shl_recommender.git
cd shl_recommender
pip install -r requirements.txt
cp .env.example .env
# edit .env → GEMINI_API_KEY=your_key
uvicorn api.main:app --reload --port 8000
```

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


## Design Decisions

See `approach_document.pdf` (or `approach_document.md`) for the full 2-page writeup.

| Decision | Choice | Rationale |
|---|---|---|
| LLM | gemini-flash-latest | Free tier, fast, fits 30s timeout |
| Retrieval | TF-IDF (numpy) | No heavy deps, deterministic, debuggable |
| Framework | Raw FastAPI | Shows genuine understanding over LangChain abstraction |
| State | Fully stateless | Caller owns history; simpler ops |
| Hallucination guard | URL whitelist + name fuzzy-match | Every returned URL verified against catalog |
