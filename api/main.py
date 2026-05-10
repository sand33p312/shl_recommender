"""
SHL Assessment Recommender — FastAPI Service

Endpoints:
  GET  /health  -> {"status": "ok"}
  POST /chat    -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}

The service is fully stateless. Every /chat call must carry the full
conversation history. No per-conversation state is stored server-side.

Design notes:
  - Lifespan pre-loads the catalog at startup (avoids cold-path overhead).
  - Pydantic validates every request before the agent runs.
  - A request-ID header is injected for tracing (useful on Render logs).
  - The global exception handler ensures schema-valid responses on all errors.
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

from dotenv import load_dotenv
load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("shl_recommender.api")

# ─── Paths ────────────────────────────────────────────────────────────────────

HERE = Path(__file__).parent
CATALOG_PATH = str(HERE.parent / "catalog" / "catalog.json")

# ─── Lazy globals ─────────────────────────────────────────────────────────────

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        import sys
        sys.path.insert(0, str(HERE.parent))
        from catalog.retriever import CatalogRetriever
        _retriever = CatalogRetriever(CATALOG_PATH)
        logger.info("Catalog loaded: %d assessments", len(_retriever.assessments))
    return _retriever


# ─── Pydantic schemas ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("content must not be empty or whitespace")
        return stripped


class ChatRequest(BaseModel):
    messages: list[Message]

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatRequest":
        if not self.messages:
            raise ValueError("messages list must not be empty")
        if len(self.messages) > 20:
            raise ValueError("messages exceeds the maximum allowed length of 20")
        if self.messages[-1].role != "user":
            raise ValueError("the last message in the conversation must have role 'user'")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SHL Recommender starting up…")
    try:
        get_retriever()
        logger.info("Ready.")
    except Exception as exc:
        logger.error("Catalog failed to load on startup: %s", exc)
    yield
    logger.info("SHL Recommender shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent for recommending SHL Individual Test Solutions. "
        "Stateless — pass the full conversation history on every call."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Middleware: request tracing & timing ─────────────────────────────────────

@app.middleware("http")
async def trace_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "[%s] %s %s -> %d (%.0f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Readiness check")
async def health():
    """Returns HTTP 200 when the service is ready to receive traffic."""
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Run one conversational turn",
)
async def chat(request: ChatRequest):
    """
    Run a single turn of the assessment recommendation conversation.

    Pass the **full conversation history** on every call (the service is
    stateless). The agent will:

    - Ask a clarifying question if the query is too vague
    - Recommend 1-10 SHL assessments once it has enough context
    - Refine the shortlist when constraints change
    - Compare assessments when asked
    - Refuse off-topic requests politely
    """
    import sys
    sys.path.insert(0, str(HERE.parent))
    from agent.agent import run_agent

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
    retriever = get_retriever()

    try:
        result = run_agent(messages_dicts, retriever)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="The agent encountered an unexpected error. Please retry.",
        )

    # Enforce schema — cap at 10, validate field presence
    recs = [
        Recommendation(
            name=r.get("name", ""),
            url=r.get("url", ""),
            test_type=r.get("test_type", ""),
        )
        for r in result.get("recommendations", [])[:10]
    ]

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=recs,
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )


# ─── Global error handler ─────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ─── Local dev entrypoint ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
