"""
FastAPI wrapper around the LangGraph agent.

Endpoints
  POST /investigate  run the agent on an entity-category profile (or, when
                     DEMO_MODE=prerecorded, replay a saved response)
  GET  /examples     example queries the frontend can offer
  GET  /health       liveness (no dependencies touched)
  GET  /ready        configuration/index readiness, safe to expose (no secrets)

Run from the repo root:  uvicorn api.main:app --port 8000

Index building and ingestion are deliberately NOT triggered by requests;
they are offline steps (see README). A request against a missing index
returns 503 with the fix, rather than building anything.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.graph import (
    GuardrailViolation,
    IndexNotFoundError,
    UpstreamLLMError,
    build_graph,
    check_guardrail,
    initial_state,
)
from config import ConfigurationError, Settings

log = logging.getLogger("api")

VERSION = "0.2.0"
MIN_QUERY_CHARS = 10
MAX_QUERY_CHARS = 300
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class InvestigateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Name kept for backward compatibility with the original API. In this
    # build it is a free-text entity-category profile, e.g. "cryptocurrency
    # exchange with prior AML actions" -- no external clustering system
    # supplies real cluster IDs.
    entity_cluster_id: str = Field(
        description="Free-text entity-category profile to investigate.",
        examples=["money services business with prior consent order for AML violations"],
    )

    @field_validator("entity_cluster_id")
    @classmethod
    def _clean(cls, v: str) -> str:
        if _CONTROL_CHARS.search(v):
            raise ValueError("must not contain control characters")
        v = " ".join(v.split())
        if not (MIN_QUERY_CHARS <= len(v) <= MAX_QUERY_CHARS):
            raise ValueError(
                f"must be {MIN_QUERY_CHARS}-{MAX_QUERY_CHARS} characters after trimming"
            )
        return v


class Source(BaseModel):
    index: int
    doc_id: str | None
    source: str | None
    title: str | None
    url: str | None
    excerpt: str
    cited: bool


class InvestigateResponse(BaseModel):
    entity_cluster_id: str
    report: str
    citations: list[str]  # doc_ids the report actually cites
    sources: list[Source]  # all retrieved excerpts, flagged cited / not cited
    token_usage: dict
    mode: Literal["live", "prerecorded"]
    notice: str | None = None  # set for prerecorded responses


class SlidingWindowRateLimiter:
    """In-memory per-client limiter. Fine for a single instance; with several
    instances each keeps its own counts (use a shared store such as Redis, or
    an edge/API-gateway limit, if that matters)."""

    def __init__(self, limit: int, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Returns None if allowed, else seconds until the client may retry."""
        if self.limit <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(self.window - (now - hits[0]), 0.0)
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory under many distinct clients
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[k]
            return None


def _normalize(q: str) -> str:
    return " ".join(q.split()).casefold()


def load_prerecorded(settings: Settings) -> dict:
    try:
        data = json.loads(settings.prerecorded_path.read_text())
    except FileNotFoundError as e:
        raise ConfigurationError(
            f"DEMO_MODE=prerecorded but {settings.prerecorded_path} does not exist"
        ) from e
    return data


def create_app(settings: Settings | None = None, graph=None) -> FastAPI:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = settings or Settings.from_env()
    prerecorded = load_prerecorded(settings) if settings.demo_mode == "prerecorded" else None

    app = FastAPI(
        title="Agentic OSINT Analyst",
        description="Investigates entity CATEGORIES using open-source public records. "
                    "Rejects requests that look like attempts to identify specific individuals.",
        version=VERSION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)
    graph_holder = {"graph": graph}

    def get_graph():
        if graph_holder["graph"] is None:
            graph_holder["graph"] = build_graph(settings)
        return graph_holder["graph"]

    def client_key(request: Request) -> str:
        if settings.trust_proxy_headers:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def rate_limit(request: Request) -> None:
        retry_after = limiter.check(client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait before sending another request.",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

    @app.get("/health")
    def health():
        return {"status": "ok", "version": VERSION, "mode": settings.demo_mode}

    @app.get("/ready")
    def ready():
        if settings.demo_mode == "prerecorded":
            # Replays a JSON file: no index, embedding model or LLM key involved.
            return {"status": "ready", "mode": "prerecorded",
                    "index_present": None, "llm_configured": None}
        index_present = (settings.chroma_dir / "chroma.sqlite3").exists()
        llm_configured = bool(settings.groq_api_key)
        return {
            "status": "ready" if index_present and llm_configured else "not_ready",
            "mode": "off",
            "index_present": index_present,
            "llm_configured": llm_configured,
        }

    @app.get("/examples")
    def examples():
        return {
            "mode": settings.demo_mode,
            "examples": _example_queries(settings, prerecorded),
        }

    @app.post("/investigate", response_model=InvestigateResponse,
              dependencies=[Depends(rate_limit)])
    def investigate(request: InvestigateRequest):
        profile = request.entity_cluster_id
        try:
            check_guardrail(profile)
        except GuardrailViolation as e:
            raise HTTPException(status_code=400, detail=str(e))

        if prerecorded is not None:
            return _replay(profile, prerecorded)

        try:
            result = get_graph().invoke(initial_state(profile))
        except GuardrailViolation as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ConfigurationError as e:
            log.error("Configuration error: %s", e)
            raise HTTPException(status_code=503, detail="The server is not configured to run "
                                "live investigations (missing LLM credentials).")
        except IndexNotFoundError as e:
            log.error("Index error: %s", e)
            raise HTTPException(status_code=503, detail="The search index is not available on "
                                "this server. It must be built offline before queries are served.")
        except UpstreamLLMError as e:
            raise HTTPException(status_code=502, detail="The language-model provider failed or "
                                "timed out. Please retry shortly.")

        return InvestigateResponse(
            entity_cluster_id=profile,
            report=result["report"],
            citations=result["citations"],
            sources=result["sources"],
            token_usage=result["token_usage"],
            mode="live",
        )

    return app


def _example_queries(settings: Settings, prerecorded: dict | None) -> list[str]:
    if prerecorded is not None:
        return [r["query"] for r in prerecorded["responses"]]
    try:
        data = json.loads(settings.prerecorded_path.read_text())
        return [r["query"] for r in data["responses"]]
    except (OSError, KeyError, ValueError):
        return []


def _replay(profile: str, prerecorded: dict) -> InvestigateResponse:
    wanted = _normalize(profile)
    for rec in prerecorded["responses"]:
        if _normalize(rec["query"]) == wanted:
            return InvestigateResponse(
                entity_cluster_id=profile,
                report=rec["report"],
                citations=rec["citations"],
                sources=rec["sources"],
                token_usage=rec["token_usage"],
                mode="prerecorded",
                notice=prerecorded["notice"],
            )
    raise HTTPException(
        status_code=404,
        detail="This deployment is in prerecorded demo mode: only the example queries "
               "are available. Choose one of the examples.",
    )


app = create_app()
