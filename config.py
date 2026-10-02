"""
Central, environment-driven configuration.

Every path and tunable the project uses is read here, once, so that
(a) nothing depends on the current working directory, and
(b) a deployment can relocate the corpus / vector store / traces with
    environment variables alone.

Values come from the process environment, with a local `.env` file as a
convenience for development (see `.env.example`). Secrets are only ever
read from the environment -- never hard-coded and never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(REPO_ROOT / ".env")

# Chroma's anonymous telemetry is noisy and pointless for a backend service.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

DEMO_DIR = REPO_ROOT / "demo"


class ConfigurationError(RuntimeError):
    """Raised for missing or invalid configuration, with a fix in the message."""


def _path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    p = Path(raw).expanduser() if raw else default
    return p if p.is_absolute() else REPO_ROOT / p


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from e


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str) -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def require_contact_user_agent(settings: "Settings") -> str:
    """SEC and OFAC block requests that lack a real contact identity in the
    User-Agent. Ingestion scripts call this at run time (not import time)."""
    if not settings.contact_user_agent:
        raise ConfigurationError(
            "CONTACT_USER_AGENT is not set. Add `CONTACT_USER_AGENT=Your Name your.email@example.com` "
            "to .env -- SEC and OFAC reject requests without a real contact identity."
        )
    return settings.contact_user_agent


@dataclass(frozen=True)
class Settings:
    # --- storage ---
    data_dir: Path
    corpus_path: Path
    chroma_dir: Path
    collection_name: str
    trace_dir: Path | None  # None disables trace files (recommended for hosted APIs)

    # --- models ---
    embedding_model: str
    llm_model: str
    llm_timeout_seconds: int
    top_k: int
    groq_api_key: str | None

    # --- API ---
    cors_allowed_origins: list[str]
    rate_limit_per_minute: int  # 0 disables
    trust_proxy_headers: bool
    demo_mode: str  # "off" | "prerecorded"
    prerecorded_path: Path

    # --- ingestion ---
    contact_user_agent: str | None
    courtlistener_token: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = _path("DATA_DIR", REPO_ROOT / "data" / "processed")

        trace_raw = os.environ.get("TRACE_DIR")
        if trace_raw is None:
            trace_dir: Path | None = REPO_ROOT / "logs" / "traces"
        elif trace_raw.strip() == "":
            trace_dir = None
        else:
            trace_dir = _path("TRACE_DIR", REPO_ROOT / "logs" / "traces")

        origins = _csv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
        if "*" in origins:
            raise ConfigurationError(
                "CORS_ALLOWED_ORIGINS must list explicit origins, not '*'. "
                "Example: CORS_ALLOWED_ORIGINS=https://your-app.vercel.app"
            )

        demo_mode = os.environ.get("DEMO_MODE", "off").strip().lower() or "off"
        if demo_mode not in {"off", "prerecorded"}:
            raise ConfigurationError(f"DEMO_MODE must be 'off' or 'prerecorded', got {demo_mode!r}")

        return cls(
            data_dir=data_dir,
            corpus_path=_path("CORPUS_PATH", data_dir / "corpus_final.jsonl"),
            chroma_dir=_path("CHROMA_PERSIST_DIR", REPO_ROOT / "vectorstore" / "chroma_db"),
            collection_name=os.environ.get("CHROMA_COLLECTION", "osint_entity_corpus"),
            trace_dir=trace_dir,
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
            llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 30),
            top_k=_int("TOP_K", 5),
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            cors_allowed_origins=origins,
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 10),
            trust_proxy_headers=_bool("TRUST_PROXY_HEADERS", False),
            demo_mode=demo_mode,
            prerecorded_path=_path("PRERECORDED_PATH", DEMO_DIR / "prerecorded_responses.json"),
            contact_user_agent=os.environ.get("CONTACT_USER_AGENT") or None,
            courtlistener_token=os.environ.get("COURTLISTENER_API_TOKEN") or None,
        )
