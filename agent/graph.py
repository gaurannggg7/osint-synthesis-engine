"""
LangGraph agent for entity-category investigation.

Graph: START -> retrieve -> synthesize -> END

retrieve:   queries the Chroma vector store for documents relevant to an
            entity-category profile.
synthesize: sends the retrieved chunks + profile to an LLM (Groq-hosted
            openai/gpt-oss-120b by default) and asks for a report in which
            every claim cites an excerpt number like [1]. After the call the
            report is parsed, and only excerpts the report actually cites
            are returned as `citations`; every retrieved excerpt is
            returned as `sources` so a reader can inspect the evidence.

Guardrail: `check_guardrail` runs before any retrieval or LLM call and
rejects inputs that look like an attempt to identify a specific person.
It is a keyword filter -- a first line of defence, not a classifier (see
README "Limitations").

The retriever and LLM are injectable (`build_graph(retriever=..., llm=...)`)
so the graph can be tested without a vector store, model download or API key.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Protocol, TypedDict

from config import ConfigurationError, Settings

log = logging.getLogger("agent_graph")

EXCERPT_CHARS = 600

_INDIVIDUAL_TARGETING_PATTERNS = [
    r"\bwho is\b",
    r"\breal identity of\b",
    r"\bidentify (the |this )?person\b",
    r"\bfind (out )?who\b",
    r"\bunmask\b",
    r"\bdox\b",
]


class GuardrailViolation(Exception):
    pass


class IndexNotFoundError(RuntimeError):
    """The vector index is missing or empty."""


class UpstreamLLMError(RuntimeError):
    """The LLM provider call failed (network, rate limit, bad response)."""


def check_guardrail(entity_profile: str) -> None:
    lowered = entity_profile.lower()
    for pattern in _INDIVIDUAL_TARGETING_PATTERNS:
        if re.search(pattern, lowered):
            log.warning("Guardrail blocked a request (pattern %s)", pattern)
            raise GuardrailViolation(
                "This looks like an attempt at individual-targeting (identifying a specific "
                "person). This system investigates entity categories and public organizational "
                "records only; try describing a type of entity instead."
            )


class AgentState(TypedDict):
    entity_profile: str
    retrieved_chunks: list[dict]
    report: str
    citations: list[str]  # doc_ids the report actually cites, in citation order
    sources: list[dict]   # every retrieved excerpt, flagged `cited` or not
    token_usage: dict


def initial_state(entity_profile: str) -> AgentState:
    return {
        "entity_profile": entity_profile,
        "retrieved_chunks": [],
        "report": "",
        "citations": [],
        "sources": [],
        "token_usage": {},
    }


class Retriever(Protocol):
    def __call__(self, query: str, k: int) -> list[dict]: ...


class LLM(Protocol):
    def invoke(self, prompt: str): ...


# ---------------------------------------------------------------- retrieval

def make_chroma_retriever(settings: Settings) -> Retriever:
    """Lazy Chroma retriever. The embedding model and the store are loaded on
    first use, and a missing/empty index raises IndexNotFoundError instead of
    silently returning no results (Chroma would otherwise create an empty DB)."""
    store = {}

    def _load():
        if "vs" in store:
            return store["vs"]
        if not (settings.chroma_dir / "chroma.sqlite3").exists():
            raise IndexNotFoundError(
                f"No vector index at {settings.chroma_dir}. Build one with "
                "`python -m vectorstore.build_index` (full corpus) or "
                "`python -m vectorstore.build_index --corpus demo/sample_corpus.jsonl` "
                "(small public-record sample)."
            )
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=HuggingFaceEmbeddings(model_name=settings.embedding_model),
            persist_directory=str(settings.chroma_dir),
        )
        if vs._collection.count() == 0:
            raise IndexNotFoundError(
                f"Collection '{settings.collection_name}' at {settings.chroma_dir} is empty "
                "or missing. Re-run `python -m vectorstore.build_index`."
            )
        store["vs"] = vs
        return vs

    def retrieve(query: str, k: int) -> list[dict]:
        results = _load().similarity_search(query, k=k)
        return [
            {
                "doc_id": r.metadata.get("doc_id"),
                "source": r.metadata.get("source"),
                "title": r.metadata.get("title"),
                "url": r.metadata.get("url"),
                "text": r.page_content,
            }
            for r in results
        ]

    return retrieve


# ---------------------------------------------------------------- synthesis

SYNTHESIS_PROMPT_TEMPLATE = """You are an OSINT analyst producing a structured report on an entity CATEGORY based only on the provided public-record excerpts. You are investigating a type of entity or behavior pattern, never a specific private individual.

ENTITY CLUSTER PROFILE (what is being investigated):
{entity_profile}

RETRIEVED PUBLIC-RECORD EXCERPTS:
{context}

INSTRUCTIONS:
1. Write a structured report describing what these excerpts actually say about this entity category.
2. Every factual claim MUST cite which excerpt number [1], [2], etc. it comes from.
3. If the excerpts do not contain enough information to answer some aspect of the profile, say so explicitly -- do not fill gaps with outside knowledge or speculation.
4. Do not name or attempt to identify any specific private individual, even if one is incidentally mentioned in the excerpts (e.g. a corporate officer's name in a filing). Refer to organizational roles, not individuals, when such text appears.
5. Keep the report factual and grounded. No speculation beyond what the excerpts state.

REPORT:"""

# Accepts [1], [1, 2] and the fullwidth 【1】 form some models emit.
_CITATION_GROUP = re.compile(r"[\[【](\d+(?:\s*[,;，]\s*\d+)*)[\]】]")


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] Source: {c['source']} — {c['title']} (doc_id: {c['doc_id']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )


def extract_cited_indices(report: str, n_sources: int) -> list[int]:
    """1-based excerpt numbers the report cites, in first-citation order.
    Numbers outside 1..n_sources (e.g. a '[7]' the model invented) are ignored."""
    seen: list[int] = []
    for group in _CITATION_GROUP.findall(report):
        for num in re.split(r"[,;，]", group):
            idx = int(num.strip())
            if 1 <= idx <= n_sources and idx not in seen:
                seen.append(idx)
    return seen


def build_sources(chunks: list[dict], cited: list[int]) -> list[dict]:
    return [
        {
            "index": i,
            "doc_id": c["doc_id"],
            "source": c["source"],
            "title": c["title"],
            "url": c["url"],
            "excerpt": c["text"][:EXCERPT_CHARS],
            "cited": i in cited,
        }
        for i, c in enumerate(chunks, 1)
    ]


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content blocks
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in content
        )
    return str(content)


def _make_groq_llm(settings: Settings) -> LLM:
    if not settings.groq_api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a key from "
            "https://console.groq.com, or set it in the server environment."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.llm_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
        reasoning_effort="medium",  # gpt-oss: ask Groq to return its reasoning separately
    )


def write_trace(settings: Settings, entity_profile: str, state: AgentState,
                reasoning_trace: str | None) -> None:
    """Writes an inspectable record of one invocation (retrieval hits, the
    model's reasoning trace if the API returned one, report, token usage).
    Disabled when TRACE_DIR is empty; a write failure never fails a request."""
    if settings.trace_dir is None:
        return
    try:
        settings.trace_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace = {
            "timestamp": timestamp,
            "entity_profile": entity_profile,
            "step_1_retrieval": {
                "num_chunks_retrieved": len(state["retrieved_chunks"]),
                "chunks": [
                    {"doc_id": c["doc_id"], "source": c["source"], "title": c["title"]}
                    for c in state["retrieved_chunks"]
                ],
            },
            "step_2_reasoning_trace": reasoning_trace or "(not returned by API for this call)",
            "step_3_final_report": state["report"],
            "step_4_citations": state["citations"],
            "token_usage": state["token_usage"],
        }
        out_path = settings.trace_dir / f"trace_{timestamp}.json"
        out_path.write_text(json.dumps(trace, indent=2))
        log.info("Wrote inspectable trace to %s", out_path)
    except OSError as e:
        log.warning("Could not write trace (continuing): %s", e)


# -------------------------------------------------------------------- graph

def build_graph(settings: Settings | None = None,
                retriever: Retriever | None = None,
                llm: LLM | None = None):
    from langgraph.graph import END, START, StateGraph

    settings = settings or Settings.from_env()
    retriever = retriever or make_chroma_retriever(settings)
    llm_holder: dict = {"llm": llm}

    def get_llm() -> LLM:
        if llm_holder["llm"] is None:
            llm_holder["llm"] = _make_groq_llm(settings)
        return llm_holder["llm"]

    def retrieve_node(state: AgentState) -> AgentState:
        check_guardrail(state["entity_profile"])
        get_llm()  # fail fast on missing credentials, before the slow retrieval step
        log.info("RETRIEVE: querying vector store (k=%d)", settings.top_k)
        chunks = retriever(state["entity_profile"], settings.top_k)
        log.info("RETRIEVE: got %d chunks from sources: %s",
                 len(chunks), [c["source"] for c in chunks])
        return {**state, "retrieved_chunks": chunks}

    def synthesize_node(state: AgentState) -> AgentState:
        chunks = state["retrieved_chunks"]
        if not chunks:
            return {**state, "report": "No relevant documents found.", "citations": [],
                    "sources": [], "token_usage": {"input_tokens": 0, "output_tokens": 0}}

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            entity_profile=state["entity_profile"], context=format_context(chunks)
        )
        model = get_llm()  # raises ConfigurationError before any network call

        start = time.time()
        try:
            response = model.invoke(prompt)
        except Exception as e:  # provider/network errors vary by SDK version
            log.error("LLM call failed: %s: %s", type(e).__name__, e)
            raise UpstreamLLMError(f"LLM provider call failed ({type(e).__name__})") from e
        elapsed = time.time() - start

        metadata = getattr(response, "response_metadata", None) or {}
        usage = metadata.get("token_usage", {}) or {}
        extras = getattr(response, "additional_kwargs", None) or {}
        reasoning_trace = extras.get("reasoning_content") or extras.get("reasoning")

        report = _message_text(response.content)
        cited = extract_cited_indices(report, len(chunks))
        citations: list[str] = []
        for i in cited:
            doc_id = chunks[i - 1]["doc_id"]
            if doc_id not in citations:
                citations.append(doc_id)

        details = usage.get("completion_tokens_details") or {}
        token_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": details.get("reasoning_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            # wall-clock time of the LLM call only (excludes retrieval)
            "latency_seconds": round(elapsed, 2),
        }
        log.info("SYNTHESIZE: LLM call %.2fs, usage=%s, cited excerpts=%s",
                 elapsed, token_usage, cited)

        new_state: AgentState = {
            **state,
            "report": report,
            "citations": citations,
            "sources": build_sources(chunks, cited),
            "token_usage": token_usage,
        }
        write_trace(settings, state["entity_profile"], new_state, reasoning_trace)
        return new_state

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Run one investigation from the command line.")
    parser.add_argument("profile", nargs="?",
                        default="cryptocurrency exchange with prior AML compliance actions and sanctions exposure")
    args = parser.parse_args()

    app = build_graph()
    result = app.invoke(initial_state(args.profile))
    print("=== REPORT ===")
    print(result["report"])
    print("\n=== CITED DOCUMENTS ===")
    print(json.dumps(result["citations"], indent=2))
    print("\n=== TOKEN USAGE / TIMING ===")
    print(json.dumps(result["token_usage"], indent=2))


if __name__ == "__main__":
    main()
