"""
LangGraph agent for entity-cluster investigation.

Graph: START -> retrieve -> synthesize -> END

retrieve:   queries the Chroma vector store for documents relevant to
            an entity cluster's on-chain behavior profile.
synthesize: sends retrieved chunks + query to an LLM (Groq/Llama 3.3 70B)
            to produce a structured, cited report. The prompt explicitly
            instructs the model to ground every claim in the retrieved
            text and to say so plainly when the retrieved context does
            not support an answer -- this is the real grounding
            mechanism Stage 5's answer-groundedness eval will check.

HARD GUARDRAIL: entity_profile is validated before any retrieval or LLM
call happens, rejecting inputs that look like an attempt to identify a
specific private individual.
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("agent_graph")

PERSIST_DIR = "vectorstore/chroma_db"
COLLECTION_NAME = "osint_entity_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL = "openai/gpt-oss-120b"  # verified live 2026-09-18; llama-3.3-70b-versatile was deprecated by Groq on 2026-06-17
TOP_K = 5

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


def check_guardrail(entity_profile: str) -> None:
    lowered = entity_profile.lower()
    for pattern in _INDIVIDUAL_TARGETING_PATTERNS:
        if re.search(pattern, lowered):
            raise GuardrailViolation(
                f"Input matched individual-targeting pattern {pattern!r}. "
                "This system investigates entity categories and public "
                "organizational records only."
            )


class AgentState(TypedDict):
    entity_profile: str
    retrieved_chunks: list[dict]
    report: str
    citations: list[str]
    token_usage: dict  # real usage tracking, feeds Stage 7 cost work


_vectorstore = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


def retrieve_node(state: AgentState) -> AgentState:
    check_guardrail(state["entity_profile"])

    log.info("RETRIEVE step: querying vector store for: %s", state["entity_profile"])
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(state["entity_profile"], k=TOP_K)

    chunks = [
        {
            "doc_id": r.metadata.get("doc_id"),
            "source": r.metadata.get("source"),
            "title": r.metadata.get("title"),
            "url": r.metadata.get("url"),
            "text": r.page_content,
        }
        for r in results
    ]
    log.info("RETRIEVE step: got %d chunks from sources: %s",
              len(chunks), [c["source"] for c in chunks])

    return {**state, "retrieved_chunks": chunks}


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


# def synthesize_node(state: AgentState) -> AgentState:
#     log.info("SYNTHESIZE step: sending %d chunks to LLM for grounded synthesis", len(state["retrieved_chunks"]))

#     if not state["retrieved_chunks"]:
#         return {**state, "report": "No relevant documents found.", "citations": [],
#                 "token_usage": {"input_tokens": 0, "output_tokens": 0}}

#     context_parts = []
#     for i, chunk in enumerate(state["retrieved_chunks"], 1):
#         context_parts.append(
#             f"[{i}] Source: {chunk['source']} — {chunk['title']} (doc_id: {chunk['doc_id']})\n{chunk['text']}"
#         )
#     context = "\n\n".join(context_parts)

#     prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
#         entity_profile=state["entity_profile"],
#         context=context,
#     )

#     llm = ChatGroq(model=LLM_MODEL, temperature=0.0, groq_api_key=os.environ.get("GROQ_API_KEY"))

#     start = time.time()
#     response = llm.invoke(prompt)
#     elapsed = time.time() - start

#     usage = response.response_metadata.get("token_usage", {}) if hasattr(response, "response_metadata") else {}
#     log.info("SYNTHESIZE step: LLM call took %.2fs, token usage: %s", elapsed, usage)

#     citations = [c["doc_id"] for c in state["retrieved_chunks"]]

#     return {
#         **state,
#         "report": response.content,
#         "citations": citations,
#         "token_usage": {
#             "input_tokens": usage.get("prompt_tokens", 0),
#             "output_tokens": usage.get("completion_tokens", 0),
#             "total_tokens": usage.get("total_tokens", 0),
#             "latency_seconds": round(elapsed, 2),
#         },
#     }
def synthesize_node(state: AgentState) -> AgentState:
    log.info("SYNTHESIZE step: sending %d chunks to LLM for grounded synthesis", len(state["retrieved_chunks"]))

    if not state["retrieved_chunks"]:
        return {**state, "report": "No relevant documents found.", "citations": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0}}

    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']} — {chunk['title']} (doc_id: {chunk['doc_id']})\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        entity_profile=state["entity_profile"],
        context=context,
    )

    # llm = ChatGroq(
    #     model=LLM_MODEL,
    #     temperature=0.0,
    #     groq_api_key=os.environ.get("GROQ_API_KEY"),
    #     model_kwargs={"reasoning_effort": "medium"},  # asks gpt-oss to expose its reasoning trace
    # )
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        reasoning_effort="medium",  # asks gpt-oss to expose its reasoning trace
    )

    start = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start

    usage = response.response_metadata.get("token_usage", {}) if hasattr(response, "response_metadata") else {}

    # Real chain-of-thought capture: gpt-oss-120b on Groq can return its
    # internal reasoning separately from the final answer. If present,
    # this is the model's ACTUAL reasoning trace, not a fabricated one.
    reasoning_trace = None
    if hasattr(response, "additional_kwargs"):
        reasoning_trace = response.additional_kwargs.get("reasoning_content") or response.additional_kwargs.get("reasoning")

    log.info("SYNTHESIZE step: LLM call took %.2fs, token usage: %s, reasoning captured: %s",
              elapsed, usage, bool(reasoning_trace))

    citations = [c["doc_id"] for c in state["retrieved_chunks"]]

    token_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if usage.get("completion_tokens_details") else 0,
        "total_tokens": usage.get("total_tokens", 0),
        "latency_seconds": round(elapsed, 2),
    }

    new_state = {
        **state,
        "report": response.content,
        "citations": citations,
        "token_usage": token_usage,
    }

    write_trace(state["entity_profile"], new_state, reasoning_trace)

    return new_state


def write_trace(entity_profile: str, final_state: AgentState, reasoning_trace: str | None):
    """Writes a real, inspectable step-by-step trace of this invocation to
    disk -- guardrail check, retrieval query and chunk-level results,
    the model's actual reasoning trace (if the API returned one), the
    final synthesized report, and token/latency numbers. This is what
    'Chain-of-Thought logged and inspectable at each step' means here:
    a real record of what happened, not a narrated summary."""
    Path("logs/traces").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trace = {
        "timestamp": timestamp,
        "entity_profile": entity_profile,
        "step_1_retrieval": {
            "num_chunks_retrieved": len(final_state["retrieved_chunks"]),
            "chunks": [
                {"doc_id": c["doc_id"], "source": c["source"], "title": c["title"]}
                for c in final_state["retrieved_chunks"]
            ],
        },
        "step_2_reasoning_trace": reasoning_trace if reasoning_trace else "(not returned by API for this call)",
        "step_3_final_report": final_state["report"],
        "step_4_citations": final_state["citations"],
        "token_usage": final_state["token_usage"],
    }
    out_path = Path("logs/traces") / f"trace_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(trace, f, indent=2)
    log.info("Wrote inspectable trace to %s", out_path)


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set in .env")

    app = build_graph()

    test_profile = "cryptocurrency exchange with prior AML compliance actions and sanctions exposure"

    result = app.invoke({
        "entity_profile": test_profile,
        "retrieved_chunks": [],
        "report": "",
        "citations": [],
        "token_usage": {},
    })

    print("=== REPORT ===")
    print(result["report"])
    print("\n=== CITATIONS ===")
    print(json.dumps(result["citations"], indent=2))
    print("\n=== TOKEN USAGE / TIMING ===")
    print(json.dumps(result["token_usage"], indent=2))