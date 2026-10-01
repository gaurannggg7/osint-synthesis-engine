"""
Stage 9: Minimal FastAPI deployment wrapper.

POST /investigate takes an entity cluster ID (or a raw entity profile
string, for now, since there is no external CryptoGraph/GuardianAI
system feeding real cluster IDs into this project), runs it through the
LangGraph agent, and returns the structured report.

Guardrail note: the agent's own check_guardrail() runs inside
retrieve_node() before any retrieval happens, so a request that
attempts to target a named individual is rejected here too, surfaced
as a 400 error rather than a silent failure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.graph import build_graph, GuardrailViolation

app = FastAPI(
    title="Agentic OSINT Analyst",
    description="Investigates entity CATEGORIES using open-source public records. "
                "Never accepts or outputs identity claims about specific individuals.",
    version="0.1.0",
)

_graph = build_graph()


class InvestigateRequest(BaseModel):
    entity_cluster_id: str  # in this build, a free-text entity-category profile
                             # (e.g. "cryptocurrency exchange with prior AML actions")
                             # rather than an integer ID from an external clustering
                             # system, since no such system is connected here.


class InvestigateResponse(BaseModel):
    entity_cluster_id: str
    report: str
    citations: list[str]
    token_usage: dict


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(request: InvestigateRequest):
    try:
        result = _graph.invoke({
            "entity_profile": request.entity_cluster_id,
            "retrieved_chunks": [],
            "report": "",
            "citations": [],
            "token_usage": {},
        })
    except GuardrailViolation as e:
        raise HTTPException(status_code=400, detail=str(e))

    return InvestigateResponse(
        entity_cluster_id=request.entity_cluster_id,
        report=result["report"],
        citations=result["citations"],
        token_usage=result["token_usage"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}