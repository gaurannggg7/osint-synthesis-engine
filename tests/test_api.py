import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from agent.graph import IndexNotFoundError, UpstreamLLMError, build_graph
from api.main import create_app
from config import ConfigurationError, Settings
from tests.conftest import FakeLLM, FakeRetriever

PROFILE = "money services business with prior consent order for AML violations"
ORIGIN = "https://app.example.com"


def client_for(settings, llm=None, retriever=None, **kw):
    graph = build_graph(settings, retriever=retriever or FakeRetriever(), llm=llm or FakeLLM())
    return TestClient(create_app(settings, graph=graph), **kw)


def test_health_and_ready(settings):
    c = client_for(settings)
    assert c.get("/health").json()["status"] == "ok"
    ready = c.get("/ready").json()
    assert ready["status"] == "not_ready" and ready["index_present"] is False  # no index in tmp dir
    assert "key" not in json.dumps(ready).lower().replace("llm_configured", "")


def test_investigate_returns_report_sources_and_mode(settings):
    r = client_for(settings).post("/investigate", json={"entity_cluster_id": PROFILE})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "live" and body["notice"] is None
    assert body["citations"] == ["doc-a"]
    assert len(body["sources"]) == 3 and body["sources"][0]["cited"] is True
    assert body["entity_cluster_id"] == PROFILE


@pytest.mark.parametrize("payload", [
    {}, {"entity_cluster_id": "short"}, {"entity_cluster_id": "x" * 301},
    {"entity_cluster_id": "valid looking text\x00with null byte"},
    {"entity_cluster_id": PROFILE, "extra": "field"}, {"entity_cluster_id": 12345678901},
])
def test_invalid_requests_rejected_with_422(settings, payload):
    assert client_for(settings).post("/investigate", json=payload).status_code == 422


def test_input_is_whitespace_normalized(settings):
    retriever = FakeRetriever()
    client_for(settings, retriever=retriever).post(
        "/investigate", json={"entity_cluster_id": "  money   services\n business with AML orders  "})
    assert retriever.calls[0][0] == "money services business with AML orders"


def test_guardrail_violation_is_400(settings):
    r = client_for(settings).post("/investigate", json={"entity_cluster_id": "who is the owner of this exchange"})
    assert r.status_code == 400 and "individual-targeting" in r.json()["detail"]


@pytest.mark.parametrize("error,status", [
    (ConfigurationError("GROQ_API_KEY missing"), 503),
    (IndexNotFoundError("no index at /secret/path"), 503),
])
def test_infrastructure_errors_map_to_503_without_leaking_details(settings, error, status):
    class Boom(FakeRetriever):
        def __call__(self, q, k):
            raise error
    r = client_for(settings, retriever=Boom()).post("/investigate", json={"entity_cluster_id": PROFILE})
    assert r.status_code == status
    assert "secret/path" not in r.text and "GROQ_API_KEY" not in r.text


def test_llm_failure_maps_to_502(settings):
    c = client_for(settings, llm=FakeLLM(error=RuntimeError("provider down")))
    r = c.post("/investigate", json={"entity_cluster_id": PROFILE})
    assert r.status_code == 502 and "provider down" not in r.text


def test_cors_allows_configured_origin_only(settings):
    c = client_for(settings)
    ok = c.options("/investigate", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
    assert ok.headers["access-control-allow-origin"] == ORIGIN
    bad = c.options("/investigate", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    simple = c.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_wildcard_cors_origin_is_refused(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(ConfigurationError):
        Settings.from_env()


def test_rate_limit_per_client_with_retry_after(settings):
    c = client_for(dataclasses.replace(settings, rate_limit_per_minute=2))
    body = {"entity_cluster_id": PROFILE}
    assert c.post("/investigate", json=body).status_code == 200
    assert c.post("/investigate", json=body).status_code == 200
    third = c.post("/investigate", json=body)
    assert third.status_code == 429 and int(third.headers["retry-after"]) >= 1
    assert c.get("/health").status_code == 200  # health is never limited


def test_rate_limit_keys_on_forwarded_for_only_when_trusted(settings):
    limited = dataclasses.replace(settings, rate_limit_per_minute=1)
    body = {"entity_cluster_id": PROFILE}

    untrusted = client_for(limited)  # header ignored: both requests share the socket peer
    assert untrusted.post("/investigate", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert untrusted.post("/investigate", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429

    trusted = client_for(dataclasses.replace(limited, trust_proxy_headers=True))
    assert trusted.post("/investigate", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    assert trusted.post("/investigate", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200
    assert trusted.post("/investigate", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 429


@pytest.fixture
def prerecorded_settings(settings, tmp_path):
    path = tmp_path / "pre.json"
    path.write_text(json.dumps({
        "notice": "PRERECORDED RESPONSE: test", "model": "m",
        "responses": [{"query": PROFILE, "report": "saved [1]", "citations": ["doc-a"],
                       "sources": [{"index": 1, "doc_id": "doc-a", "source": "SEC_EDGAR", "title": "T",
                                    "url": "https://x", "excerpt": "e", "cited": True}],
                       "token_usage": {"latency_seconds": 3.1}}]}))
    return dataclasses.replace(settings, demo_mode="prerecorded", prerecorded_path=path)


def test_prerecorded_mode_replays_and_labels(prerecorded_settings):
    llm = FakeLLM()
    c = client_for(prerecorded_settings, llm=llm)
    r = c.post("/investigate", json={"entity_cluster_id": PROFILE.upper()})  # case-insensitive match
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "prerecorded" and body["notice"].startswith("PRERECORDED")
    assert body["report"] == "saved [1]" and llm.prompts == []
    assert c.get("/examples").json()["examples"] == [PROFILE]
    ready = c.get("/ready").json()
    assert ready["status"] == "ready" and ready["index_present"] is None


def test_prerecorded_mode_rejects_other_queries_and_still_applies_guardrail(prerecorded_settings):
    c = client_for(prerecorded_settings)
    assert c.post("/investigate", json={"entity_cluster_id": "some other unrecorded query text"}).status_code == 404
    assert c.post("/investigate", json={"entity_cluster_id": "who is the owner of this exchange"}).status_code == 400


def test_prerecorded_mode_requires_the_file(settings, tmp_path):
    missing = dataclasses.replace(settings, demo_mode="prerecorded", prerecorded_path=tmp_path / "nope.json")
    with pytest.raises(ConfigurationError):
        create_app(missing)


def test_prerecorded_mode_loads_no_heavy_dependencies():
    """Replay must not import torch / chromadb / langgraph or touch an index."""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import os, sys
        os.environ["DEMO_MODE"] = "prerecorded"
        from fastapi.testclient import TestClient
        from api.main import app
        c = TestClient(app)
        assert c.get("/ready").status_code == 200
        ex = c.get("/examples").json()["examples"]
        assert c.post("/investigate", json={"entity_cluster_id": ex[0]}).status_code == 200
        heavy = [m for m in ("torch", "chromadb", "sentence_transformers", "langchain_chroma",
                             "langchain_huggingface", "langgraph", "langchain_groq") if m in sys.modules]
        assert not heavy, heavy
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
