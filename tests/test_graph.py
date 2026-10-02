import dataclasses
import json

import pytest

from agent.graph import (GuardrailViolation, UpstreamLLMError, build_graph, build_sources,
                         check_guardrail, extract_cited_indices, initial_state)
from config import ConfigurationError
from tests.conftest import CHUNKS, FakeLLM, FakeRetriever

PROFILE = "money services business with prior consent order for AML violations"


@pytest.mark.parametrize("text", [
    "who is behind this wallet", "find out who runs the exchange", "real identity of the operator",
    "identify this person from the filing", "unmask the founder", "dox the developer",
])
def test_guardrail_blocks_individual_targeting(text):
    with pytest.raises(GuardrailViolation):
        check_guardrail(text)


def test_guardrail_allows_category_queries():
    check_guardrail(PROFILE)
    check_guardrail("crypto mixing service with links to sanctioned jurisdictions")


@pytest.mark.parametrize("report,n,expected", [
    ("A [1] and B [2].", 3, [1, 2]),
    ("Grouped [1, 3] and [2; 3].", 3, [1, 3, 2]),
    ("Invented [9] and [0] ignored, real [2].", 3, [2]),
    ("No citations at all.", 3, []),
    ("Repeated [2] [2] [1].", 3, [2, 1]),
    ("Fullwidth 【1】 and 【2, 3】.", 3, [1, 2, 3]),
])
def test_extract_cited_indices(report, n, expected):
    assert extract_cited_indices(report, n) == expected


def test_build_sources_flags_cited_and_truncates():
    chunks = [dict(CHUNKS[0], text="x" * 5000), CHUNKS[1]]
    sources = build_sources(chunks, cited=[2])
    assert [s["index"] for s in sources] == [1, 2]
    assert [s["cited"] for s in sources] == [False, True]
    assert len(sources[0]["excerpt"]) == 600


def test_graph_returns_only_actually_cited_documents(settings):
    llm, retriever = FakeLLM("Claim [1]. Another [3]. Fake [8]."), FakeRetriever()
    result = build_graph(settings, retriever=retriever, llm=llm).invoke(initial_state(PROFILE))

    assert result["citations"] == ["doc-a"]          # [1] and [3] are both doc-a; deduped
    assert [s["cited"] for s in result["sources"]] == [True, False, True]
    assert result["token_usage"]["input_tokens"] == 10
    assert retriever.calls == [(PROFILE, settings.top_k)]
    assert "[2] Source: OFAC_SDN" in llm.prompts[0]


def test_graph_no_chunks_skips_llm(settings):
    llm = FakeLLM()
    result = build_graph(settings, retriever=FakeRetriever([]), llm=llm).invoke(initial_state(PROFILE))
    assert result["report"] == "No relevant documents found."
    assert result["sources"] == [] and llm.prompts == []


def test_guardrail_runs_before_retrieval(settings):
    retriever = FakeRetriever()
    graph = build_graph(settings, retriever=retriever, llm=FakeLLM())
    with pytest.raises(GuardrailViolation):
        graph.invoke(initial_state("who is the owner of this exchange"))
    assert retriever.calls == []


def test_missing_api_key_is_a_configuration_error(settings):
    graph = build_graph(dataclasses.replace(settings, groq_api_key=None), retriever=FakeRetriever())
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        graph.invoke(initial_state(PROFILE))


def test_llm_failure_becomes_upstream_error(settings):
    graph = build_graph(settings, retriever=FakeRetriever(), llm=FakeLLM(error=TimeoutError("slow")))
    with pytest.raises(UpstreamLLMError):
        graph.invoke(initial_state(PROFILE))


def test_trace_written_when_enabled_and_skipped_when_disabled(settings, tmp_path):
    traced = dataclasses.replace(settings, trace_dir=tmp_path / "traces")
    build_graph(traced, retriever=FakeRetriever(), llm=FakeLLM()).invoke(initial_state(PROFILE))
    files = list((tmp_path / "traces").glob("trace_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["step_4_citations"] == ["doc-a"]

    build_graph(settings, retriever=FakeRetriever(), llm=FakeLLM()).invoke(initial_state(PROFILE))
    assert len(list((tmp_path / "traces").glob("trace_*.json"))) == 1  # unchanged


def test_trace_write_failure_does_not_fail_request(settings, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file in the way")
    bad = dataclasses.replace(settings, trace_dir=blocker / "traces")
    result = build_graph(bad, retriever=FakeRetriever(), llm=FakeLLM()).invoke(initial_state(PROFILE))
    assert result["report"]
