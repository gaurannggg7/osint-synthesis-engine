"""Integrity checks on the committed demo assets, so a hand edit can't silently break the demo."""
import json

from agent.graph import extract_cited_indices
from config import DEMO_DIR, REPO_ROOT


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def test_prerecorded_responses_are_internally_consistent():
    data = json.loads((DEMO_DIR / "prerecorded_responses.json").read_text())
    assert data["notice"].startswith("PRERECORDED RESPONSE")
    assert len(data["responses"]) >= 3
    for rec in data["responses"]:
        sources = rec["sources"]
        assert [s["index"] for s in sources] == list(range(1, len(sources) + 1))
        cited = extract_cited_indices(rec["report"], len(sources))
        assert [s["index"] for s in sources if s["cited"]] == sorted(cited)
        assert set(rec["citations"]) == {sources[i - 1]["doc_id"] for i in cited}
        assert all(s["url"].startswith("https://") and s["excerpt"] for s in sources)


def test_sample_corpus_has_attribution_fields_and_supports_demo_golden_set():
    docs = load_jsonl(DEMO_DIR / "sample_corpus.jsonl")
    ids = {d["doc_id"] for d in docs}
    assert len(ids) == len(docs)
    for d in docs:
        assert d["text"].strip() and d["url"].startswith("https://") and d["source"] in {
            "OFAC_SDN", "SEC_EDGAR", "CourtListener"}
    golden = json.loads((DEMO_DIR / "golden_set.json").read_text())
    assert all(set(g["relevant_doc_ids"]) <= ids for g in golden)


def test_prerecorded_sources_exist_in_sample_corpus():
    ids = {d["doc_id"] for d in load_jsonl(DEMO_DIR / "sample_corpus.jsonl")}
    data = json.loads((DEMO_DIR / "prerecorded_responses.json").read_text())
    assert {s["doc_id"] for r in data["responses"] for s in r["sources"]} <= ids


def test_no_individual_ofac_records_in_sample_corpus():
    for d in load_jsonl(DEMO_DIR / "sample_corpus.jsonl"):
        if d["source"] == "OFAC_SDN":
            assert d["entity_type"] in {"entity", "vessel", "aircraft"}


def test_env_example_has_placeholders_only():
    text = (REPO_ROOT / ".env.example").read_text()
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            if "KEY" in key or "TOKEN" in key:
                assert value.strip() in {"", "your-key-here", "your-token-here"}, key


def test_demo_requirements_are_minimal_and_consistent_with_full_requirements():
    def pins(name):
        lines = (REPO_ROOT / name).read_text().splitlines()
        return {l.split("==")[0].lower(): l.split("==")[1] for l in lines if "==" in l and not l.startswith("#")}
    demo, full = pins("requirements-demo.txt"), pins("requirements.txt")
    assert not {"torch", "chromadb", "sentence-transformers", "langgraph", "langchain-core",
                "langchain-groq", "langchain-huggingface", "langchain-chroma"} & set(demo)
    assert all(full[k] == v for k, v in demo.items())  # same versions as the tested full set


def test_prerecorded_fixture_has_four_responses():
    data = json.loads((DEMO_DIR / "prerecorded_responses.json").read_text())
    assert len(data["responses"]) == 4
