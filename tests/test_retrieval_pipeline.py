import json

import pytest

from eval.run_eval import (attribute_precision_at_k, query_group, recall_at_k, rrf_hybrid, summarize)
from vectorstore.build_index import CHUNK_THRESHOLD_CHARS, build_langchain_documents
from vectorstore.corpus import load_corpus


def doc(i, text, source="SEC_EDGAR"):
    return {"doc_id": f"d{i}", "source": source, "title": f"T{i}", "url": f"https://x/{i}", "text": text}


def test_short_docs_kept_whole_and_long_docs_chunked_with_parent_pointers():
    long_text = ("Sentence about compliance. " * 100).strip()
    docs = build_langchain_documents([doc(1, "short ofac record", "OFAC_SDN"), doc(2, long_text)])
    short = [d for d in docs if d.metadata["doc_id"] == "d1"]
    chunks = [d for d in docs if d.metadata["doc_id"] == "d2"]
    assert len(short) == 1 and short[0].metadata["chunk_of"] == 1
    assert len(long_text) > CHUNK_THRESHOLD_CHARS and len(chunks) > 1
    assert {c.metadata["chunk_of"] for c in chunks} == {len(chunks)}
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["url"] == "https://x/2" and len(c.page_content) <= 500 for c in chunks)


def test_empty_text_skipped_and_duplicate_doc_ids_rejected():
    assert build_langchain_documents([doc(1, "   "), doc(2, "ok")])[0].metadata["doc_id"] == "d2"
    with pytest.raises(ValueError, match="Duplicate doc_id"):
        build_langchain_documents([doc(1, "a"), doc(1, "b")])


def test_load_corpus_errors_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="demo/sample_corpus.jsonl"):
        load_corpus(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": 1}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_corpus(bad)


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "c"]) == 0.5
    assert recall_at_k([], ["a"]) == 0.0
    assert recall_at_k(["a"], []) == 0.0


def test_rrf_rewards_agreement_between_rankers():
    assert rrf_hybrid(["a", "b", "c"], ["c", "a", "d"], k=3)[0] == "a"
    assert rrf_hybrid(["a"], ["b"], k=1) == ["a"]  # tie broken by insertion order: vector first
    assert len(rrf_hybrid(list("abcdef"), list("ghijkl"), k=4)) == 4


def test_attribute_precision_requires_all_substrings_case_insensitively():
    texts = {"a": "Linked To: NATIONAL IRANIAN TANKER COMPANY; Tug", "b": "national iranian", "c": "other"}
    assert attribute_precision_at_k(["a", "b", "c", "zz"], texts, ["national iranian tanker", "tug"]) == 0.25
    assert attribute_precision_at_k([], texts, ["x"]) == 0.0


def test_query_group_from_doc_id_prefix():
    assert query_group({"relevant_doc_ids": ["ofac-sdn-1"]}) == "OFAC_SDN"
    assert query_group({"relevant_doc_ids": ["sec-edgar-1:f.htm"]}) == "SEC_EDGAR"
    assert query_group({"relevant_doc_ids": ["courtlistener-9"]}) == "CourtListener"


def test_summarize_reports_groups_and_paired_comparison():
    rows = [
        {"group": "OFAC_SDN", "vector_recall": 0.0, "bm25_recall": 0.0, "hybrid_recall": 0.0},
        {"group": "SEC_EDGAR", "vector_recall": 1.0, "bm25_recall": 0.0, "hybrid_recall": 1.0},
        {"group": "SEC_EDGAR", "vector_recall": 0.0, "bm25_recall": 1.0, "hybrid_recall": 1.0},
        {"group": "SEC_EDGAR", "vector_recall": 0.5, "bm25_recall": 0.5, "hybrid_recall": 1.0},
    ]
    s = summarize(rows)
    assert s["mean_vector_recall_at_10"] == 0.375 and s["mean_hybrid_recall_at_10"] == 0.75
    assert s["by_source_group"]["SEC_EDGAR"]["n"] == 3
    assert s["hybrid_vs_best_single_method"] == {"wins": 1, "ties": 3, "losses": 0}


def test_committed_eval_results_match_their_own_summary():
    from config import REPO_ROOT
    res = json.loads((REPO_ROOT / "eval" / "eval_results.json").read_text())
    assert summarize([{k: v for k, v in q.items()} for q in res["per_query"]])["mean_hybrid_recall_at_10"] \
        == res["summary"]["mean_hybrid_recall_at_10"]
    assert res["summary"]["n_queries"] == len(res["per_query"]) == 17


@pytest.mark.slow
def test_real_index_build_is_idempotent_and_retrievable(tmp_path, settings):
    """Real embedding model + Chroma on the committed sample corpus."""
    import dataclasses
    from agent.graph import make_chroma_retriever
    from config import DEMO_DIR
    from vectorstore.build_index import build_index

    s = dataclasses.replace(settings, chroma_dir=tmp_path / "chroma")
    first = build_index(DEMO_DIR / "sample_corpus.jsonl", settings=s)
    second = build_index(DEMO_DIR / "sample_corpus.jsonl", settings=s)
    assert first["vectors_in_collection"] == second["vectors_in_collection"] == first["embedded_chunks"]

    hits = make_chroma_retriever(s)("tug boat linked to Petroleos de Venezuela state oil company", 5)
    assert "ofac-sdn-26651" in [h["doc_id"] for h in hits]
