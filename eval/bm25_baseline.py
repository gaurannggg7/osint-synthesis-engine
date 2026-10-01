"""
BM25 keyword-only baseline retrieval, run against the SAME corpus as the
vector store, so Stage 5's eval harness can report both recall@10 side
by side with an honest delta -- per the spec, even if the baseline wins
on some queries.
"""
import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bm25_baseline")

CORPUS_PATH = "data/processed/corpus_final.jsonl"


def load_corpus(path: str = CORPUS_PATH) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    log.info("Loaded %d documents for BM25 indexing", len(docs))
    return docs


def build_bm25_index(docs: list[dict]):
    """Tokenization is deliberately simple (lowercase + whitespace split) --
    matching the standard BM25 baseline setup, not tuned to flatter this
    corpus. A fairer, more real baseline than a hand-tuned one."""
    tokenized = [d["text"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    return bm25, docs


def search(bm25, docs: list[dict], query: str, k: int = 10) -> list[dict]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "doc_id": docs[i]["doc_id"],
            "source": docs[i]["source"],
            "title": docs[i].get("title", ""),
            "score": float(scores[i]),
            "text": docs[i]["text"][:200],
        }
        for i in ranked_indices
    ]


if __name__ == "__main__":
    docs = load_corpus()
    bm25, docs = build_bm25_index(docs)

    test_query = "cryptocurrency exchange with prior AML compliance actions and sanctions exposure"
    results = search(bm25, docs, test_query, k=10)

    print(f"=== BM25 Top 10 for: {test_query} ===\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.2f} | {r['source']} | {r['title']}")
        print(f"    {r['text']}...")
        print()