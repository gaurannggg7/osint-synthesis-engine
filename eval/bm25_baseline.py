"""
BM25 keyword-only baseline retrieval, run against the SAME corpus as the
vector store, so the eval harness can report both side by side with an
honest delta -- even if the baseline wins on some queries.
"""
import logging

from rank_bm25 import BM25Okapi

from config import Settings
from vectorstore.corpus import load_corpus

log = logging.getLogger("bm25_baseline")


def build_bm25_index(docs: list[dict]):
    """Tokenization is deliberately simple (lowercase + whitespace split) --
    the standard BM25 baseline setup, not tuned to flatter this corpus.
    (Punctuation is not stripped, so "venezuela," != "venezuela"; a regex
    tokenizer was tried and did not change recall on the OFAC queries.)"""
    tokenized = [d["text"].lower().split() for d in docs]
    return BM25Okapi(tokenized), docs


def search(bm25, docs: list[dict], query: str, k: int = 10) -> list[dict]:
    scores = bm25.get_scores(query.lower().split())
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    docs = load_corpus(Settings.from_env().corpus_path)
    bm25, docs = build_bm25_index(docs)

    test_query = "cryptocurrency exchange with prior AML compliance actions and sanctions exposure"
    print(f"=== BM25 Top 10 for: {test_query} ===\n")
    for i, r in enumerate(search(bm25, docs, test_query, k=10), 1):
        print(f"[{i}] score={r['score']:.2f} | {r['source']} | {r['title']}")
        print(f"    {r['text']}...\n")
