"""
Stage 5: recall@10 evaluation -- vector, BM25 baseline, and RRF hybrid --
against the golden set built from real corpus documents (eval/golden_set.json).

Recall@10 definition used here: for each query, of the documents listed
as relevant in the golden set, what fraction appear anywhere in the
method's top 10 results? (Standard recall@k -- not precision@k.)

Implementation notes (real, not glossed over):
- Vector search returns CHUNKS, not whole documents. A query's top-k
  chunks can include multiple chunks from the same parent document.
  Results are deduped to unique parent doc_id, in rank order, before
  computing recall -- otherwise "top 10" would sometimes mean "top 3
  distinct documents," inflating apparent coverage.
- Hybrid combines vector and BM25 rankings via Reciprocal Rank Fusion
  (RRF), the standard method for combining rankings that come from
  different, non-comparable scoring functions (cosine similarity vs.
  BM25 score) -- summing or averaging raw scores directly would be
  mathematically invalid since the scales aren't compatible.
"""
import json
import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from bm25_baseline import load_corpus, build_bm25_index, search as bm25_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_eval")

GOLDEN_SET_PATH = "eval/golden_set.json"
PERSIST_DIR = "vectorstore/chroma_db"
COLLECTION_NAME = "osint_entity_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 10
VECTOR_OVERFETCH = 30  # fetch more chunks than K, then dedupe to unique docs
RRF_CONSTANT = 60      # standard RRF constant from the original paper (Cormack et al.)


def load_golden_set(path: str = GOLDEN_SET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def vector_top_k(vectorstore, query: str, k: int = K) -> list[str]:
    results = vectorstore.similarity_search(query, k=VECTOR_OVERFETCH)
    seen, ranked_doc_ids = set(), []
    for r in results:
        doc_id = r.metadata.get("doc_id")
        if doc_id not in seen:
            seen.add(doc_id)
            ranked_doc_ids.append(doc_id)
        if len(ranked_doc_ids) >= k:
            break
    return ranked_doc_ids


def bm25_top_k(bm25, docs, query: str, k: int = K) -> list[str]:
    results = bm25_search(bm25, docs, query, k=k)
    return [r["doc_id"] for r in results]


def rrf_hybrid(vector_ranked: list[str], bm25_ranked: list[str], k: int = K) -> list[str]:
    scores = {}
    for rank, doc_id in enumerate(vector_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
    for rank, doc_id in enumerate(bm25_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
    ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return ranked[:k]


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for r in relevant if r in retrieved)
    return hit / len(relevant)


def run_eval():
    golden_set = load_golden_set()
    log.info("Loaded %d golden queries", len(golden_set))

    log.info("Loading embedding model + vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    log.info("Loading corpus + building BM25 index...")
    docs = load_corpus()
    bm25, docs = build_bm25_index(docs)

    per_query_results = []
    vector_recalls, bm25_recalls, hybrid_recalls = [], [], []

    for item in golden_set:
        query = item["query"]
        relevant = item["relevant_doc_ids"]

        v_ranked = vector_top_k(vectorstore, query)
        b_ranked = bm25_top_k(bm25, docs, query)
        h_ranked = rrf_hybrid(v_ranked, b_ranked)

        v_recall = recall_at_k(v_ranked, relevant)
        b_recall = recall_at_k(b_ranked, relevant)
        h_recall = recall_at_k(h_ranked, relevant)

        vector_recalls.append(v_recall)
        bm25_recalls.append(b_recall)
        hybrid_recalls.append(h_recall)

        per_query_results.append({
            "id": item["id"],
            "query": query,
            "query_type": item.get("query_type", ""),
            "relevant_doc_ids": relevant,
            "vector_recall": round(v_recall, 3),
            "bm25_recall": round(b_recall, 3),
            "hybrid_recall": round(h_recall, 3),
        })

        log.info("[%s] (%s) vector=%.2f bm25=%.2f hybrid=%.2f -- %s",
                  item["id"], item.get("query_type", "?"), v_recall, b_recall, h_recall, query)

    summary = {
        "n_queries": len(golden_set),
        "mean_vector_recall_at_10": round(sum(vector_recalls) / len(vector_recalls), 3),
        "mean_bm25_recall_at_10": round(sum(bm25_recalls) / len(bm25_recalls), 3),
        "mean_hybrid_recall_at_10": round(sum(hybrid_recalls) / len(hybrid_recalls), 3),
    }
    summary["vector_minus_bm25_delta"] = round(
        summary["mean_vector_recall_at_10"] - summary["mean_bm25_recall_at_10"], 3
    )

    out_path = Path("eval/eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_query": per_query_results}, f, indent=2)
    log.info("Wrote eval results to %s", out_path)

    return summary, per_query_results


if __name__ == "__main__":
    summary, per_query = run_eval()
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))