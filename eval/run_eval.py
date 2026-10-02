"""
Retrieval evaluation: recall@10 for vector search, a BM25 baseline, and an
RRF hybrid, against a golden set of (query -> relevant doc_ids).

    python -m eval.run_eval                                   # full corpus + golden set
    python -m eval.run_eval --corpus demo/sample_corpus.jsonl \
        --golden demo/golden_set.json --persist-dir /tmp/demo_chroma --out /tmp/demo_eval.json

Metric definitions
- recall@10 (strict): of the doc_ids the golden set lists as relevant for a
  query, the fraction found in the method's top 10 unique documents.
- attribute precision@10 (lenient, optional): for queries that carry a
  `relevant_text_contains` list, the fraction of the top-10 documents whose
  text contains ALL of those substrings (case-insensitive). It exists
  because several golden queries name a *group* (e.g. 64 OFAC records are
  "Linked To: NATIONAL IRANIAN TANKER COMPANY") while only one document is
  labelled relevant, so strict recall is capped by label incompleteness,
  not just by retrieval quality. These patterns were added after the first
  results were seen; treat them as a diagnostic, not a pre-registered metric.

Implementation notes
- Vector search returns CHUNKS. Results are deduplicated to unique parent
  doc_id in rank order before scoring, otherwise "top 10" could mean "top 3
  distinct documents". Vector search over-fetches 30 chunks; if those
  collapse to fewer than 10 documents, fewer than 10 are scored.
- Hybrid uses Reciprocal Rank Fusion (k=60, Cormack et al.), because cosine
  distances and BM25 scores are not on comparable scales.
- n is small (17 queries): differences of one or two queries are noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import REPO_ROOT, Settings
from eval.bm25_baseline import build_bm25_index, search as bm25_search
from vectorstore.corpus import load_corpus

log = logging.getLogger("run_eval")

GOLDEN_SET_PATH = REPO_ROOT / "eval" / "golden_set.json"
DEFAULT_OUT = REPO_ROOT / "eval" / "eval_results.json"
K = 10
VECTOR_OVERFETCH = 30  # fetch more chunks than K, then dedupe to unique docs
RRF_CONSTANT = 60      # standard RRF constant (Cormack et al.)
METHODS = ("vector", "bm25", "hybrid")


# ------------------------------------------------------------ pure metrics

def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in relevant if r in retrieved) / len(relevant)


def rrf_hybrid(vector_ranked: list[str], bm25_ranked: list[str], k: int = K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (vector_ranked, bm25_ranked):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_CONSTANT + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]


def attribute_precision_at_k(retrieved: list[str], text_by_id: dict[str, str],
                             required: list[str]) -> float:
    if not retrieved:
        return 0.0
    needles = [r.lower() for r in required]
    hits = sum(1 for d in retrieved
               if all(n in text_by_id.get(d, "").lower() for n in needles))
    return hits / len(retrieved)


def query_group(item: dict) -> str:
    first = item["relevant_doc_ids"][0]
    for prefix, name in (("ofac", "OFAC_SDN"), ("sec-edgar", "SEC_EDGAR"),
                         ("courtlistener", "CourtListener")):
        if first.startswith(prefix):
            return name
    return "other"


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def summarize(per_query: list[dict]) -> dict:
    summary = {"n_queries": len(per_query)}
    for m in METHODS:
        summary[f"mean_{m}_recall_at_10"] = mean([q[f"{m}_recall"] for q in per_query])
    summary["vector_minus_bm25_delta"] = round(
        summary["mean_vector_recall_at_10"] - summary["mean_bm25_recall_at_10"], 3)

    by_group: dict[str, dict] = {}
    for q in per_query:
        by_group.setdefault(q["group"], []).append(q)
    summary["by_source_group"] = {
        g: {"n": len(qs), **{f"{m}_recall": mean([q[f"{m}_recall"] for q in qs]) for m in METHODS}}
        for g, qs in sorted(by_group.items())
    }

    # Paired comparison: on how many queries does hybrid beat / tie / lose to
    # the best single method? Guards against over-reading a small mean gap.
    wins = ties = losses = 0
    for q in per_query:
        best_single = max(q["vector_recall"], q["bm25_recall"])
        if q["hybrid_recall"] > best_single:
            wins += 1
        elif q["hybrid_recall"] == best_single:
            ties += 1
        else:
            losses += 1
    summary["hybrid_vs_best_single_method"] = {"wins": wins, "ties": ties, "losses": losses}

    attr = [q for q in per_query if "attr_precision" in q]
    if attr:
        summary["attribute_precision_at_10"] = {
            "n": len(attr),
            **{m: mean([q["attr_precision"][m] for q in attr]) for m in METHODS},
        }
    return summary


def corpus_provenance(corpus_path: Path) -> dict:
    h = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"corpus_file": corpus_path.name,
            "corpus_sha256": h.hexdigest(),
            "corpus_docs": sum(1 for _ in open(corpus_path, encoding="utf-8"))}


# ---------------------------------------------------------------- retrieval

def vector_top_k(vectorstore, query: str, k: int = K) -> list[str]:
    results = vectorstore.similarity_search(query, k=VECTOR_OVERFETCH)
    seen: set = set()
    ranked: list[str] = []
    for r in results:
        doc_id = r.metadata.get("doc_id")
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= k:
            break
    return ranked


def bm25_top_k(bm25, docs, query: str, k: int = K) -> list[str]:
    return [r["doc_id"] for r in bm25_search(bm25, docs, query, k=k)]


def run_eval(golden_path: Path = GOLDEN_SET_PATH, corpus_path: Path | None = None,
             persist_dir: Path | None = None, out_path: Path = DEFAULT_OUT,
             settings: Settings | None = None):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = settings or Settings.from_env()
    corpus_path = Path(corpus_path or settings.corpus_path)
    persist_dir = Path(persist_dir or settings.chroma_dir)

    golden_set = json.loads(Path(golden_path).read_text())
    log.info("Loaded %d golden queries from %s", len(golden_set), golden_path)

    if not (persist_dir / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"No vector index at {persist_dir}. Build one first: "
            f"python -m vectorstore.build_index --corpus {corpus_path} --persist-dir {persist_dir}")
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name=settings.embedding_model),
        persist_directory=str(persist_dir),
    )
    n_vectors = vectorstore._collection.count()

    docs = load_corpus(corpus_path)
    text_by_id = {d["doc_id"]: d["text"] for d in docs}
    missing = {r for item in golden_set for r in item["relevant_doc_ids"]} - set(text_by_id)
    if missing:
        raise ValueError(
            f"Golden set references doc_ids not in {corpus_path}: {sorted(missing)[:5]}... "
            "The golden set is tied to a specific corpus snapshot; compare the corpus_sha256 in eval/eval_results.json.")
    bm25, docs = build_bm25_index(docs)

    per_query = []
    for item in golden_set:
        query, relevant = item["query"], item["relevant_doc_ids"]
        ranked = {"vector": vector_top_k(vectorstore, query),
                  "bm25": bm25_top_k(bm25, docs, query)}
        ranked["hybrid"] = rrf_hybrid(ranked["vector"], ranked["bm25"])

        row = {
            "id": item["id"], "query": query,
            "query_type": item.get("query_type", ""),
            "group": query_group(item),
            "relevant_doc_ids": relevant,
            **{f"{m}_recall": round(recall_at_k(ranked[m], relevant), 3) for m in METHODS},
        }
        if item.get("relevant_text_contains"):
            row["attr_precision"] = {
                m: round(attribute_precision_at_k(ranked[m], text_by_id,
                                                  item["relevant_text_contains"]), 3)
                for m in METHODS}
        per_query.append(row)
        log.info("[%s] vector=%.2f bm25=%.2f hybrid=%.2f -- %s", item["id"],
                 row["vector_recall"], row["bm25_recall"], row["hybrid_recall"], query)

    summary = summarize(per_query)
    result = {
        "summary": summary,
        "provenance": {
            **corpus_provenance(corpus_path),
            "vectors_in_index": n_vectors,
            "embedding_model": settings.embedding_model,
            "k": K, "vector_overfetch": VECTOR_OVERFETCH, "rrf_constant": RRF_CONSTANT,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "per_query": per_query,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote eval results to %s", out_path)
    return summary, per_query


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Recall@10 evaluation: vector vs BM25 vs RRF hybrid.")
    p.add_argument("--golden", type=Path, default=GOLDEN_SET_PATH)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--persist-dir", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    summary, _ = run_eval(args.golden, args.corpus, args.persist_dir, args.out)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
