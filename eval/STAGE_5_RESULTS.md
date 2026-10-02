# Retrieval evaluation results

Reproduce: `python -m eval.run_eval` (needs the full corpus snapshot and index; see
[README](../README.md#evaluation)). Raw numbers, per-query rows and corpus provenance
(file hash, document count, vector count) are in [`eval_results.json`](eval_results.json).

## Method

- 17-query golden set ([`golden_set.json`](golden_set.json)), built backwards from documents
  already present in the corpus, each paired with 1-3 relevant `doc_id`s.
- **Recall@10** for three methods over the same corpus:
  pure vector (all-MiniLM-L6-v2 + ChromaDB, chunks deduplicated to parent documents),
  BM25 (rank_bm25, whitespace tokenizer, untuned), and Reciprocal Rank Fusion (k=60) of the two.
- Corpus snapshot: 2,339 documents (1,882 OFAC SDN, 382 SEC EDGAR, 75 CourtListener),
  8,036 vectors, ingested 2026-09-15 to 2026-09-18.
- The 2026-10-01 re-run on the preserved index reproduced the original headline numbers exactly.

## Measured results (n = 17)

| Method | Mean recall@10 |
|---|---|
| Vector | 0.471 |
| BM25 | 0.412 |
| RRF hybrid | 0.529 |

By query group:

| Group | n | Vector | BM25 | Hybrid |
|---|---|---|---|---|
| OFAC SDN | 7 | 0.143 | 0.000 | 0.143 |
| SEC EDGAR | 7 | 0.714 | 0.571 | 0.714 |
| CourtListener | 3 | 0.667 | 1.000 | 1.000 |

**Hybrid vs. the best single method, per query: 0 wins, 17 ties, 0 losses.**
Hybrid's higher mean than vector alone comes entirely from inheriting BM25's wins
(for example g17, where BM25 finds the document and vector does not). It never beat the
better of the two on any query. The earlier statement that hybrid "outperforms both,
as expected when combining independent signals" is not supported by this data.

## What the OFAC result does and does not show

OFAC recall is near zero (vector 0.143, BM25 0.000). The earlier write-up attributed this to
short, field-labelled documents breaking BM25 length normalisation and embeddings. That
hypothesis was **not** supported when tested:

1. **Label incompleteness dominates.** Each OFAC query has one labelled document, but the corpus
   holds many equally valid answers: 64 OFAC records are "Linked To: NATIONAL IRANIAN TANKER
   COMPANY", 40 are linked to the Venezuelan state oil company, 121 mention DPRK programs, 67
   mention Iran Air. Strict recall@10 against one label is capped by chance for such queries.
2. **A diagnostic metric agrees.** For the 6 OFAC queries with an attribute rule
   (`relevant_text_contains`), the share of the top 10 that satisfies the rule is
   vector 0.667, BM25 0.450, hybrid 0.550. On g01 and g02, both vector and BM25 return ten
   matching documents and still score 0.0 strict recall. These rules were written after seeing
   the results, so treat them as a diagnostic, not a pre-registered metric.
3. **There are genuine misses.** g03 (Venezuelan aeronautics) and g05 (chemical tanker under an
   Iran executive order) score 0 on attribute precision for vector search too. The query says
   "Venezuelan aeronautics industry"; the record says "CONSORCIO VENEZOLANO DE INDUSTRIAS
   AERONAUTICAS". That vocabulary gap is a real, unfixed weakness.
4. **Tokenization is not the cause.** A regex tokenizer in place of whitespace splitting left BM25
   OFAC recall at 0.000.
5. **g01 is a poor label.** Its relevant document is a "Platform Supply Ship", while the query
   asks for an "oil tanker".

## A cleaning bug found during this review (fixed in code, not re-measured)

`ingestion/clean_dedupe.py` removed any line shared by more than 30% of a source's documents.
For OFAC that deleted the `Type: vessel` / `Type: aircraft` line from 1,540 of 1,882 records,
and `-0-` null placeholders remained in 473. Both are now fixed (OFAC is exempt from the
repeated-line stripper; `-0-` is cleaned in the vessel fields) with unit tests. **The numbers above
were measured before the fix** on the original corpus. A BM25-only check that restored the
missing text did not change OFAC recall (still 0.000), so this bug does not explain the headline
gap, but the full pipeline (re-clean, re-index, re-evaluate) has not been re-run.

## Limitations of this evaluation

- 17 queries, one author, no held-out set; differences of one or two queries are noise.
- Golden queries were written from known documents, which favours lexical overlap.
- Relevance labels are incomplete (see above), so recall understates quality on group-style queries.
- Near-duplicate documents (61 flagged) stay in the index.
- Exact reproduction needs the 2026-09 corpus snapshot. Re-running ingestion later returns different
  data, and the golden set's `doc_id`s may not exist. The run script fails loudly in that case.
- `demo/golden_set.json` runs the same harness on a 24-document public sample. It is a smoke test of
  the harness, not a benchmark: with 24 documents, a top-10 list covers a large share of the corpus.

## Not done (candidate next steps)

- Re-run clean, index and eval after the fix and report the change.
- Expand relevance labels for group-style queries (or label by attribute) and add more queries.
- Test rewriting terse OFAC fields into a sentence before embedding; tune BM25 `k1`/`b`.
- Evaluate answer groundedness (whether report claims are supported by the cited excerpt), which
  this retrieval evaluation does not cover.
