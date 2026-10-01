# Stage 5 — Retrieval Evaluation Results

## Method
17-query golden set, built backwards from real documents already
confirmed present in the corpus (not written first and hoped to match).
Each query paired with 1-3 known-relevant doc_ids. Recall@10 measured
for: pure vector (all-MiniLM-L6-v2 + ChromaDB), pure BM25 keyword
baseline (rank_bm25, same corpus), and RRF hybrid (Reciprocal Rank
Fusion, k=60, combining both rankings by rank position rather than
raw score, since cosine similarity and BM25 scores are not on
comparable scales).

## Headline numbers

| Method | Mean Recall@10 (n=17) |
|---|---|
| Vector | 0.471 |
| BM25   | 0.412 |
| Hybrid (RRF) | 0.529 |

Hybrid outperforms both individual methods, as expected when combining
independent signal sources.

## The real finding: source-dependent retrieval failure

The aggregate numbers above hide a large, consistent gap that only
appears when the results are broken down by source:

| Query group | Vector | BM25 | Hybrid |
|---|---|---|---|
| OFAC SDN (n=7) | 0.143 | 0.000 | 0.143 |
| SEC EDGAR / CourtListener (n=10) | 0.700 | 0.700 | 0.800 |

BM25 recall on OFAC queries was **exactly zero** -- not one OFAC query's
relevant document appeared anywhere in BM25's top 10, across 7 distinct
queries. Vector search fared only slightly better (1 of 7).

## Root cause (hypothesis, not yet fixed)

OFAC entries in the corpus are short, terse, field-labeled records
(e.g. `"Name: VANITY\nProgram(s): IRAN-EO13902\nVessel type: Crude Oil
Tanker..."`), typically under 50 words, in contrast to SEC/CourtListener
documents which run to hundreds or thousands of words of natural
prose. Two plausible, non-exclusive explanations:

1. **BM25 term-frequency dynamics break down on very short documents.**
   BM25's scoring formula includes a document-length normalization term;
   behavior on 50-word structured records has not been separately tuned
   or verified here, and may not behave as expected out of the box.
2. **Semantic embeddings may encode less useful signal from terse,
   field-labeled text** than from flowing prose, since the model was
   trained predominantly on natural language.

This has NOT been fixed in this stage. Per the project's own standard
(verify before moving on, flag anything that looks better than it is),
this is reported as an open, quantified limitation rather than silently
patched or hidden behind the healthier aggregate number.

## What is honestly resume-claimable from this stage

- "Built a 17-query golden evaluation set grounded in real corpus
  documents, and implemented recall@10 evaluation across vector, BM25,
  and RRF-hybrid retrieval methods."
- "Identified and quantified a significant source-dependent retrieval
  gap (OFAC records: 0.14 recall@10 vs. 0.70-0.80 for prose-based
  sources), pointing to a specific, named root-cause hypothesis for
  future improvement work."

What is NOT claimable: "high-performing retrieval system" or a single
headline recall number without this breakdown -- the aggregate 0.47-0.53
range is real but conceals the OFAC weakness, and citing it alone would
misrepresent the system's actual behavior.

## Suggested future work (not done, explicitly out of scope for this build)
- Test OFAC-specific chunking/embedding: e.g. converting terse fields
  into a fuller natural-language sentence before embedding
  ("VANITY is a Crude Oil Tanker sanctioned under the Iran-EO13902
  program...") rather than embedding the raw field-labeled text.
- Test a smaller BM25 k1/b parameter tuned for short documents.