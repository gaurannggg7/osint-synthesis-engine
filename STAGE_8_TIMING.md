# Stage 8 — Manual vs. Agent-Assisted Timing Comparison

## Method (stated upfront, limitations included)
This is a single, self-conducted trial (N=1), not a controlled or
blinded study. The person conducting the manual trial is also this
project's builder, which is a real, acknowledged bias risk in EITHER
direction -- familiarity with the entity profile and target documents
could make manual search faster (already knowing what to look for) or
slower (overthinking, trying to be unusually thorough to seem fair).
This number should be read as illustrative, not statistically valid.

Both conditions used the same real task: entity profile "cryptocurrency
exchange with prior AML compliance actions and sanctions exposure,"
same three sources (OFAC, SEC EDGAR, CourtListener), same expected
output (a short structured, cited summary).

## Manual baseline (real, timed)
- **Elapsed time: ~18-22 minutes** for a realistic, moderately-fast pass
  (top 2-3 results per search only, consistent with how a real analyst
  under time pressure would actually search -- not exhaustively reading
  every result).
- The person conducting the trial estimated an additional ~10 minutes
  would be needed for a more careful, thorough pass (reading more
  results per query, verifying relevance more carefully before
  stopping). **Realistic estimate for a genuinely careful manual pass:
  ~30 minutes.**
- Real friction encountered during the trial, recorded as it happened
  (not reconstructed afterward):
  - OFAC's public search tool is a name-matching tool, not a full-text/
    category search tool. Searching "virtual currency" returned 0
    results -- a real dead end requiring the researcher to recognize
    the tool's limitation and move on.
  - SEC EDGAR's search box does not default to exact-phrase matching.
    An unquoted query ("virtual currency exchange") returned 10,000
    loosely-related results; only adding literal quote marks around
    the phrase brought this down to a usable 215 results. A manual
    researcher unfamiliar with this quirk would lose meaningful time
    here.
  - Manual search DID surface two real, relevant leads the agent's
    automated retrieval had NOT surfaced in its top-5 results:
    Coincheck Group N.V. (a real cryptocurrency exchange with
    recurring SEC filings) and United States v. Harmon (the "Helix"
    mixing-service prosecution) -- both genuinely on-topic and missed
    by the agent. This is an honest, useful finding: manual search
    is not strictly dominated by the agent on recall for this query.

## Agent-assisted (real, measured in Stage 7)
- **Elapsed time: 3.45 seconds average** (measured across 4 real runs,
  Stage 7), covering retrieval + LLM synthesis end to end.

## Stated comparison (with explicit caveats)
| | Manual (self-timed, N=1) | Agent-assisted (measured, N=4) |
|---|---|---|
| Time | ~18-30 minutes | ~3.45 seconds |
| Approx. reduction | ~99.7-99.8% | -- |

**This reduction percentage is reported with the following explicit
caveats, and should not be quoted without them:**
1. N=1 for the manual condition; not independently replicated.
2. Not blinded; the manual trial was conducted by this project's
   builder, a real source of bias in either direction.
3. The manual trial found genuinely relevant documents (Coincheck,
   Harmon/Helix) that the agent's retrieval missed -- so while the
   agent is dramatically faster, it is not shown here to be more
   thorough or more accurate; speed and recall are different axes,
   and this comparison only measures speed.
4. This replaces the prior, unverified "90% synthesis time reduction"
   claim referenced in this project's original brief. Unlike that
   number, this one is grounded in an actual timed, documented trial
   -- but it is still a single trial, and is reported as such rather
   than as a validated benchmark.

## What is honestly resume-claimable
- "Measured agent-assisted synthesis time (3.45s average, n=4 real
  runs) against a self-timed manual research baseline (~20-30 min,
  single trial) for the same entity-category research task across
  three live public-record sources."
- NOT claimable: a precise, validated "% time reduction" figure without
  the N=1/self-conducted caveats attached every time it's cited.