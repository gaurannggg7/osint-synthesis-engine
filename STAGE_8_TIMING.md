# Manual vs. agent-assisted timing: one informal trial

This is an anecdote, not a benchmark. Read the caveats before quoting any number.

## Method

A single self-conducted trial (N=1) by the project's author, not blinded. Author familiarity with the
target documents could bias manual time in either direction.

Both conditions used the same task: profile "cryptocurrency exchange with prior AML compliance actions
and sanctions exposure", the same three sources (OFAC, SEC EDGAR, CourtListener), and the same output
(a short, cited summary).

## Manual baseline (self-timed, recorded as it happened)

- About 18-22 minutes for a moderately fast pass (top 2-3 results per search). The author estimated
  roughly 30 minutes for a careful pass; that figure is an estimate, not a second timed trial.
- Friction noted during the trial:
  - OFAC's public search is a name-matching tool, not a category search; "virtual currency" returned 0 results.
  - SEC EDGAR full-text search does not default to phrase matching; an unquoted query returned
    10,000 loosely related results and only a quoted phrase brought it to 215.
- Manual search surfaced two relevant leads the agent's top-5 retrieval had **not** returned
  (Coincheck Group N.V. and a mixing-service prosecution). Manual search found things the agent missed.

## Agent-assisted (measured)

- Mean **LLM-call latency 3.45 s over the first 4 recorded runs**; 3.57 s over all 5 recorded traces
  (`python -m eval.cost_report`, from local trace files on 2026-09-18).
- This is the wall-clock time of the LLM call only. Retrieval time (query embedding plus Chroma
  lookup) is not included, and an earlier version of this note wrongly described it as end-to-end.
  On a cold start the embedding model also has to load; one local run of the API's first request
  took about 7 s before reaching the LLM step (single observation, not a measurement series).

## Comparison

| | Manual (self-timed, N=1) | Agent (LLM call only, N=5) |
|---|---|---|
| Time | ~18-30 min | ~3.6 s |

The ratio is large, but it compares a human reading documents with a model call over five excerpts that
retrieval already selected. It says nothing about quality:

1. N=1 for the manual condition, not replicated, not blinded.
2. The agent missed leads the manual search found, so this measures speed only, not thoroughness or accuracy.
3. The agent's output was not scored for correctness against the manual summary.

No "percent time reduction" figure is claimed. An earlier brief referenced an unverified 90% number; that
number has no support in this project.
