"""
Aggregates token usage / latency from the trace files the agent writes
(TRACE_DIR, default logs/traces) and projects cost per 100 queries.

    python -m eval.cost_report [--out path.json]

Notes on what is and is not measured
- `latency_seconds` in a trace is the wall-clock time of the LLM call only.
  Retrieval time (embedding the query + Chroma lookup) is not included.
- Prices below are the Groq on-demand rates recorded by the author in
  2026-09 and are NOT re-verified by this script; check Groq's pricing page
  before relying on the projection. The free tier is rate-limited, not
  guaranteed.
"""
import argparse
import json
from pathlib import Path

from config import Settings

PRICE_PER_M_INPUT = 0.15   # USD per 1M input tokens (recorded 2026-09, unverified since)
PRICE_PER_M_OUTPUT = 0.60  # USD per 1M output tokens


def load_traces(trace_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(trace_dir.glob("trace_*.json"))]


def compute_cost(input_tokens: float, output_tokens: float) -> float:
    return (input_tokens / 1_000_000) * PRICE_PER_M_INPUT + (output_tokens / 1_000_000) * PRICE_PER_M_OUTPUT


def build_report(traces: list[dict]) -> dict:
    n = len(traces)
    usage = [t["token_usage"] for t in traces]
    avg = lambda key: sum(u.get(key, 0) for u in usage) / n
    return {
        "n_queries_measured": n,
        "avg_input_tokens_per_query": round(avg("input_tokens"), 1),
        "avg_output_tokens_per_query": round(avg("output_tokens"), 1),
        "avg_reasoning_tokens_per_query": round(avg("reasoning_tokens"), 1),
        "avg_llm_call_latency_seconds": round(avg("latency_seconds"), 2),
        "projected_cost_per_100_queries_usd_paid_tier": round(
            compute_cost(avg("input_tokens"), avg("output_tokens")) * 100, 4),
        "price_basis": f"${PRICE_PER_M_INPUT}/1M input, ${PRICE_PER_M_OUTPUT}/1M output (recorded 2026-09; re-check before relying)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    trace_dir = Settings.from_env().trace_dir
    traces = load_traces(trace_dir) if trace_dir and trace_dir.exists() else []
    if not traces:
        raise SystemExit(f"No trace files in {trace_dir}. Run an investigation first "
                         "(python -m agent.graph) with TRACE_DIR enabled.")
    report = build_report(traces)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
