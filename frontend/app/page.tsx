"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ApiError,
  Investigation,
  MAX_QUERY_CHARS,
  MIN_QUERY_CHARS,
  fetchExamples,
  investigate,
} from "@/lib/api";

// Shown if /examples is unreachable. In prerecorded mode only the server's own list works.
const FALLBACK_EXAMPLES = [
  "cryptocurrency exchange with prior AML compliance actions and sanctions exposure",
  "crypto mixing service with links to sanctioned jurisdictions",
  "money services business with prior consent order for AML violations",
  "digital asset exchange under regulatory investigation for compliance failures",
];

// Turn "[1]", "[1, 3]" and fullwidth "【1】" citation markers into links to the matching source card.
function linkCitations(text: string): string {
  return text.replace(/[\[【](\d+(?:\s*[,;，]\s*\d+)*)[\]】]/g, (_m, group: string) =>
    "[" + group.split(/[,;，]/).map((n) => `[${n.trim()}](#source-${n.trim()})`).join(", ") + "]",
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [examples, setExamples] = useState<string[]>(FALLBACK_EXAMPLES);
  const [backendMode, setBackendMode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [result, setResult] = useState<Investigation | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const resultRef = useRef<HTMLElement>(null);

  useEffect(() => {
    fetchExamples()
      .then((d) => {
        if (d.examples.length) setExamples(d.examples);
        setBackendMode(d.mode);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!loading) return;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [loading]);

  const trimmed = query.trim();
  const tooShort = trimmed.length < MIN_QUERY_CHARS;
  const report = useMemo(() => (result ? linkCitations(result.report) : ""), [result]);

  async function run(q: string) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await investigate(q.trim());
      setResult(data);
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError("Something went wrong.", "unknown"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header>
        <h1>Agentic OSINT Analyst</h1>
        <p className="lede">
          Describe a <strong>category</strong> of entity (not a person). The agent retrieves matching excerpts from
          public OFAC, SEC EDGAR and CourtListener records and writes a report in which every claim cites an excerpt.
          The excerpts are shown so you can check the work.
        </p>
        {backendMode === "prerecorded" && (
          <p className="banner banner-demo" role="status">
            <strong>Prerecorded demo.</strong> This deployment replays saved responses from real agent runs. Only the
            example queries below are available.
          </p>
        )}
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!tooShort && !loading) run(query);
        }}
      >
        <label htmlFor="q">Entity-category profile</label>
        <textarea
          id="q"
          rows={3}
          value={query}
          maxLength={MAX_QUERY_CHARS}
          placeholder="e.g. money services business with prior consent order for AML violations"
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="form-row">
          <span className="count" aria-live="polite">
            {trimmed.length}/{MAX_QUERY_CHARS}
          </span>
          <button type="submit" disabled={tooShort || loading}>
            {loading ? "Investigating…" : "Investigate"}
          </button>
        </div>
      </form>

      <section aria-label="Example queries">
        <h2 className="small">Try an example</h2>
        <ul className="chips">
          {examples.map((ex) => (
            <li key={ex}>
              <button
                type="button"
                className="chip"
                disabled={loading}
                onClick={() => {
                  setQuery(ex);
                  run(ex);
                }}
              >
                {ex}
              </button>
            </li>
          ))}
        </ul>
      </section>

      {loading && (
        <div className="status" role="status" aria-live="polite">
          <span className="spinner" aria-hidden="true" />
          Retrieving excerpts and writing the report… {elapsed}s
          {elapsed > 10 && <span className="muted"> (a cold backend can take longer on the first request)</span>}
        </div>
      )}

      {error && (
        <div className="banner banner-error" role="alert">
          <strong>
            {error.kind === "guardrail" && "Request declined. "}
            {error.kind === "rate_limit" && "Slow down. "}
            {error.kind === "unavailable" && "Service unavailable. "}
            {error.kind === "network" && "Backend unreachable. "}
          </strong>
          {error.message}
          {error.kind === "rate_limit" && error.retryAfter ? ` Try again in about ${error.retryAfter}s.` : ""}
        </div>
      )}

      {result && (
        <article ref={resultRef} aria-label="Investigation result">
          {result.mode === "prerecorded" ? (
            <p className="banner banner-demo" role="status">
              <strong>Prerecorded response.</strong> {result.notice}
            </p>
          ) : (
            <p className="banner banner-live" role="status">
              <strong>Live response</strong> generated just now from the retrieved excerpts below.
            </p>
          )}

          <h2>
            Report{" "}
            {result.mode === "prerecorded" && <span className="tag tag-demo">PRERECORDED</span>}
          </h2>
          <div className="report">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
          </div>

          <h2>
            Source evidence{" "}
            {result.mode === "prerecorded" && <span className="tag tag-demo">PRERECORDED</span>}{" "}
            <span className="muted">({result.sources.length} excerpts retrieved, {result.citations.length} documents cited)</span>
          </h2>
          <ol className="sources">
            {result.sources.map((s) => (
              <li key={s.index} id={`source-${s.index}`} className={s.cited ? "cited" : ""}>
                <div className="source-head">
                  <span className="index">[{s.index}]</span>
                  <span className="tag">{s.source}</span>
                  <span className={`tag ${s.cited ? "tag-cited" : "tag-uncited"}`}>
                    {s.cited ? "cited in report" : "retrieved, not cited"}
                  </span>
                </div>
                <p className="source-title">
                  {s.url ? (
                    <a href={s.url} target="_blank" rel="noopener noreferrer">
                      {s.title}
                    </a>
                  ) : (
                    s.title
                  )}
                </p>
                <blockquote>{s.excerpt}</blockquote>
              </li>
            ))}
          </ol>

          <p className="muted footnote">
            LLM call: {result.token_usage.latency_seconds ?? "n/a"}s
            {result.mode === "prerecorded" ? " (when originally recorded)" : ""}. Retrieval time is not included.
          </p>
        </article>
      )}

      <footer>
        <p>
          Limitations: reports are generated by a language model from a small, partial snapshot of public records. They
          can omit relevant documents and are not legal or compliance advice. Source excerpts may incidentally contain
          names of individuals; the system does not try to identify people. A keyword guardrail blocks requests that ask
          to identify a person, but it is not a complete safeguard.
        </p>
      </footer>
    </main>
  );
}
