export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const MAX_QUERY_CHARS = 300;
export const MIN_QUERY_CHARS = 10;

export type Source = {
  index: number;
  doc_id: string | null;
  source: string | null;
  title: string | null;
  url: string | null;
  excerpt: string;
  cited: boolean;
};

export type Investigation = {
  entity_cluster_id: string;
  report: string;
  citations: string[];
  sources: Source[];
  token_usage: { latency_seconds?: number; total_tokens?: number };
  mode: "live" | "prerecorded";
  notice: string | null;
};

export class ApiError extends Error {
  constructor(message: string, public kind: "validation" | "guardrail" | "rate_limit" | "unavailable" | "not_found" | "network" | "unknown", public retryAfter?: number) {
    super(message);
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_URL}${path}`, { ...init, signal: AbortSignal.timeout(60_000) });
  } catch {
    throw new ApiError(
      `Could not reach the backend at ${API_URL}. It may be offline, waking up, or blocking this site's origin (CORS).`,
      "network",
    );
  }
}

export async function fetchExamples(): Promise<{ mode: string; examples: string[] }> {
  const res = await request("/examples");
  if (!res.ok) throw new ApiError("Could not load examples", "unknown");
  return res.json();
}

export async function investigate(query: string): Promise<Investigation> {
  const res = await request("/investigate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entity_cluster_id: query }),
  });
  if (res.ok) return res.json();

  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : "Request was rejected as invalid.";
  } catch {
    /* non-JSON error body */
  }
  switch (res.status) {
    case 400:
      return Promise.reject(new ApiError(detail, "guardrail"));
    case 404:
      return Promise.reject(new ApiError(detail, "not_found"));
    case 422:
      return Promise.reject(new ApiError(`Please enter ${MIN_QUERY_CHARS}-${MAX_QUERY_CHARS} characters describing an entity category.`, "validation"));
    case 429:
      return Promise.reject(new ApiError(detail, "rate_limit", Number(res.headers.get("Retry-After")) || undefined));
    case 502:
    case 503:
      return Promise.reject(new ApiError(detail, "unavailable"));
    default:
      return Promise.reject(new ApiError(`Unexpected error (HTTP ${res.status}).`, "unknown"));
  }
}
