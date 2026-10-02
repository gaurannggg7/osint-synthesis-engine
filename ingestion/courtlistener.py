"""
CourtListener ingestion — free, public API. Used as the legitimate
substitute for raw PACER (see Stage 1 README: PACER charges per page and
requires a paid account; CourtListener/RECAP, run by the nonprofit Free
Law Project, mirrors public RECAP-archived filings and judicial opinions
via a free REST API).

Source: https://www.courtlistener.com/api/rest/v4/search/?q=...&type=o
"""
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from config import Settings, require_contact_user_agent

log = logging.getLogger("courtlistener")

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"



def _headers() -> dict:
    settings = Settings.from_env()
    headers = {"User-Agent": require_contact_user_agent(settings)}
    if settings.courtlistener_token:
        headers["Authorization"] = f"Token {settings.courtlistener_token}"
    return headers


ENTITY_CATEGORY_QUERIES = [
    "cryptocurrency exchange money laundering forfeiture",
    "unlicensed money transmitting business",
    "virtual currency mixing service seizure",
    "bank secrecy act civil penalty exchange",
]

RATE_LIMIT_SECONDS = 1.0
MAX_RETRIES = 2


@dataclass
class NormalizedDoc:
    doc_id: str
    source: str
    entity_type: str
    title: str
    programs: list
    text: str
    url: str
    retrieved_at: str


def search(query: str, max_pages: int = 1) -> list[dict]:
    results = []
    url = SEARCH_URL
    params = {"q": query, "type": "o", "order_by": "score desc"}
    for _ in range(max_pages):
        resp = None
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.get(url, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 200:
                break
            log.warning("Query %r got %d (attempt %d/%d): %s",
                        query, resp.status_code, attempt + 1, MAX_RETRIES + 1, resp.text[:200])
            time.sleep(1.0 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            log.error("Giving up on query %r after retries -- SKIPPING", query)
            break

        data = resp.json()
        results.extend(data.get("results", []))
        next_url = data.get("next")
        if not next_url:
            break
        url, params = next_url, None
        time.sleep(RATE_LIMIT_SECONDS)
    return results


def normalize(all_results: list[dict]) -> list[NormalizedDoc]:
    """Verified against a real CourtListener v4 search response
    (2026-09-18): there is NO top-level 'id' or 'snippet' field. The
    real unique key is 'cluster_id'; snippet text is nested under
    opinions[0]['snippet'] because one case cluster can have multiple
    attached opinions. Using the wrong assumed keys silently collapsed
    76 real hits into 1 document with empty text -- caught by checking
    per_query_hit_counts against normalized_docs, not by inspection."""
    now = datetime.now(timezone.utc).isoformat()
    docs, seen = [], set()
    for r in all_results:
        cluster_id = r.get("cluster_id")
        if cluster_id is None:
            continue  # can't dedupe or link back without a real id
        doc_id = f"courtlistener-{cluster_id}"
        if doc_id in seen:
            continue
        seen.add(doc_id)

        opinions = r.get("opinions", [])
        snippet = opinions[0].get("snippet", "") if opinions else ""

        text_parts = [f"Case: {r.get('caseName', 'Untitled')}"]
        if r.get("court_citation_string"):
            text_parts.append(f"Court: {r['court_citation_string']}")
        if r.get("dateFiled"):
            text_parts.append(f"Filed: {r['dateFiled']}")
        if r.get("docketNumber"):
            text_parts.append(f"Docket: {r['docketNumber']}")
        if r.get("judge"):
            text_parts.append(f"Judge: {r['judge']}")
        if snippet:
            text_parts.append(f"Excerpt: {snippet}")

        docs.append(
            NormalizedDoc(
                doc_id=doc_id,
                source="CourtListener",
                entity_type="opinion",
                title=r.get("caseName", "Untitled opinion"),
                programs=[r.get("court", "")],
                text="\n".join(text_parts),
                url=f"https://www.courtlistener.com{r.get('absolute_url', '')}",
                retrieved_at=now,
            )
        )
    return docs


def run(out_dir: str = "data/processed"):
    all_results, per_query_counts = [], {}
    for q in ENTITY_CATEGORY_QUERIES:
        log.info("Querying CourtListener: %s", q)
        results = search(q)
        per_query_counts[q] = len(results)
        all_results.extend(results)
        time.sleep(RATE_LIMIT_SECONDS)

    docs = normalize(all_results)
    empty_text_count = sum(1 for d in docs if not d.text)
    out_path = Path(out_dir) / "courtlistener_normalized.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")
    log.info("Wrote %d normalized CourtListener docs to %s (%d empty text)",
              len(docs), out_path, empty_text_count)
    return {
        "per_query_hit_counts": per_query_counts,
        "normalized_docs": len(docs),
        "docs_with_empty_text": empty_text_count,
        "out_path": str(out_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run()
    print(json.dumps(stats, indent=2))