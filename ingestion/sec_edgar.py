"""
SEC EDGAR Full-Text Search (EFTS) ingestion.

Source (verified 2026-09):
  https://efts.sec.gov/LATEST/search-index?q=...&forms=...

Real constraints discovered through actual testing against this API
(not assumed from documentation alone):
  - efts.sec.gov (the search index) has no bot-detection layer -- any
    User-Agent string works there.
  - www.sec.gov/Archives (the actual filing documents) DOES run bot
    detection, and it specifically checks whether the User-Agent looks
    like a real declared contact identity (name + email), not just
    whether the header is present. A placeholder string without an "@"
    triggered an explicit "Undeclared Automated Tool" block page; a
    real name+email string returned 200. Confirmed via direct testing,
    2026-09-15.
  - Fair-access rate limit: 10 requests/second (SEC published policy).
    This script sleeps well under that.

Entity-category framing: queries below target FILING CONTENT about
categories relevant to this project's cluster types (enforcement
actions, compliance orders involving money-transmission/virtual-currency
businesses) -- never a named private individual.
"""
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from config import Settings, require_contact_user_agent

log = logging.getLogger("sec_edgar")

SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"



def _headers() -> dict:
    return {"User-Agent": require_contact_user_agent(Settings.from_env()),
            "Accept": "application/json"}


ENTITY_CATEGORY_QUERIES = [
    {"q": "\"virtual currency exchange\" enforcement", "forms": ""},
    {"q": "\"money services business\" \"consent order\"", "forms": ""},
    {"q": "\"anti-money laundering\" \"cease and desist\"", "forms": ""},
    {"q": "\"digital asset\" \"suspicious activity report\"", "forms": ""},
]

RATE_LIMIT_SECONDS = 0.4
PAGE_SIZE = 100
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


def search(query: dict, max_pages: int = 1) -> list[dict]:
    """max_pages=1 by default -- deliberately small for this first real
    verification run. Raise it once we've confirmed the full pipeline
    works end to end without silent failures."""
    hits = []
    for page in range(max_pages):
        params = {"q": query["q"], "from": page * PAGE_SIZE}
        if query.get("forms"):
            params["forms"] = query["forms"]

        resp = None
        for attempt in range(MAX_RETRIES + 1):
            resp = requests.get(SEARCH_URL, headers=_headers(), params=params, timeout=30)
            if resp.status_code == 200:
                break
            log.warning("Query %r page %d got %d (attempt %d/%d): %s",
                        query["q"], page, resp.status_code, attempt + 1, MAX_RETRIES + 1, resp.text[:200])
            time.sleep(1.0 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            log.error("Giving up on query %r page %d after retries -- SKIPPING, not crashing the run",
                       query["q"], page)
            break

        data = resp.json()
        page_hits = data.get("hits", {}).get("hits", [])
        if not page_hits:
            break
        hits.extend(page_hits)
        time.sleep(RATE_LIMIT_SECONDS)
    return hits


def filing_url(hit: dict) -> str | None:
    """Direct URL of the filing document (what a reader should be linked to),
    built from the hit's _id ("{accession-with-dashes}:{filename}") and CIK."""
    _id = hit.get("_id", "")
    ciks = hit.get("_source", {}).get("ciks", [])
    if ":" not in _id or not ciks:
        return None
    adsh_dashed, filename = _id.split(":", 1)
    return f"{ARCHIVE_BASE}/{ciks[0].lstrip('0') or '0'}/{adsh_dashed.replace('-', '')}/{filename}"


def fetch_filing_text(hit: dict, max_chars: int = 20000) -> str:
    """Fetch the actual filing document body. The search-index response has
    NO body/excerpt field (verified against a real hit) -- _id is
    "{accession_no_dashes}:{filename}", which is what we need to build the
    real Archives URL and pull real text."""
    url = filing_url(hit)
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        time.sleep(RATE_LIMIT_SECONDS)
        return resp.text[:max_chars]
    except requests.RequestException as e:
        log.warning("Could not fetch filing body %s: %s", url, e)
        return ""


def normalize(all_hits: list, fetch_bodies: bool = True) -> list:
    now = datetime.now(timezone.utc).isoformat()
    docs, seen = [], set()
    for hit in all_hits:
        src = hit.get("_source", {})
        doc_id = f"sec-edgar-{hit.get('_id')}"
        if doc_id in seen:
            continue
        seen.add(doc_id)

        body = fetch_filing_text(hit) if fetch_bodies else ""
        cik = (src.get("ciks") or [""])[0]

        docs.append(
            NormalizedDoc(
                doc_id=doc_id,
                source="SEC_EDGAR",
                entity_type="filing",
                title=f"{src.get('form','?')} — {', '.join(src.get('display_names', []))}",
                programs=src.get("root_forms", []),
                text=body,
                url=filing_url(hit) or f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}",
                retrieved_at=now,
            )
        )
    return docs


def run(out_dir: str = "data/processed", fetch_bodies: bool = True):
    all_hits, per_query_counts = [], {}
    for q in ENTITY_CATEGORY_QUERIES:
        log.info("Querying EDGAR full-text search: %s", q["q"])
        hits = search(q)
        per_query_counts[q["q"]] = len(hits)
        all_hits.extend(hits)

    docs = normalize(all_hits, fetch_bodies=fetch_bodies)
    empty_text_count = sum(1 for d in docs if not d.text)
    out_path = Path(out_dir) / "sec_edgar_normalized.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")
    log.info("Wrote %d normalized SEC EDGAR docs to %s (%d have EMPTY text -- flag these, don't hide them)",
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