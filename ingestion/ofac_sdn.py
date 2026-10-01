"""
OFAC SDN List ingestion.

Source (verified 2026-09, via OFAC's own technical notices — not guessed):
  https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV

Important, documented-by-OFAC quirks this script accounts for:
  1. The SLS host requires a real User-Agent header or it returns 403.
     (OFAC "Technical Notice: 403 Errors on Redirect to OFAC Sanctions
     List Data" — .NET and some HTTP clients omit User-Agent by default.)
  2. The endpoint 302-redirects once to a signed S3 URL. requests handles
     this automatically as long as allow_redirects is left True (default).
  3. SDN.CSV is the PRIMARY file only. Full identification requires the
     companion ADD.CSV (addresses) and ALT.CSV (aliases) files, joined on
     the integer ID in the first column. Downloading SDN.CSV alone silently
     drops alias/address data -- OFAC's own tutorial warns about this.

HARD GUARDRAIL ENFORCEMENT (not optional, not a style choice):
  The SDN list mixes individuals ("Type: individual") and organizational /
  physical targets ("Type: entity", "vessel", "aircraft"). This project's
  scope is entity-category investigation only. This script drops every
  "individual" record at ingestion time -- before it ever reaches the
  vector store -- so a downstream retrieval step can never surface a named
  person's sanctions record.
"""
import csv
import io
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ofac_sdn")

BASE_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports"
FILES = {
    "primary": f"{BASE_URL}/SDN.CSV",
    "addresses": f"{BASE_URL}/ADD.CSV",
    "aliases": f"{BASE_URL}/ALT.CSV",
}
# Required or OFAC's host 403s the request -- documented by OFAC itself.
HEADERS = {
    "User-Agent": "OSINT-Research-Pipeline/1.0 (contact: PUT_YOUR_REAL_EMAIL_HERE)"
}

# SDN.CSV has no header row. Column order per OFAC's SDN data specification.
SDN_COLUMNS = [
    "ent_num", "sdn_name", "sdn_type", "program", "title", "call_sign",
    "vess_type", "tonnage", "grt", "vess_flag", "vess_owner", "remarks",
]
ADD_COLUMNS = ["ent_num", "add_num", "address", "city_state_prov_postal", "country", "add_remarks"]
ALT_COLUMNS = ["ent_num", "alt_num", "alt_type", "alt_name", "alt_remarks"]

# Only these sdn_type values are in scope. "individual" (and blank, which
# OFAC uses for legacy individual rows) is explicitly excluded per the
# hard guardrail above.
ENTITY_TYPES_IN_SCOPE = {"entity", "vessel", "aircraft"}


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


def _fetch_csv(url: str, columns: list) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    reader = csv.reader(io.StringIO(resp.text))
    rows = []
    for row in reader:
        if not row:
            continue
        row = row + [""] * (len(columns) - len(row))
        rows.append(dict(zip(columns, row[: len(columns)])))
    return rows


def fetch_all():
    log.info("Fetching SDN primary file...")
    primary = _fetch_csv(FILES["primary"], SDN_COLUMNS)
    log.info("Fetched %d primary SDN rows", len(primary))

    log.info("Fetching address file...")
    addresses = _fetch_csv(FILES["addresses"], ADD_COLUMNS)
    log.info("Fetched %d address rows", len(addresses))

    log.info("Fetching alias file...")
    aliases = _fetch_csv(FILES["aliases"], ALT_COLUMNS)
    log.info("Fetched %d alias rows", len(aliases))

    return primary, addresses, aliases


def normalize(primary, addresses, aliases):
    addr_by_ent, alias_by_ent = {}, {}
    for a in addresses:
        addr_by_ent.setdefault(a["ent_num"], []).append(a)
    for a in aliases:
        alias_by_ent.setdefault(a["ent_num"], []).append(a)

    docs, excluded_individuals = [], 0
    now = datetime.now(timezone.utc).isoformat()

    for row in primary:
        sdn_type_raw = (row.get("sdn_type") or "").strip().lower()
        if sdn_type_raw not in ENTITY_TYPES_IN_SCOPE:
            excluded_individuals += 1
            continue  # GUARDRAIL: drop individuals here, not downstream

        # ent = row["ent_num"]
        # addr_txt = "; ".join(
        #     f"{a['address']}, {a['city_state_prov_postal']}, {a['country']}".strip(", ")
        #     for a in addr_by_ent.get(ent, [])
        # )
        # alias_txt = "; ".join(a["alt_name"] for a in alias_by_ent.get(ent, []) if a["alt_name"])
                # OFAC uses the literal string "-0-" as a null/blank placeholder in
        # its CSVs. Left in place, it pollutes document text with fake
        # "content" (e.g. "Address(es): -0- , -0- , -0-") that would add
        # noise to embeddings later. Stripped here, at cleaning time, not
        # left for the retrieval stage to somehow learn to ignore.
        def _clean_field(v: str) -> str:
            v = (v or "").strip()
            return "" if v in ("-0-", "") else v

        ent = row["ent_num"]
        addr_txt_parts = []
        for a in addr_by_ent.get(ent, []):
            parts = [_clean_field(a["address"]), _clean_field(a["city_state_prov_postal"]), _clean_field(a["country"])]
            joined = ", ".join(p for p in parts if p)
            if joined:
                addr_txt_parts.append(joined)
        addr_txt = "; ".join(addr_txt_parts)

        alias_txt = "; ".join(
            _clean_field(a["alt_name"]) for a in alias_by_ent.get(ent, []) if _clean_field(a["alt_name"])
        )



        text_parts = [
            f"Name: {row['sdn_name']}",
            f"Type: {sdn_type_raw}",
            f"Program(s): {row['program']}",
        ]
        if row.get("vess_type"):
            text_parts.append(f"Vessel type: {row['vess_type']}, flag: {row.get('vess_flag','')}")
        if alias_txt:
            text_parts.append(f"Also known as: {alias_txt}")
        if addr_txt:
            text_parts.append(f"Address(es): {addr_txt}")
        # if row.get("remarks"):
        #     text_parts.append(f"Remarks: {row['remarks']}")
        remarks_clean = _clean_field(row.get("remarks"))
        if remarks_clean:
            text_parts.append(f"Remarks: {remarks_clean}")

        docs.append(
            NormalizedDoc(
                doc_id=f"ofac-sdn-{ent}",
                source="OFAC_SDN",
                entity_type=sdn_type_raw,
                title=row["sdn_name"],
                programs=[p.strip() for p in row["program"].split(";") if p.strip()],
                text="\n".join(text_parts),
                url="https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV",
                retrieved_at=now,
            )
        )

    log.info(
        "Normalized %d in-scope entity/vessel/aircraft records; excluded %d individual records (guardrail)",
        len(docs), excluded_individuals,
    )
    return docs


def run(out_dir: str = "data/processed"):
    primary, addresses, aliases = fetch_all()
    docs = normalize(primary, addresses, aliases)
    out_path = Path(out_dir) / "ofac_sdn_normalized.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(asdict(d)) + "\n")
    log.info("Wrote %d normalized OFAC docs to %s", len(docs), out_path)
    return {
        "raw_primary_rows": len(primary),
        "raw_address_rows": len(addresses),
        "raw_alias_rows": len(aliases),
        "normalized_docs": len(docs),
        "out_path": str(out_path),
    }


if __name__ == "__main__":
    stats = run()
    print(json.dumps(stats, indent=2))