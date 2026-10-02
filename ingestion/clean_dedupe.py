"""
Shared cleaning + deduplication pass, run after the source-specific
ingestion scripts (ofac_sdn.py, sec_edgar.py, courtlistener.py) produce
their *_normalized.jsonl files in data/processed/.
"""
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from config import Settings

log = logging.getLogger("clean_dedupe")

# Sources whose documents are short, field-labelled records ("Type: vessel",
# "Program(s): IRAN"). Their repeated lines ARE the signal, so the
# repeated-line boilerplate stripper must not touch them. (Before this was
# added, it deleted the "Type:" line from 1540 of 1882 OFAC documents.)
STRUCTURED_SOURCES = {"OFAC_SDN"}

WHITESPACE_RUN = re.compile(r"\s{2,}")
SINGLE_CHAR_LINE = re.compile(r"^\W$")


def strip_html(docs: list[dict]) -> list[dict]:
    stripped_count = 0
    for d in docs:
        text = d["text"]
        upper = text.upper()
        looks_like_html = "<" in text and ">" in text and (
            "<P " in upper or "<TABLE" in upper or "<HTML" in upper or "<DOCUMENT>" in upper
        )
        if looks_like_html:
            soup = BeautifulSoup(text, "html.parser")
            plain = soup.get_text(separator=" ")
            plain = " ".join(plain.split())
            d["text"] = plain
            d["html_stripped"] = True
            stripped_count += 1
        else:
            d["html_stripped"] = False
    log.info("HTML stripping: %d of %d documents had markup removed", stripped_count, len(docs))
    return docs


def _remove_exact_dupes_only(docs: list[dict]) -> list[dict]:
    seen, unique = {}, []
    for d in docs:
        h = hashlib.sha256(d["text"].strip().encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen[h] = d["doc_id"]
        unique.append(d)
    return unique


def strip_boilerplate(docs: list[dict]) -> list[dict]:
    by_source = defaultdict(list)
    for d in docs:
        by_source[d["source"]].append(d)

    for source, group in by_source.items():
        if source in STRUCTURED_SOURCES:
            continue
        line_counts = Counter()
        for d in group:
            for line in set(d["text"].splitlines()):
                line_counts[line.strip()] += 1
        n = len(group)
        boilerplate_lines = {
            line for line, count in line_counts.items()
            if line and n > 3 and count / n > 0.30
        }
        for d in group:
            kept = [ln for ln in d["text"].splitlines() if ln.strip() not in boilerplate_lines]
            d["text"] = "\n".join(kept)
        if boilerplate_lines:
            log.info("Source %s: stripped %d boilerplate line-patterns across %d docs",
                      source, len(boilerplate_lines), n)
    return docs


def clean_ocr_noise(docs: list[dict]) -> list[dict]:
    for d in docs:
        lines = d["text"].splitlines()
        lines = [ln for ln in lines if not SINGLE_CHAR_LINE.match(ln.strip())]
        text = WHITESPACE_RUN.sub(" ", "\n".join(lines))
        non_ascii = sum(1 for c in text if ord(c) > 127)
        d["ocr_suspect"] = bool(text) and (non_ascii / max(len(text), 1)) > 0.05
        d["text"] = text
    return docs


def _shingles(text: str, k: int = 5) -> set:
    words = text.lower().split()
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 0))}


def dedupe(docs: list[dict], jaccard_threshold: float = 0.85) -> list[dict]:
    for d in docs:
        d["duplicate_of"] = None

    shingle_sets = {d["doc_id"]: _shingles(d["text"]) for d in docs}
    ids = list(shingle_sets.keys())
    dup_of = {}
    near_dupes = 0
    for i in range(len(ids)):
        if ids[i] in dup_of:
            continue
        for j in range(i + 1, len(ids)):
            if ids[j] in dup_of:
                continue
            a, b = shingle_sets[ids[i]], shingle_sets[ids[j]]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= jaccard_threshold:
                dup_of[ids[j]] = ids[i]
                near_dupes += 1

    for d in docs:
        if d["doc_id"] in dup_of:
            d["duplicate_of"] = dup_of[d["doc_id"]]

    log.info("Near-duplicate pass: %d near-duplicates flagged (kept, tagged) out of %d docs",
              near_dupes, len(docs))
    return docs


def run(processed_dir: str | Path | None = None):
    p = Path(processed_dir) if processed_dir else Settings.from_env().data_dir
    all_docs = []
    per_file_counts = {}
    for f in p.glob("*_normalized.jsonl"):
        with open(f, encoding="utf-8") as fh:
            docs = [json.loads(line) for line in fh if line.strip()]
        per_file_counts[f.name] = len(docs)
        all_docs.extend(docs)

    if not all_docs:
        raise FileNotFoundError(
            f"No *_normalized.jsonl files in {p}. Run the ingestion scripts first "
            "(python -m ingestion.ofac_sdn / sec_edgar / courtlistener).")
    log.info("Loaded %d raw normalized docs from %d files: %s", len(all_docs), len(per_file_counts), per_file_counts)

    n_before = len(all_docs)
    all_docs = _remove_exact_dupes_only(all_docs)
    log.info("Exact-dupe pre-pass: %d -> %d docs", n_before, len(all_docs))

    all_docs = strip_html(all_docs)
    all_docs = strip_boilerplate(all_docs)
    all_docs = clean_ocr_noise(all_docs)
    final = dedupe(all_docs)

    out_path = p / "corpus_final.jsonl"
    with open(out_path, "w") as f:
        for d in final:
            f.write(json.dumps(d) + "\n")
    log.info("Wrote final cleaned corpus: %d docs -> %s", len(final), out_path)
    return {
        "input_files": per_file_counts,
        "total_raw": n_before,
        "final_corpus_size": len(final),
        "out_path": str(out_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run()
    print(json.dumps(stats, indent=2))