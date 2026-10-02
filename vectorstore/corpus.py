"""Shared JSONL corpus loader with an actionable error when the corpus is missing."""
import json
import logging
from pathlib import Path

log = logging.getLogger("corpus")


def load_corpus(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Build it with the ingestion pipeline "
            "(python -m ingestion.ofac_sdn / sec_edgar / courtlistener, then "
            "python -m ingestion.clean_dedupe), or use the bundled public-record "
            "sample: --corpus demo/sample_corpus.jsonl"
        )
    docs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {e}") from e
    log.info("Loaded %d documents from %s", len(docs), path)
    return docs
