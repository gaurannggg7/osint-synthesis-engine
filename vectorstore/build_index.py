"""
Build a persistent ChromaDB vector store from the cleaned corpus, using
local HuggingFace embeddings (all-MiniLM-L6-v2). Runs offline, never as part
of a user request.

    python -m vectorstore.build_index                       # full corpus
    python -m vectorstore.build_index --corpus demo/sample_corpus.jsonl

Chunking: OFAC entries are short (1-3 sentences) and are embedded whole.
SEC EDGAR filings and CourtListener opinions can run to thousands of
characters, so documents longer than CHUNK_THRESHOLD_CHARS are split with a
RecursiveCharacterTextSplitter (chunk_size=500 *characters*, overlap=50)
before embedding. Each chunk keeps its parent doc_id/source/url so a
retrieval hit can always be traced to the original document.

The build is idempotent: the target collection is dropped and recreated,
and chunk IDs are deterministic (`<doc_id>::<chunk_index>`), so re-running
never duplicates vectors.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from config import Settings
from vectorstore.corpus import load_corpus

log = logging.getLogger("build_index")

# Docs longer than this (in characters) get chunked; shorter ones are embedded whole.
CHUNK_THRESHOLD_CHARS = 800
CHUNK_SIZE = 500       # characters (the earlier comments/docs said "tokens"; the splitter counts characters)
CHUNK_OVERLAP = 50


def build_langchain_documents(raw_docs: list[dict]):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs, n_chunked, n_whole = [], 0, 0
    seen_ids: set[str] = set()

    for d in raw_docs:
        text = d.get("text", "")
        if not text.strip():
            continue  # skip anything with no real content
        if d["doc_id"] in seen_ids:
            raise ValueError(f"Duplicate doc_id in corpus: {d['doc_id']}")
        seen_ids.add(d["doc_id"])

        metadata = {
            "doc_id": d["doc_id"],
            "source": d["source"],
            "entity_type": d.get("entity_type", ""),
            "title": d.get("title", ""),
            "url": d.get("url", ""),
        }

        pieces = splitter.split_text(text) if len(text) > CHUNK_THRESHOLD_CHARS else [text]
        for i, piece in enumerate(pieces):
            lc_docs.append(Document(
                page_content=piece,
                metadata={**metadata, "chunk_index": i, "chunk_of": len(pieces)},
            ))
        if len(pieces) > 1:
            n_chunked += 1
        else:
            n_whole += 1

    log.info("Prepared %d embeddable chunks from %d source docs (%d chunked, %d kept whole)",
             len(lc_docs), len(raw_docs), n_chunked, n_whole)
    return lc_docs


def build_index(corpus_path: Path | None = None, persist_dir: Path | None = None,
                settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    corpus_path = Path(corpus_path or settings.corpus_path)
    persist_dir = Path(persist_dir or settings.chroma_dir)

    raw_docs = load_corpus(corpus_path)
    lc_docs = build_langchain_documents(raw_docs)
    if not lc_docs:
        raise ValueError(f"No embeddable documents in {corpus_path}")
    ids = [f"{d.metadata['doc_id']}::{d.metadata['chunk_index']}" for d in lc_docs]

    import chromadb
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    log.info("Loading embedding model: %s (downloads on first run)", settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(settings.collection_name)
        log.info("Dropped existing collection '%s' (rebuilding)", settings.collection_name)
    except Exception:  # collection does not exist yet
        pass

    log.info("Embedding %d chunks into ChromaDB at %s ...", len(lc_docs), persist_dir)
    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        ids=ids,
        collection_name=settings.collection_name,
        client=client,
    )
    count = vectorstore._collection.count()
    log.info("Index built. Collection '%s' has %d vectors.", settings.collection_name, count)
    return {
        "corpus_path": str(corpus_path),
        "source_docs": len(raw_docs),
        "embedded_chunks": len(lc_docs),
        "vectors_in_collection": count,
        "persist_dir": str(persist_dir),
        "collection_name": settings.collection_name,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the Chroma vector index (offline step).")
    parser.add_argument("--corpus", type=Path, help="JSONL corpus (default: $CORPUS_PATH or data/processed/corpus_final.jsonl)")
    parser.add_argument("--persist-dir", type=Path, help="Chroma directory (default: $CHROMA_PERSIST_DIR)")
    args = parser.parse_args()
    print(json.dumps(build_index(args.corpus, args.persist_dir), indent=2))


if __name__ == "__main__":
    main()
