"""
Build a persistent ChromaDB vector store from the Stage 1 cleaned corpus,
using local HuggingFace embeddings (all-MiniLM-L6-v2), per the project
spec.

Chunking decision (real, not left implicit): OFAC entries are short
(1-3 sentences) and are embedded whole. SEC EDGAR filings and
CourtListener opinions can run to thousands of characters even after
HTML stripping -- embedding one of these as a single vector would
average away the specific paragraph a query actually cares about. Long
documents are split into ~500-token chunks with 50-token overlap
(RecursiveCharacterTextSplitter, tuned for prose) before embedding.
Each chunk keeps a pointer back to its parent doc_id and source, so a
retrieval hit can always be traced back to the exact original document
for citation (Stage 3 requirement).
"""
import json
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_index")

CORPUS_PATH = "data/processed/corpus_final.jsonl"
PERSIST_DIR = "vectorstore/chroma_db"
COLLECTION_NAME = "osint_entity_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Docs longer than this (in characters) get chunked; shorter ones (like
# most OFAC entries) are embedded whole.
CHUNK_THRESHOLD_CHARS = 800
CHUNK_SIZE = 500       # roughly tokens, via character-based splitter with prose-tuned separators
CHUNK_OVERLAP = 50


def load_corpus(path: str = CORPUS_PATH) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    log.info("Loaded %d documents from cleaned corpus", len(docs))
    return docs


def build_langchain_documents(raw_docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    lc_docs, n_chunked, n_whole = [], 0, 0

    for d in raw_docs:
        text = d.get("text", "")
        if not text.strip():
            continue  # skip anything with no real content

        metadata = {
            "doc_id": d["doc_id"],
            "source": d["source"],
            "entity_type": d.get("entity_type", ""),
            "title": d.get("title", ""),
            "url": d.get("url", ""),
        }

        if len(text) > CHUNK_THRESHOLD_CHARS:
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_of"] = len(chunks)
                lc_docs.append(Document(page_content=chunk, metadata=chunk_meta))
            n_chunked += 1
        else:
            chunk_meta = dict(metadata)
            chunk_meta["chunk_index"] = 0
            chunk_meta["chunk_of"] = 1
            lc_docs.append(Document(page_content=text, metadata=chunk_meta))
            n_whole += 1

    log.info("Prepared %d embeddable chunks from %d source docs (%d chunked, %d kept whole)",
              len(lc_docs), len(raw_docs), n_chunked, n_whole)
    return lc_docs


def build_index():
    raw_docs = load_corpus()
    lc_docs = build_langchain_documents(raw_docs)

    log.info("Loading embedding model: %s (this downloads the model on first run)", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    log.info("Embedding %d chunks and writing to ChromaDB at %s ...", len(lc_docs), PERSIST_DIR)
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    log.info("Index built. Collection '%s' now has %d vectors.",
              COLLECTION_NAME, vectorstore._collection.count())
    return {
        "source_docs": len(raw_docs),
        "embedded_chunks": len(lc_docs),
        "persist_dir": PERSIST_DIR,
        "collection_name": COLLECTION_NAME,
    }


if __name__ == "__main__":
    stats = build_index()
    print(json.dumps(stats, indent=2))