# Backend API image. Serves /investigate, /health, /ready.
# It contains NO secrets, NO ingested corpus and NO vector index:
#   - pass GROQ_API_KEY etc. as environment variables at run time
#   - provision the index separately (see docs/DEPLOYMENT.md)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    CHROMA_PERSIST_DIR=/data/chroma \
    TRACE_DIR= \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only PyTorch (the default Linux wheel bundles CUDA and is ~2 GB larger).
COPY requirements.txt .
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the embedding model into the image so cold starts do not download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY config.py ./
COPY agent ./agent
COPY api ./api
COPY vectorstore ./vectorstore
COPY eval ./eval
COPY ingestion ./ingestion
COPY demo ./demo

RUN useradd --create-home --uid 10001 app && mkdir -p /data/chroma && chown -R app /data /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')"

# `sh -c` so $PORT (set by most PaaS hosts) is honoured; `exec` so signals reach uvicorn.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
