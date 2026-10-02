# Deployment readiness

Target: a Next.js front end on Vercel and the Python backend hosted separately. Nothing here has been
deployed; no paid services were used.

## Assessment

| Area | Finding |
|---|---|
| API surface | `POST /investigate`, `GET /examples`, `/health`, `/ready` (in prerecorded mode `/ready` is always `ready` and reports `index_present`/`llm_configured` as `null`). Stateless per request; no sessions or writes (traces are off by default in the container). |
| Why the backend cannot run on Vercel | It needs PyTorch + sentence-transformers + Chroma (hundreds of MB), a persistent index on disk, and a model loaded in memory. That does not fit a serverless function's size and cold-start constraints. Host it as a container. |
| Why the front end calls the backend directly | The browser calls `NEXT_PUBLIC_API_URL`; the backend allow-lists the front end's origin (CORS). Nothing secret is in the front end. A Next.js proxy route would hide the backend URL but is not needed for security here. |
| Agent execution time | LLM call mean 3.57 s (5 recorded runs, free-tier Groq, 2026-09-18). Not measured: retrieval time, p95, behaviour under load. First request after start loads the embedding model; one local observation was about 7 s before the LLM step. The front end has a 60 s client timeout and shows elapsed time. The LLM call itself has a 30 s timeout and one retry. |
| Filesystem | Reads: Chroma directory (`CHROMA_PERSIST_DIR`), Hugging Face model cache (`HF_HOME`). Writes: Chroma may touch its SQLite files even for reads, so the index directory must be writable; traces only if `TRACE_DIR` is set (the Docker image sets it empty). Ephemeral container disks lose data on restart, so the index must be baked in or on a volume. |
| Database | ChromaDB (embedded, SQLite-backed), built offline by `python -m vectorstore.build_index`. Not safe for concurrent writers; the API only reads. Run a single index writer, and treat each deployment's index as a deployed artifact. |
| Environment | See the table below. Secrets: `GROQ_API_KEY` only, server-side. |
| Request safety | Input 10-300 characters, control characters rejected, unknown fields rejected; CORS allow-list (wildcard refused at startup); per-IP rate limit (default 10/min); LLM and index errors mapped to generic 502/503 with details only in server logs. |

## Environment

| Variable | Where | Notes |
|---|---|---|
| `GROQ_API_KEY` | backend | secret; not needed in prerecorded mode |
| `CORS_ALLOWED_ORIGINS` | backend | exact origins, comma-separated, e.g. `https://your-app.vercel.app`. Vercel preview URLs change per deployment and must be added explicitly if you want them to work. |
| `RATE_LIMIT_PER_MINUTE`, `TRUST_PROXY_HEADERS` | backend | set `TRUST_PROXY_HEADERS=true` only if the host's proxy overwrites `X-Forwarded-For`; otherwise all users share the proxy's IP and one limit |
| `CHROMA_PERSIST_DIR` | backend | image default `/data/chroma` |
| `DEMO_MODE` | backend | `prerecorded` needs no key and no index |
| `NEXT_PUBLIC_API_URL` | front end (Vercel) | public by design |

## Exact settings for the prerecorded demo (verified locally)

**Backend** (run from the repo root, Python 3.13 environment from `requirements.txt`):

```bash
DEMO_MODE=prerecorded \
CORS_ALLOWED_ORIGINS=https://<your-app>.vercel.app \
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

| Variable | Value | Required |
|---|---|---|
| `DEMO_MODE` | `prerecorded` | yes |
| `CORS_ALLOWED_ORIGINS` | the exact Vercel origin(s), comma-separated, no trailing slash, no `*` | yes (default only allows localhost:3000) |
| `RATE_LIMIT_PER_MINUTE` | default `10` per client IP | no |
| `TRUST_PROXY_HEADERS` | `true` only behind a proxy that overwrites `X-Forwarded-For` | no |
| `GROQ_API_KEY`, `CHROMA_PERSIST_DIR`, index, `HF_HOME` | not used in this mode | no |

In this mode the API imports no torch, chromadb, sentence-transformers or langgraph and never opens a
vector index (checked by `tests/test_api.py::test_prerecorded_mode_loads_no_heavy_dependencies`). A smaller
dependency set would therefore suffice, but only the full `requirements.txt` has been tested.

**Frontend (Vercel project settings):**

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework preset | Next.js (auto-detected) |
| Build / install | defaults (`npm run build` / `npm install`) |
| Environment variable | `NEXT_PUBLIC_API_URL=https://<backend-host>` (no trailing slash) |

`NEXT_PUBLIC_*` values are inlined **at build time**, so changing the API URL requires a redeploy.
Add the Vercel production domain (and any preview domain you want to work) to the backend's
`CORS_ALLOWED_ORIGINS`.

**Checked against a production build** (`next build` + `next start` on :3100, backend on :8000):
`/health`, `/ready`, all four examples return `mode: "prerecorded"` with a notice and 5 sources; every
citation in each report renders as a link to an existing evidence card (one recorded report used fullwidth
`【1】` markers, which the parser originally missed and now handles); CORS allows the configured origin and
sends no CORS header for another origin. The "PRERECORDED" label appears in a banner and next to both the
Report and Source evidence headings.

## Native Python host (no Docker) for the prerecorded demo

Use any host that runs a Python 3.13 web service from a Git repository (Render, Railway, Fly.io buildpacks,
and similar). Nothing here has been deployed.

| Setting | Value |
|---|---|
| Python version | 3.13 (the only version tested) |
| Build command | `pip install -r requirements-demo.txt` |
| Start command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Environment | `DEMO_MODE=prerecorded` and `CORS_ALLOWED_ORIGINS=https://<your-app>.vercel.app` |

`requirements-demo.txt` has four packages (fastapi, uvicorn, pydantic, python-dotenv; 13 installed in total,
about 28 MB). It was derived from the real import graph: `api/main.py`, `agent/graph.py` and `config.py`
import nothing else at module level, and every langchain / langgraph / chroma / torch import sits inside a
function that only live mode calls.

**Verified:** a fresh venv built from `requirements-demo.txt` alone, running a copy of the repo containing
only `config.py`, `agent/`, `api/` and `demo/prerecorded_responses.json` (no `.env`, no index), started with
the start command above and served `/health`, `/ready`, `/examples` and all four examples (each
`mode: "prerecorded"`, 5 sources, all cited), sent the CORS header for the configured origin, and returned
404 for an unrecorded query. A test keeps the demo pins identical to `requirements.txt`.

**Not verified:** any specific hosting provider's build, its port variable name (`$PORT` is the common
convention), and behaviour under load.

**Live mode is unchanged.** `requirements.txt` (full set), the Dockerfile and all live code paths are
untouched; switch a host to live mode by installing `requirements.txt`, provisioning an index and setting
`GROQ_API_KEY` and `DEMO_MODE=off`.

**Fixtures.** The four prerecorded responses are the entries of `demo/prerecorded_responses.json`. The file is
not git-ignored and is picked up by `git add`; it only becomes tracked once you commit, because no commit
has been made in this work. Only that one file is needed by the prerecorded backend.

## Provisioning the index (kept out of requests)

The image contains no corpus and no index. Choose one:

1. **Prerecorded demo (recommended first deployment).** `DEMO_MODE=prerecorded`. No key, no index, no LLM
   spend; four real responses, clearly labelled.
2. **Sample corpus.** Build into a volume, then run live with a key:
   ```bash
   docker run --rm -v osint-chroma:/data/chroma osint-analyst-api:local \
     python -m vectorstore.build_index --corpus demo/sample_corpus.jsonl
   ```
3. **Full corpus.** Run ingestion and `build_index` on a trusted machine, then ship the Chroma directory
   (about 8,000 vectors for the author's snapshot) to the host's volume. The corpus is not in git.

## Local run (verified)

```bash
# backend, prerecorded
DEMO_MODE=prerecorded CORS_ALLOWED_ORIGINS=http://localhost:3000 uvicorn api.main:app --port 8000
# front end
cd frontend && cp .env.example .env.local && npm install && npm run dev
```

Verified in this review: backend in prerecorded and live modes; front end type-check and production build;
browser exercise of example query, guardrail rejection, backend-offline error and a 375 px mobile layout;
CORS preflight allowed for the configured origin only; the container start command and healthcheck
snippet run outside Docker.

## Docker

```bash
docker build -t osint-analyst-api:local .
docker run --rm -p 8000:8000 -e DEMO_MODE=prerecorded -e CORS_ALLOWED_ORIGINS=http://localhost:3000 osint-analyst-api:local
```

**Not verified (still).** A second attempt was not made because the host disk was at 99% with about 158 MB free and Docker's store still returned I/O errors; nothing was deleted to free space. In the first attempt every build step ran (CPU-only torch install, pinned requirements, model
download baked into the image) but the final image export failed with an I/O error from Docker Desktop's
storage while the host disk was about 98% full. So the image was never produced or run, and its size and
memory use are unknown. Re-run the two commands above on a machine with free disk.

## Remaining blockers before a public deployment

1. **Build and run the Docker image** and record its size and memory. Pick a backend host (a container
   platform with at least enough RAM for torch and the model; the requirement is unmeasured).
2. **Decide the deployment mode.** Prerecorded needs nothing more. Live needs a Groq key with a spend or rate
   cap, plus an index provisioned as above.
3. **Set `CORS_ALLOWED_ORIGINS`** to the real Vercel URL and **`NEXT_PUBLIC_API_URL`** to the backend's HTTPS URL.
4. **Rate limiting is per process.** With more than one instance, limits multiply; use a gateway or shared store.
5. **No authentication, monitoring or alerting.** Acceptable for a labelled demo; not for anything sensitive.
6. **Data terms.** Confirm the sources' terms before exposing live retrieval over a corpus, and decide how to
   present a partial snapshot honestly.
7. **Cold start.** The first request after idle can be slow (model load); consider a warm instance or a
   startup warm-up if the host scales to zero.
