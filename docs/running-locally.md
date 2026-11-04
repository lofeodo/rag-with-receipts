# Running locally

The app runs entirely on your machine: the embedding model, reranker and FAISS index are all local. The only external call is to the Anthropic API for answer generation, so you need an Anthropic API key. Everything else ships in the repo, including the prebuilt index under `data/index/`, so there is nothing to download or rebuild first.

## 1. Add your API key

```
cp .env.example .env
```

Open `.env` and set `ANTHROPIC_API_KEY`. Leave `DEMO_API_KEY` empty; when it is set, `/query` requires a matching `X-Demo-Key` header and the demo page doesn't send one.

## 2a. Run with Docker (recommended)

```
docker compose up --build
```

Open **http://localhost:8080**. The first build takes several minutes because it downloads CPU PyTorch and bakes the two models into the image. After that, the container starts in about 30 seconds and needs no network access except to the Anthropic API.

Stop it with `Ctrl+C`, or with `docker compose down` if it runs detached (`-d`).

## 2b. Run with Python

Python 3.11 in a conda env named after the project. All commands run through `conda run -n rag-with-receipts`:

```
conda create -n rag-with-receipts python=3.11 -y
conda run -n rag-with-receipts pip install -e ".[index,serve]"
conda run -n rag-with-receipts --no-capture-output uvicorn rag_receipts.api.app:app --port 8080 --env-file .env
```

`--no-capture-output` streams the server log to your terminal; without it `conda run` holds output until the process exits.

Open **http://localhost:8080**. The first start downloads the embedding model (`bge-large`, ~1.3 GB) and the reranker from Hugging Face; later starts use the cache.

On Windows, `pip install torch` gives a CPU-only wheel. To use an NVIDIA GPU, install torch from the CUDA index first (`conda run -n rag-with-receipts pip install torch --index-url https://download.pytorch.org/whl/cu121`), then install the project.

## What's served

| Path | What it is |
|---|---|
| `/` | Demo page: ask a question, get a cited answer, plus the benchmarks panel |
| `POST /query` | `{"query": "..."}` returns the answer, citations and per-stage timing |
| `/benchmarks` | The committed benchmark results as JSON |
| `/livez`, `/readyz` | Liveness, and readiness (503 until the models and index are loaded) |

```
curl -s localhost:8080/query -H "Content-Type: application/json" \
  -d '{"query": "What items are required to start Monkey Madness I?"}'
```

`/query` is rate-limited per IP (10 requests per 5 minutes by default). Change it with `RATE_LIMIT_MAX_REQUESTS` and `RATE_LIMIT_WINDOW_SECONDS` in `.env`.

## Rebuilding the pipeline from source

Optional: the committed index is all you need to run the app. Each step has a script under `scripts/`; they run in this order:

| Step | Command |
|---|---|
| Fetch the wiki pages | `conda run -n rag-with-receipts python scripts/fetch_corpus.py --config config/config.yaml` |
| Parse and chunk | `conda run -n rag-with-receipts python scripts/ingest.py --config config/config.yaml` |
| Embed and build the FAISS index | `conda run -n rag-with-receipts python scripts/build_index.py --config config/config.yaml` |
| Ask a question from the CLI | `conda run -n rag-with-receipts python scripts/generate.py "What items are required to start Monkey Madness I?"` |
| Run the full eval | `conda run -n rag-with-receipts python scripts/run_eval.py` |
| Measure latency | `conda run -n rag-with-receipts python scripts/measure_latency.py` |
| Reranker sweep | `conda run -n rag-with-receipts python scripts/sweep_reranker.py` |

Install the extras you need: `conda run -n rag-with-receipts pip install -e ".[ingest,index,eval,serve,dev]"`. Scripts read `ANTHROPIC_API_KEY` from the environment, never from config.

Tests: `conda run -n rag-with-receipts pytest -q`. Tests marked `slow` load real models or call real APIs and are skipped by default (`conda run -n rag-with-receipts pytest -m slow` runs them).

## Windows / Git Bash gotchas

- Docker bind mounts from Git Bash need `MSYS_NO_PATHCONV=1` and a literal `C:\...` host path. Otherwise MSYS rewrites both the host path and container-side paths. (The compose setup above uses no bind mounts, so this only matters for custom `docker run` commands.)
