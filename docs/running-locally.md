# Running locally

## Docker (matches the deployed service)

```
docker build -t rag-receipts-api .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$(pwd)/data/index:/app/data/index:ro" \
  rag-receipts-api
```

Then open `http://localhost:8080/`. No `DEMO_API_KEY` is set locally, so `/query` is open. You can also hit `/livez`, `/readyz` and `/query` directly.

## Rebuilding the pipeline from source

Each step has a script under `scripts/`. They run in this order:

| Step | Command |
|---|---|
| Fetch the wiki pages | `python scripts/fetch_corpus.py --config config/config.yaml` |
| Parse and chunk | `python scripts/ingest.py --config config/config.yaml` |
| Embed and build the FAISS index | `python scripts/build_index.py --config config/config.yaml` |
| Ask a question | `python scripts/generate.py "What items are required to start Monkey Madness I?"` |
| Run the full eval | `python scripts/run_eval.py` |
| Measure latency | `python scripts/measure_latency.py` |
| Reranker sweep | `python scripts/sweep_reranker.py` |

Install notes:

- `pip install -e ".[ingest,index,eval,serve,gcp,dev]"` (pick the extras you need).
- On Windows, `pip install torch` gives a CPU-only wheel. For a GPU, install torch first from the CUDA index (`pip install torch --index-url https://download.pytorch.org/whl/cu121`), then install the project.
- `ANTHROPIC_API_KEY` is read from the environment, never from config.
- Tests: `pytest -q`. Tests marked `slow` load real models or call real APIs and are skipped by default (`pytest -m slow` to run them).

## Windows / Git Bash gotchas

- `gcloud` may need `CLOUDSDK_PYTHON` pointed at a real Python. The default `python` on PATH can be a Windows Store stub.
- Docker bind mounts from Git Bash need `MSYS_NO_PATHCONV=1` and a literal `C:\...` host path. Otherwise MSYS rewrites both the host path and container-side paths.
