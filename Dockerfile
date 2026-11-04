# RAG With Receipts - self-contained local image.
#
# Everything the app needs is baked in: the CPU-only torch wheel, the
# embedding and reranker models (read from config.yaml), the FAISS index and
# its metadata (committed under data/index/), the demo page and the committed
# benchmark results. The only network call at query time is the Anthropic API.
#
# Run it with `docker compose up` (reads ANTHROPIC_API_KEY from .env).
# See docs/running-locally.md.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only wheel: the container has no GPU, and the default PyPI wheel on
# Linux would pull in several GB of CUDA libraries for nothing.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/

RUN pip install -e ".[index,serve]"

# Bake the embedding + reranker models, read straight from config.yaml so the
# image always matches whatever models are configured.
RUN python -c "\
import yaml; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
cfg = yaml.safe_load(open('config/config.yaml')); \
SentenceTransformer(cfg['indexing']['embedding_model']); \
CrossEncoder(cfg['retrieval']['reranker_model']); \
print('models cached')"

COPY static/ ./static/
COPY results/ ./results/
COPY data/index/ ./data/index/

# Set only after the bake step: every runtime model load uses the baked cache
# with zero Hugging Face Hub calls.
ENV HF_HUB_OFFLINE=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn rag_receipts.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
