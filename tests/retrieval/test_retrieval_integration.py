"""Real-corpus retrieval regression check.

Loads the actual data/index/faiss.index + data/index/index_metadata.parquet (built by
scripts/build_index.py in Step 2) and runs a real query end-to-end through Retriever.
Skipped when those artifacts aren't present (they're gitignored, regenerated locally),
so this costs nothing on a fresh clone / CI but catches real regressions when they are.

Run explicitly with:
    pytest -q -m slow tests/retrieval/test_retrieval_integration.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_receipts.config import load_config
from rag_receipts.retrieval.pipeline import Retriever

pytestmark = pytest.mark.slow

INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/index_metadata.parquet")


@pytest.mark.skipif(
    not (INDEX_PATH.exists() and METADATA_PATH.exists()),
    reason="real index/metadata not present - run scripts/build_index.py first",
)
def test_real_corpus_monkey_madness_query():
    cfg = load_config("config/config.yaml")
    retriever = Retriever.from_config(cfg)

    results = retriever.retrieve("What items do I need to start Monkey Madness I?")

    assert results
    assert any("Monkey Madness" in r.page_title for r in results[:3])
