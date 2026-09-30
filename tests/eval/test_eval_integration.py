"""Real end-to-end eval run: real index, real retrieval, real Anthropic API calls for
both generation and judging. Costs real API dollars per run, and needs the real FAISS
index built - the dual skipif gate is the actual guard against running (and costing
money/failing) unintentionally on a fresh clone or CI.

Run explicitly:
    pytest -q -m slow tests/eval/test_eval_integration.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.eval.judge import Judge
from rag_receipts.eval.runner import run_eval, summarize
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.pipeline import Retriever

pytestmark = pytest.mark.slow

INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/index_metadata.parquet")
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qa_pairs_small.json"


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping real Anthropic API calls",
)
@pytest.mark.skipif(
    not (INDEX_PATH.exists() and METADATA_PATH.exists()),
    reason="real index/metadata not present - run scripts/build_index.py first",
)
def test_eval_runs_end_to_end_against_real_pipeline():
    cfg = load_config("config/config.yaml")
    questions = load_eval_questions(FIXTURE_PATH)

    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)
    judge = Judge.from_config(cfg.eval)

    results = run_eval(questions, retriever=retriever, generator=generator, judge=judge)
    summary = summarize(results)

    assert summary.question_count == len(questions)
    assert summary.error_count == 0
    # At least one of the fixture's answerable questions should resolve to a real,
    # non-error correctness label - confirms the judge call path works end-to-end.
    assert any(r.correctness_label in {"correct", "partially_correct"} for r in results)
