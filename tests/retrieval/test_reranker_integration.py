"""The one real-model test in the retrieval suite so far.

Loads the actual BAAI/bge-reranker-base cross-encoder and scores real query/passage
pairs end-to-end. Excluded from the default `pytest -q` run via the `slow` marker
(see pyproject.toml's `addopts`). Run explicitly with:

    pytest -q -m slow tests/retrieval/test_reranker_integration.py
"""

from __future__ import annotations

import pytest

from rag_receipts.retrieval.config import RerankerConfig
from rag_receipts.retrieval.reranker import Reranker

pytestmark = pytest.mark.slow

QUERY = "What is the Abyssal whip's special attack?"
RELEVANT_PASSAGE = (
    "Abyssal whip > Special attack\n\n"
    "The Abyssal whip does not have a special attack, but it is popular for its "
    "fast attack speed and low weight."
)
IRRELEVANT_PASSAGE = (
    "Herblore training > Overview\n\n"
    "Herblore is a members-only skill that allows players to create potions from "
    "secondary ingredients and herbs."
)


def test_real_reranker_scores_relevant_pair_higher():
    reranker = Reranker.from_config(RerankerConfig())

    scores = reranker.score(QUERY, [RELEVANT_PASSAGE, IRRELEVANT_PASSAGE])

    assert scores[0] > scores[1]
