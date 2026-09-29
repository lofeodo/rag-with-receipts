"""The one real-model test in the indexing suite.

Loads the actual bge-large-en-v1.5 sentence-transformers model (~1.3GB) and encodes a
handful of real strings end-to-end. Excluded from the default `pytest -q` run via the
`slow` marker (see pyproject.toml's `addopts`). Run explicitly with:

    pytest -q -m slow tests/indexing/test_embedder_integration.py
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_receipts.indexing.config import EmbeddingConfig
from rag_receipts.indexing.embedder import Embedder

pytestmark = pytest.mark.slow

PASSAGES = [
    "The Abyssal whip is a members-only one-handed slashing weapon.",
    "Slayer training involves killing monsters assigned by a Slayer master.",
    "Herblore is a members-only skill that allows players to create potions.",
    "Monkey Madness I is a quest that grants access to Zooknock and the Gorilla Greegree.",
    "Abyssal demons are members-only monsters that can teleport around players.",
]
QUERY = "What weapon has a special attack that hits twice?"


def test_real_model_encodes_passages_and_query():
    embedder = Embedder.from_config(EmbeddingConfig())

    passage_embeddings = embedder.encode_passages(PASSAGES)

    assert passage_embeddings.shape == (len(PASSAGES), 1024)
    norms = np.linalg.norm(passage_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

    from rag_receipts.indexing.faiss_index import build_flat_ip_index

    index = build_flat_ip_index(passage_embeddings)
    query_embedding = embedder.encode_queries([QUERY])
    scores, ids = index.search(query_embedding, 1)

    assert scores[0][0] > 0.3
