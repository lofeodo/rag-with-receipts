from __future__ import annotations

import numpy as np
import pandas as pd


class FakeEncoder:
    """Deterministic fake standing in for SentenceTransformer.encode in unit tests.

    Produces a fixed-dim vector per string derived from its hash, so different texts
    get different (but reproducible) vectors, and records what it was called with.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, sentences, *, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append(list(sentences))
        vecs = np.array(
            [[float((hash(s) >> (i * 4)) % 97) for i in range(self.dim)] for s in sentences],
            dtype="float32",
        )
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs


def make_chunks_df(n: int = 5) -> pd.DataFrame:
    rows = [
        {
            "chunk_id": f"Test_page__{i:03d}",
            "page_title": "Test page",
            "url": "https://oldschool.runescape.wiki/w/Test_page",
            "section_path": "Section A" if i % 2 == 0 else "",
            "heading_anchor": "Section_A" if i % 2 == 0 else "",
            "token_count": 50 + i,
            "source_type": "prose",
            "text": f"This is synthetic chunk number {i}.",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)
