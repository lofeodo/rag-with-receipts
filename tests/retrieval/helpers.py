from __future__ import annotations

import numpy as np
import pandas as pd


class FakeCrossEncoder:
    """Deterministic fake standing in for CrossEncoder.predict in unit tests.

    Score = whitespace-token overlap count (case-insensitive) between query and
    passage - a real, checkable relevance signal (not a hash), so sort-order
    assertions in test_pipeline.py are meaningful.
    """

    def __init__(self):
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences, *, batch_size, show_progress_bar):
        self.calls.append(list(sentences))
        return np.array(
            [
                float(len(set(query.lower().split()) & set(passage.lower().split())))
                for query, passage in sentences
            ],
            dtype="float32",
        )


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
