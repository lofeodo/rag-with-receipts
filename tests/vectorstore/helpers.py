from __future__ import annotations

import pandas as pd


def make_chunks_df(n: int = 5) -> pd.DataFrame:
    """Duplicated from tests/retrieval/helpers.py rather than cross-imported -
    this repo's no-__init__.py-across-package pytest layout keeps each test
    package's imports self-contained."""
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
