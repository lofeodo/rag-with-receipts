"""Duplicated fakes for Vertex client Protocols - matches the project's
established convention (duplicated fakes rather than cross-package imports,
see tests/generation/helpers.py)."""

from __future__ import annotations

import pandas as pd


def make_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "Attack_range__000",
                "page_title": "Attack range",
                "url": "https://oldschool.runescape.wiki/w/Attack_range",
                "section_path": "Overview",
                "heading_anchor": "Overview",
                "token_count": 120,
                "source_type": "prose",
                "text": "Attack range is a measure of how far away...",
            },
            {
                "chunk_id": "Monkey_Madness_I__003",
                "page_title": "Monkey Madness I",
                "url": "https://oldschool.runescape.wiki/w/Monkey_Madness_I",
                "section_path": "Requirements",
                "heading_anchor": "Requirements",
                "token_count": 90,
                "source_type": "infobox",
                "text": "Items required: A gold bar, Five empty inventory slots",
            },
        ]
    )


class FakeDocument:
    def __init__(self, id: str, struct_data: dict):
        self.id = id
        self.struct_data = struct_data


class FakeSearchResult:
    def __init__(self, document: FakeDocument):
        self.document = document


class FakeSearchServiceClient:
    """Satisfies VertexSearchClientLike. Records every request it receives."""

    def __init__(self, results: list[FakeSearchResult]):
        self._results = results
        self.requests: list[object] = []

    def search(self, request):
        self.requests.append(request)
        return self._results


def make_search_result(chunk_id: str, *, document_id: str | None = None) -> FakeSearchResult:
    """document_id defaults to chunk_id (the normal case); pass a different
    value to exercise the id/structData mismatch path."""
    return FakeSearchResult(
        FakeDocument(id=document_id if document_id is not None else chunk_id, struct_data={"chunk_id": chunk_id})
    )
