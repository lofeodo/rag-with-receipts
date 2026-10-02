import pandas as pd

from rag_receipts.vertex.resolve import resolve_chunk


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


def test_resolve_chunk_real_hit_returns_matching_retrieved_chunk():
    metadata = make_metadata_df()

    chunk = resolve_chunk("Attack_range__000", metadata)

    assert chunk is not None
    assert chunk.chunk_id == "Attack_range__000"
    assert chunk.page_title == "Attack range"
    assert chunk.url == "https://oldschool.runescape.wiki/w/Attack_range"
    assert chunk.section_path == "Overview"
    assert chunk.heading_anchor == "Overview"
    assert chunk.source_type == "prose"
    assert chunk.token_count == 120
    assert chunk.text.startswith("Attack range is a measure")


def test_resolve_chunk_miss_returns_none_not_raise():
    metadata = make_metadata_df()

    chunk = resolve_chunk("Nonexistent_chunk__999", metadata)

    assert chunk is None


def test_resolve_chunk_default_scores_are_zero_placeholders():
    metadata = make_metadata_df()

    chunk = resolve_chunk("Attack_range__000", metadata)

    assert chunk.dense_score == 0.0
    assert chunk.rerank_score == 0.0
    assert chunk.dense_rank == 0
    assert chunk.final_rank == 0


def test_resolve_chunk_caller_supplied_rank_and_scores_are_used():
    metadata = make_metadata_df()

    chunk = resolve_chunk(
        "Monkey_Madness_I__003",
        metadata,
        dense_score=0.9,
        rerank_score=0.8,
        dense_rank=1,
        final_rank=2,
    )

    assert chunk.dense_score == 0.9
    assert chunk.rerank_score == 0.8
    assert chunk.dense_rank == 1
    assert chunk.final_rank == 2
