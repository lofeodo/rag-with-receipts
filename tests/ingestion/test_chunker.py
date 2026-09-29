import pytest

from rag_receipts.ingestion.chunker import chunk_page, merge_tiny, split_oversized
from rag_receipts.ingestion.config import ChunkParams
from rag_receipts.ingestion.models import ParsedPage, Section
from rag_receipts.ingestion.tokenizer import count_tokens, get_tokenizer

TOKENIZER_NAME = "BAAI/bge-large-en-v1.5"


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(TOKENIZER_NAME)


@pytest.fixture
def params():
    return ChunkParams(target_tokens=100, max_tokens=150, min_tokens=20, overlap_tokens=15)


FILLER_SENTENCE = (
    "The dragon breathes fire across the battlefield while adventurers scramble for cover "
    "and the ground trembles with each thunderous step of the beast. "
)


def _filler_paragraph(sentence_count: int) -> str:
    return FILLER_SENTENCE * sentence_count


def test_oversized_split_respects_max_tokens_and_overlaps(tokenizer, params):
    section = Section(
        heading_path=["Strategy"],
        heading_anchor="Strategy",
        source_type="prose",
        blocks=[_filler_paragraph(3) for _ in range(6)],
    )
    assert count_tokens(tokenizer, "\n\n".join(section.blocks)) > params.max_tokens

    result = split_oversized(section, params, tokenizer)

    assert len(result) > 1
    for chunk_section in result:
        text = "\n\n".join(chunk_section.blocks)
        assert count_tokens(tokenizer, text) <= params.max_tokens
        assert chunk_section.heading_path == ["Strategy"]

    # consecutive chunks share overlapping text
    for i in range(len(result) - 1):
        tail_words = "\n\n".join(result[i].blocks).split()[-5:]
        tail_snippet = " ".join(tail_words)
        assert tail_snippet in "\n\n".join(result[i + 1].blocks)


def test_tiny_sibling_merges_into_previous_sibling(tokenizer, params):
    normal = Section(
        heading_path=["Combat styles", "Melee"],
        heading_anchor="Melee",
        source_type="prose",
        blocks=[_filler_paragraph(2)],
    )
    tiny = Section(
        heading_path=["Combat styles", "Ranged"],
        heading_anchor="Ranged",
        source_type="prose",
        blocks=["Short."],
    )
    assert count_tokens(tokenizer, "Short.") < params.min_tokens

    merged = merge_tiny([normal, tiny], params.min_tokens, tokenizer)

    assert len(merged) == 1
    assert "Short." in merged[0].blocks
    # the earlier (previous) sibling's heading identity is kept
    assert merged[0].heading_path == ["Combat styles", "Melee"]


def test_tiny_leaf_merges_into_parent_intro(tokenizer, params):
    parent_intro = Section(
        heading_path=["Combat styles"],
        heading_anchor="Combat_styles",
        source_type="prose",
        blocks=[_filler_paragraph(2)],
    )
    tiny_child = Section(
        heading_path=["Combat styles", "Melee"],
        heading_anchor="Melee",
        source_type="prose",
        blocks=["Short."],
    )

    merged = merge_tiny([parent_intro, tiny_child], params.min_tokens, tokenizer)

    assert len(merged) == 1
    assert merged[0].heading_path == ["Combat styles"]
    assert "Short." in merged[0].blocks


def test_chunk_page_metadata_completeness(tokenizer, params):
    parsed = ParsedPage(
        title="Test Monster",
        sections=[
            Section(
                heading_path=[],
                heading_anchor="infobox",
                source_type="infobox",
                blocks=["Combat level: 100", "Members: Yes"],
            ),
            Section(
                heading_path=["Overview"],
                heading_anchor="Overview",
                source_type="prose",
                blocks=[_filler_paragraph(2)],
            ),
        ],
    )

    chunks = chunk_page(parsed, parsed.title, "https://oldschool.runescape.wiki/w/Test_Monster", tokenizer, params)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.page_title == "Test Monster"
        assert chunk.url == "https://oldschool.runescape.wiki/w/Test_Monster"
        assert chunk.token_count <= params.max_tokens
        assert chunk.source_type in {"prose", "infobox", "table"}

    infobox_chunk = next(c for c in chunks if c.source_type == "infobox")
    assert infobox_chunk.section_path == ""
    assert infobox_chunk.heading_anchor == "infobox"

    prose_chunk = next(c for c in chunks if c.source_type == "prose")
    assert prose_chunk.section_path == "Overview"
