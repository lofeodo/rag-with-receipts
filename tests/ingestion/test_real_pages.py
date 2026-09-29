import pytest

from rag_receipts.ingestion.chunker import chunk_page
from rag_receipts.ingestion.config import ChunkParams
from rag_receipts.ingestion.html_parser import build_section_tree
from rag_receipts.ingestion.tokenizer import get_tokenizer
from rag_receipts.ingestion.utils import page_url

from conftest import FIXTURES_DIR

REAL_PAGES = [
    ("Slayer_training.html", "Slayer training"),
    ("Monkey_Madness_I_Quick_guide.html", "Monkey Madness I/Quick guide"),
    ("Abyssal_demon.html", "Abyssal demon"),
    ("Abyssal_whip.html", "Abyssal whip"),
]

LEAKED_BOILERPLATE = ["[edit]", "Retrieved from", "Jump to navigation", "Jump to search"]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer("BAAI/bge-large-en-v1.5")


@pytest.fixture(scope="module")
def params():
    return ChunkParams()


@pytest.mark.parametrize("filename,title", REAL_PAGES)
def test_real_page_chunking(filename, title, tokenizer, params):
    html = (FIXTURES_DIR / "pages" / filename).read_text(encoding="utf-8")
    parsed = build_section_tree(html, title)
    chunks = chunk_page(parsed, title, page_url(title), tokenizer, params)

    assert 0 < len(chunks) < 200

    # only the infobox mini-chunk and the page's lead paragraph (before any
    # heading) are expected to have no section heading; everything else should
    unsectioned = 0
    for chunk in chunks:
        assert chunk.token_count <= params.max_tokens
        if chunk.section_path == "":
            unsectioned += 1
        for leaked in LEAKED_BOILERPLATE:
            assert leaked not in chunk.text
    assert unsectioned <= 2
