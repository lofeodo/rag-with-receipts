"""Real Anthropic API call against claude-sonnet-5 (Step 4).

Costs real API dollars per run, unlike the other `slow` tests which just load a
real local ML model - the ANTHROPIC_API_KEY skipif is the actual gate that keeps
this from running (and costing money) unintentionally.

Run explicitly:
    pytest -q -m slow tests/generation/test_generator_integration.py
"""

from __future__ import annotations

import os

import pytest

from rag_receipts.generation.config import GenerationConfig
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.models import RetrievedChunk

pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set - skipping real Anthropic API call",
)
def test_generate_real_api_call_cites_a_real_chunk():
    chunks = [
        RetrievedChunk(
            chunk_id="Abyssal_whip__special_attack",
            page_title="Abyssal whip",
            url="https://oldschool.runescape.wiki/w/Abyssal_whip",
            section_path="Special attack",
            heading_anchor="Special_attack",
            source_type="prose",
            text=(
                "The abyssal whip does not have a special attack. It is a "
                "one-handed slashing weapon requiring 70 Attack to wield."
            ),
            token_count=40,
            dense_score=0.9,
            rerank_score=0.95,
            dense_rank=1,
            final_rank=1,
        ),
        RetrievedChunk(
            chunk_id="Whip_attack_styles",
            page_title="Abyssal whip",
            url="https://oldschool.runescape.wiki/w/Abyssal_whip",
            section_path="Combat styles",
            heading_anchor="Combat_styles",
            source_type="table",
            text="Attack styles: Flick (Slash, Attack XP), Lash (Slash, Shared XP), Deflect (Slash, Defence XP).",
            token_count=30,
            dense_score=0.8,
            rerank_score=0.7,
            dense_rank=2,
            final_rank=2,
        ),
    ]

    generator = Generator.from_config(GenerationConfig())
    result = generator.generate("What attack styles does the Abyssal whip have?", chunks)

    assert result.answerable is True
    assert result.citations, "expected at least one citation"
    cited_ids = {c.chunk_id for c in result.citations}
    assert cited_ids <= {chunk.chunk_id for chunk in chunks}
    assert result.hallucinated_citation_ids == []
    assert result.input_tokens > 0
    assert result.output_tokens > 0
