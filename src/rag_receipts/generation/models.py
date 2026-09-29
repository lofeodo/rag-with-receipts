"""Result shape for the generation pipeline (Step 4)."""

from __future__ import annotations

from dataclasses import dataclass

from rag_receipts.retrieval.models import RetrievedChunk


@dataclass
class Citation:
    """A validated claim-to-chunk citation. Only citations whose chunk_id was
    actually present in the retrieved set are represented here - see
    GeneratedAnswer.hallucinated_citation_ids for the rest."""

    chunk_id: str
    claim: str
    chunk: RetrievedChunk  # resolved, not a bare id - callers render page_title/url/
    # heading_anchor without a second lookup


@dataclass
class GeneratedAnswer:
    query: str
    answer: str
    citations: list[Citation]
    answerable: bool
    hallucinated_citation_ids: list[str]  # chunk_ids the model cited that were NOT
    # in the retrieved set - not silently trusted
    has_hallucinated_citations: bool
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
