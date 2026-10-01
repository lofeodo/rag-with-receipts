"""Request/response schemas for the HTTP API (Step 8).

Deliberately a thin reshaping of the existing GeneratedAnswer/RetrievalTiming
dataclasses (Steps 4/7) into Pydantic models for FastAPI's response_model
validation + OpenAPI docs - no new domain logic lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class CitationOut(BaseModel):
    chunk_id: str
    claim: str
    page_title: str
    url: str
    section_path: str
    heading_anchor: str


class TimingOut(BaseModel):
    embed_query_s: float
    dense_search_s: float
    rerank_s: float
    generate_s: float
    total_s: float


class QueryResponse(BaseModel):
    query: str
    answerable: bool
    answer: str
    citations: list[CitationOut]
    hallucinated_citation_ids: list[str]
    timing: TimingOut
