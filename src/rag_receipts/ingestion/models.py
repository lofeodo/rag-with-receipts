"""Shared dataclasses for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["prose", "infobox", "table"]


@dataclass
class PageEntry:
    """One resolved page in the corpus, with provenance for reproducibility."""

    title: str
    category: str
    source_reason: str


@dataclass
class Section:
    """One node of a page's heading tree, with its own text content."""

    heading_path: list[str]
    heading_anchor: str
    source_type: SourceType
    blocks: list[str]


@dataclass
class ParsedPage:
    """A page's content flattened into an ordered list of sections."""

    title: str
    sections: list[Section]


@dataclass
class Chunk:
    """A citation-ready, size-bounded unit of text ready for embedding."""

    chunk_id: str
    page_title: str
    url: str
    section_path: str
    heading_anchor: str
    token_count: int
    source_type: SourceType
    text: str
