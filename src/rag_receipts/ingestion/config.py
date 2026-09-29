"""Config dataclasses for the ingestion pipeline.

Plain dataclasses (no pydantic), built from a raw dict by `_build_ingestion_config`.
The top-level `AppConfig`/`load_config` that reads config.yaml live in
`rag_receipts.config`, which assembles this step's config alongside every other step's.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CorpusConfig:
    source: str
    api_base: str
    license: str
    user_agent: str


@dataclass
class ChunkParams:
    target_tokens: int = 380
    max_tokens: int = 450
    min_tokens: int = 80
    overlap_tokens: int = 60


@dataclass
class CategorySpec:
    label: str
    category: str | None = None
    pages: list[str] = field(default_factory=list)


@dataclass
class QuestlineSpec:
    label: str
    pages: list[str]


@dataclass
class LinkFollowSpec:
    depth: int = 1
    target_categories: list[str] = field(
        default_factory=lambda: ["Category:Items", "Category:Monsters"]
    )


@dataclass
class IngestionConfig:
    categories: list[CategorySpec]
    questline: QuestlineSpec
    link_follow: LinkFollowSpec
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    results_dir: str = "results"
    tokenizer_name: str = "BAAI/bge-large-en-v1.5"
    chunk: ChunkParams = field(default_factory=ChunkParams)
    rate_limit_delay_seconds: float = 0.5
    max_retries: int = 3


def _build_ingestion_config(raw: dict) -> IngestionConfig:
    categories = [CategorySpec(**c) for c in raw["categories"]]
    questline = QuestlineSpec(**raw["questline"])
    link_follow = LinkFollowSpec(**raw["link_follow"])

    output = raw.get("output", {})
    chunking = raw.get("chunking", {})
    fetch = raw.get("fetch", {})

    chunk_params = ChunkParams(
        target_tokens=chunking.get("target_tokens", 380),
        max_tokens=chunking.get("max_tokens", 450),
        min_tokens=chunking.get("min_tokens", 80),
        overlap_tokens=chunking.get("overlap_tokens", 60),
    )

    return IngestionConfig(
        categories=categories,
        questline=questline,
        link_follow=link_follow,
        raw_dir=output.get("raw_dir", "data/raw"),
        processed_dir=output.get("processed_dir", "data/processed"),
        results_dir=output.get("results_dir", "results"),
        tokenizer_name=chunking.get("tokenizer", "BAAI/bge-large-en-v1.5"),
        chunk=chunk_params,
        rate_limit_delay_seconds=fetch.get("rate_limit_delay_seconds", 0.5),
        max_retries=fetch.get("max_retries", 3),
    )
