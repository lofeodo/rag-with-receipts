"""Config loading for the ingestion pipeline.

Plain dataclasses + PyYAML, matching the dependencies already declared in
pyproject.toml's `ingest` extra (no pydantic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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


@dataclass
class AppConfig:
    corpus: CorpusConfig
    ingestion: IngestionConfig


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


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    corpus = CorpusConfig(**raw["corpus"])
    ingestion = _build_ingestion_config(raw["ingestion"])
    return AppConfig(corpus=corpus, ingestion=ingestion)
