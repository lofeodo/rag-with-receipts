"""Orchestration for the fetch and ingest CLI scripts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_receipts.ingestion.chunker import chunk_page
from rag_receipts.ingestion.config import AppConfig
from rag_receipts.ingestion.fetch import FetchResult, fetch_and_cache, load_cached_html
from rag_receipts.ingestion.html_parser import build_section_tree
from rag_receipts.ingestion.mediawiki_client import MediaWikiClient
from rag_receipts.ingestion.models import Chunk, PageEntry
from rag_receipts.ingestion.page_list import resolve_page_list
from rag_receipts.ingestion.tokenizer import get_tokenizer
from rag_receipts.ingestion.utils import page_url

logger = logging.getLogger(__name__)


def run_fetch(config: AppConfig, force_refresh: bool = False) -> list[FetchResult]:
    client = MediaWikiClient(
        config.corpus.api_base,
        config.corpus.user_agent,
        rate_limit_delay=config.ingestion.rate_limit_delay_seconds,
        max_retries=config.ingestion.max_retries,
    )
    try:
        pages = resolve_page_list(client, config.ingestion, force_refresh=force_refresh)
        # seeds are typically already cached from resolution; this second pass
        # fetches any newly-discovered linked pages
        return fetch_and_cache(pages, client, Path(config.ingestion.raw_dir))
    finally:
        client.close()


def _bucket_label(token_count: int) -> str:
    lower = (token_count // 100) * 100
    return f"{lower}-{lower + 100}"


def run_ingest(config: AppConfig) -> tuple[list[Chunk], dict]:
    raw_dir = Path(config.ingestion.raw_dir)
    list_path = raw_dir / "page_list.json"
    if not list_path.exists():
        raise FileNotFoundError(
            f"{list_path} not found - run scripts/fetch_corpus.py first"
        )

    pages = [
        PageEntry(**entry) for entry in json.loads(list_path.read_text(encoding="utf-8"))
    ]

    tokenizer = get_tokenizer(config.ingestion.tokenizer_name)

    chunks: list[Chunk] = []
    per_category: dict[str, dict[str, int]] = {}
    histogram: dict[str, int] = {}
    parse_failures: list[dict[str, str]] = []
    pages_parsed = 0

    for page in pages:
        html = load_cached_html(raw_dir, page.title)
        if html is None:
            parse_failures.append({"title": page.title, "error": "missing cached HTML"})
            continue
        try:
            parsed = build_section_tree(html, page.title)
            page_chunks = chunk_page(
                parsed, page.title, page_url(page.title), tokenizer, config.ingestion.chunk
            )
        except Exception as exc:  # noqa: BLE001 - keep going on any single bad page
            logger.warning("Failed to parse/chunk %r: %s", page.title, exc)
            parse_failures.append({"title": page.title, "error": str(exc)})
            continue

        pages_parsed += 1
        chunks.extend(page_chunks)

        bucket = per_category.setdefault(page.category, {"pages": 0, "chunks": 0})
        bucket["pages"] += 1
        bucket["chunks"] += len(page_chunks)

        for chunk in page_chunks:
            label = _bucket_label(chunk.token_count)
            histogram[label] = histogram.get(label, 0) + 1

    stats = {
        "page_count": pages_parsed,
        "chunk_count": len(chunks),
        "per_category": per_category,
        "chunk_size_histogram": dict(sorted(histogram.items())),
        "parse_failures": parse_failures,
    }
    return chunks, stats
