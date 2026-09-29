"""Idempotent, cached page fetching via the MediaWiki API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from rag_receipts.ingestion.mediawiki_client import MediaWikiClient, MediaWikiError
from rag_receipts.ingestion.models import PageEntry
from rag_receipts.ingestion.utils import slugify

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    title: str
    status: Literal["cached", "fetched", "failed"]
    error: str | None = None


def _page_html_path(raw_dir: Path, title: str) -> Path:
    return raw_dir / "pages" / f"{slugify(title)}.html"


def fetch_and_cache(
    pages: list[PageEntry], client: MediaWikiClient, raw_dir: Path, show_progress: bool = True
) -> list[FetchResult]:
    """Fetch each page's rendered HTML, skipping pages already cached on disk."""
    pages_dir = raw_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    results: list[FetchResult] = []
    iterator = tqdm(pages, desc="fetching pages", disable=not show_progress)
    for page in iterator:
        dest = _page_html_path(raw_dir, page.title)
        if dest.exists():
            results.append(FetchResult(title=page.title, status="cached"))
            continue
        try:
            parsed = client.parse_page(page.title)
            dest.write_text(parsed["html"], encoding="utf-8")
            results.append(FetchResult(title=page.title, status="fetched"))
        except MediaWikiError as exc:
            logger.warning("Failed to fetch %r: %s", page.title, exc)
            results.append(FetchResult(title=page.title, status="failed", error=str(exc)))
    return results


def load_cached_html(raw_dir: Path, title: str) -> str | None:
    path = _page_html_path(raw_dir, title)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
