"""Reproducible page-list resolution: categories + questline + one-hop link-following."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

from rag_receipts.ingestion.config import IngestionConfig
from rag_receipts.ingestion.fetch import fetch_and_cache, load_cached_html
from rag_receipts.ingestion.mediawiki_client import MediaWikiClient
from rag_receipts.ingestion.models import PageEntry

logger = logging.getLogger(__name__)

_IGNORED_LINK_PREFIXES = ("File:", "Category:", "Special:", "Template:", "Help:")
_INTERNAL_LINK_RE = re.compile(r"^/w/(.+)$")


def resolve_seed_pages(client: MediaWikiClient, config: IngestionConfig) -> list[PageEntry]:
    """Category pulls (or manual page overrides) + the questline page list."""
    seeds: list[PageEntry] = []

    for spec in config.categories:
        if spec.pages:
            for title in spec.pages:
                seeds.append(
                    PageEntry(title=title, category=spec.label, source_reason="manual")
                )
        elif spec.category:
            titles = client.category_members(spec.category)
            for title in titles:
                seeds.append(
                    PageEntry(
                        title=title,
                        category=spec.label,
                        source_reason=f"category:{spec.category}",
                    )
                )

    for title in config.questline.pages:
        seeds.append(
            PageEntry(
                title=title,
                category=config.questline.label,
                source_reason=f"questline:{config.questline.label}",
            )
        )

    return seeds


def extract_internal_links(html: str) -> list[str]:
    """Article-namespace page titles linked from a page's rendered content body."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup

    titles: set[str] = set()
    for anchor in root.select('a[href^="/w/"]'):
        href = anchor.get("href", "")
        match = _INTERNAL_LINK_RE.match(href)
        if not match:
            continue
        target = match.group(1).split("#", 1)[0]
        if not target:
            continue
        title = unquote(target).replace("_", " ")
        if title.startswith(_IGNORED_LINK_PREFIXES):
            continue
        titles.add(title)
    return sorted(titles)


def filter_by_category(
    client: MediaWikiClient, candidate_titles: list[str], target_categories: list[str]
) -> list[PageEntry]:
    """Keep only candidates belonging to one of the target categories."""
    target_set = set(target_categories)
    linked: list[PageEntry] = []
    for title in candidate_titles:
        categories = set(client.page_categories(title))
        matched = categories & target_set
        if not matched:
            continue
        label = "linked_monster" if "Category:Monsters" in matched else "linked_item"
        linked.append(
            PageEntry(
                title=title,
                category=label,
                source_reason=f"linked, category:{sorted(matched)[0]}",
            )
        )
    return linked


def _dedupe_preserve_first(pages: list[PageEntry]) -> list[PageEntry]:
    seen: set[str] = set()
    result: list[PageEntry] = []
    for page in pages:
        if page.title in seen:
            continue
        seen.add(page.title)
        result.append(page)
    return result


def _write_page_list(pages: list[PageEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"title": p.title, "category": p.category, "source_reason": p.source_reason}
        for p in pages
    ]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_page_list(path: Path) -> list[PageEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [PageEntry(**entry) for entry in data]


def resolve_page_list(
    client: MediaWikiClient, config: IngestionConfig, force_refresh: bool = False
) -> list[PageEntry]:
    """Resolve the full corpus page list, caching the result to page_list.json.

    Link-following requires the rendered HTML of seed pages, so seed pages are
    fetched here as a side effect before their links can be extracted.
    """
    raw_dir = Path(config.raw_dir)
    list_path = raw_dir / "page_list.json"
    if list_path.exists() and not force_refresh:
        return _load_page_list(list_path)

    seeds = resolve_seed_pages(client, config)
    fetch_and_cache(seeds, client, raw_dir)

    seed_titles = {p.title for p in seeds}
    candidates: set[str] = set()
    for seed in seeds:
        html = load_cached_html(raw_dir, seed.title)
        if html is None:
            continue
        candidates.update(extract_internal_links(html))
    candidates -= seed_titles

    linked = filter_by_category(client, sorted(candidates), config.link_follow.target_categories)

    all_pages = _dedupe_preserve_first(seeds + linked)
    _write_page_list(all_pages, list_path)
    return all_pages
