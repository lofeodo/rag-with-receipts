"""Small shared helpers for the ingestion pipeline."""

from __future__ import annotations

import re

_SLUG_INVALID = re.compile(r"[^A-Za-z0-9_]+")

WIKI_BASE_URL = "https://oldschool.runescape.wiki/w/"


def slugify(title: str) -> str:
    """Filesystem-safe, human-readable slug for a MediaWiki page title.

    "Monkey Madness I/Quick guide" -> "Monkey_Madness_I_Quick_guide"
    """
    slug = title.strip().replace(" ", "_").replace("/", "_")
    slug = _SLUG_INVALID.sub("_", slug)
    return slug.strip("_")


def page_url(title: str) -> str:
    """Citation-ready URL for a MediaWiki page title."""
    return WIKI_BASE_URL + title.strip().replace(" ", "_")


def breadcrumb(page_title: str, section_path: str) -> str:
    """Context string prepended to a chunk's text before embedding (Step 2)."""
    if section_path:
        return f"{page_title} > {section_path}"
    return page_title


def make_chunk_id(page_slug: str, index: int) -> str:
    return f"{page_slug}__{index:03d}"
