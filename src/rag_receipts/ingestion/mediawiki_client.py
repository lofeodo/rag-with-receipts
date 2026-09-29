"""Thin client over the MediaWiki Action API used by the OSRS Wiki."""

from __future__ import annotations

import logging
import time

import httpx

logger = logging.getLogger(__name__)

_IGNORED_NAMESPACES = ("Category:", "File:", "Template:", "Special:")


class MediaWikiError(Exception):
    """Raised when a MediaWiki API call fails after exhausting retries."""


class MediaWikiClient:
    def __init__(
        self,
        api_base: str,
        user_agent: str,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self.api_base = api_base
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MediaWikiClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json"}
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.rate_limit_delay)
            try:
                response = self._client.get(self.api_base, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = MediaWikiError(
                        f"HTTP {response.status_code} for params={params}"
                    )
                    logger.warning(
                        "MediaWiki API retry %d/%d: %s",
                        attempt,
                        self.max_retries,
                        last_error,
                    )
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "MediaWiki API retry %d/%d: %s", attempt, self.max_retries, exc
                )
        raise MediaWikiError(
            f"Exhausted {self.max_retries} retries for params={params}: {last_error}"
        )

    def category_members(self, category: str) -> list[str]:
        """All page titles in a category, following cmcontinue pagination.

        Excludes members in non-content namespaces (subcategories, files, templates).
        """
        titles: list[str] = []
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
        }
        while True:
            data = self._get(params)
            members = data.get("query", {}).get("categorymembers", [])
            for member in members:
                title = member["title"]
                if not title.startswith(_IGNORED_NAMESPACES):
                    titles.append(title)
            cont = data.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            params["cmcontinue"] = cont
        return titles

    def page_categories(self, title: str) -> list[str]:
        """Category titles a page belongs to."""
        categories: list[str] = []
        params = {
            "action": "query",
            "prop": "categories",
            "titles": title,
            "cllimit": 500,
        }
        while True:
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for cat in page.get("categories", []):
                    categories.append(cat["title"])
            cont = data.get("continue", {}).get("clcontinue")
            if not cont:
                break
            params["clcontinue"] = cont
        return categories

    def parse_page(self, title: str) -> dict:
        """Rendered HTML for a page via action=parse.

        Returns a dict with "title" (canonical) and "html".
        """
        data = self._get(
            {
                "action": "parse",
                "page": title,
                "prop": "text",
            }
        )
        if "error" in data:
            raise MediaWikiError(f"parse_page({title!r}) failed: {data['error']}")
        parse = data["parse"]
        return {"title": parse["title"], "html": parse["text"]["*"]}
