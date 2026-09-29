"""Parse OSRS Wiki rendered HTML into an ordered list of Sections.

Verified against live pages (Abyssal demon, Slayer training) rather than assumed:
- The content root is the outermost element carrying `mw-parser-output` (it also
  carries other classes, e.g. `mw-content-ltr mw-parser-output`).
- Headings (`h2`/`h3`/`h4`) carry their anchor directly as an `id` attribute and
  their text directly as content — there is no `span.mw-headline` wrapper (that is
  older MediaWiki skin behaviour; this wiki's parse output is `id`-on-heading).
  `mw-editsection` is a sibling span immediately after the heading, not nested in it.
- Infoboxes are real `<table class="infobox ...">` elements with `<tr><th>key</th>
  <td>value</td></tr>` data rows, interleaved with structural rows that have only a
  `th` (section subheaders, colspan across the table) or only a `td` (spacer padding)
  — those are skipped. Infobox-switch templates (e.g. monster variants) repeat
  identical key/value rows for each switch state in the static HTML; exact-duplicate
  lines are deduped.
- Plain data tables (`table.wikitable`) use a normal header row of `<th>` followed by
  `<td>` data rows.
"""

from __future__ import annotations

from bs4 import BeautifulSoup
from bs4.element import Tag

from rag_receipts.ingestion.models import ParsedPage, Section

BOILERPLATE_SELECTORS = [
    ".navbox",
    ".mw-editsection",
    "#toc",
    ".toc",
    ".references",
    ".reflist",
    ".mw-references-wrap",
    "sup.reference",
    ".noprint",
    ".catlinks",
    ".messagebox",
    ".ambox",
]

HEADING_TAGS = ("h2", "h3", "h4")
CONTENT_TAGS = ("p", "ul", "ol", "dl")


def strip_boilerplate(soup: Tag) -> None:
    for selector in BOILERPLATE_SELECTORS:
        for element in soup.select(selector):
            element.decompose()


def flatten_infobox(soup: Tag) -> list[str] | None:
    table = soup.select_one("table.infobox")
    if table is None:
        return None

    lines: list[str] = []
    for row in table.find_all("tr"):
        ths = row.find_all("th", recursive=False)
        tds = row.find_all("td", recursive=False)
        if len(ths) != 1 or len(tds) != 1:
            continue
        key = ths[0].get_text(" ", strip=True)
        value = tds[0].get_text(" ", strip=True)
        if not key or not value:
            continue
        lines.append(f"{key}: {value}")

    table.decompose()
    return list(dict.fromkeys(lines)) or None


def flatten_table(table: Tag) -> list[str]:
    rows = table.find_all("tr", recursive=False) or table.find("tbody").find_all(
        "tr", recursive=False
    )
    if not rows:
        return []

    headers: list[str] = []
    data_rows = rows
    first_cells = rows[0].find_all(["th", "td"], recursive=False)
    if first_cells and all(cell.name == "th" for cell in first_cells):
        headers = [cell.get_text(" ", strip=True) for cell in first_cells]
        data_rows = rows[1:]

    lines: list[str] = []
    for row in data_rows:
        cells = row.find_all(["th", "td"], recursive=False)
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if not any(values):
            continue
        if headers:
            parts = [
                f"{headers[i]}: {v}" if i < len(headers) else v
                for i, v in enumerate(values)
            ]
        else:
            parts = values
        lines.append(" | ".join(parts))
    return lines


class _Node:
    def __init__(self, heading_path: list[str], heading_anchor: str) -> None:
        self.heading_path = heading_path
        self.heading_anchor = heading_anchor
        self.own_blocks: list[str] = []
        self.own_chars = 0
        self.table_chars = 0
        self.children: list["_Node"] = []

    def is_leaf(self) -> bool:
        return not self.children

    def content_length(self) -> int:
        return sum(len(b) for b in self.own_blocks)


def _heading_text_and_anchor(heading: Tag) -> tuple[str, str]:
    span = heading.select_one("span.mw-headline")
    text = span.get_text(strip=True) if span else heading.get_text(strip=True)
    anchor = heading.get("id", "") or ""
    return text, anchor


def _heading_level(tag_name: str) -> int:
    return {"h2": 0, "h3": 1, "h4": 2}[tag_name]


def build_section_tree(html: str, page_title: str) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup
    strip_boilerplate(root)

    sections: list[Section] = []

    infobox_lines = flatten_infobox(root)
    if infobox_lines:
        sections.append(
            Section(
                heading_path=[],
                heading_anchor="infobox",
                source_type="infobox",
                blocks=infobox_lines,
            )
        )

    root_node = _Node(heading_path=[], heading_anchor="")
    stack: list[_Node] = [root_node]

    for element in root.find_all(HEADING_TAGS + CONTENT_TAGS + ("table",), recursive=True):
        # skip elements nested inside a table we've already captured as a content block
        if element.find_parent("table") is not None:
            continue

        if element.name in HEADING_TAGS:
            level = _heading_level(element.name)
            text, anchor = _heading_text_and_anchor(element)
            if not text:
                continue
            # parent is the nearest enclosing heading at the expected depth; if a
            # level was skipped (e.g. h2 straight to h4) fall back to the deepest
            # known ancestor rather than crashing
            parent = stack[level] if level < len(stack) else stack[-1]
            node = _Node(heading_path=parent.heading_path + [text], heading_anchor=anchor)
            parent.children.append(node)
            stack = stack[: level + 1]
            stack.append(node)
            continue

        current = stack[-1]
        if element.name == "table":
            rows = flatten_table(element)
            if rows:
                text = "\n".join(rows)
                current.own_blocks.append(text)
                current.table_chars += len(text)
            continue

        text = element.get_text(" ", strip=True)
        if text:
            current.own_blocks.append(text)

    def emit(node: _Node) -> None:
        if node.own_blocks:
            source_type = "table" if node.table_chars > node.content_length() / 2 else "prose"
            sections.append(
                Section(
                    heading_path=node.heading_path,
                    heading_anchor=node.heading_anchor,
                    source_type=source_type,
                    blocks=list(node.own_blocks),
                )
            )
        for child in node.children:
            emit(child)

    # root-level content before any heading (lead paragraph) comes first in
    # document order, ahead of any heading section
    emit(root_node)

    return ParsedPage(title=page_title, sections=sections)
