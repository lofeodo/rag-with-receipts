"""Header-aware chunking: tiny-section merge + oversized-section split with overlap."""

from __future__ import annotations

import logging
import re

from rag_receipts.ingestion.config import ChunkParams
from rag_receipts.ingestion.models import Chunk, ParsedPage, Section
from rag_receipts.ingestion.tokenizer import count_tokens
from rag_receipts.ingestion.utils import make_chunk_id, slugify

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _section_text(section: Section) -> str:
    return "\n\n".join(section.blocks)


def _section_tokens(section: Section, tokenizer) -> int:
    return count_tokens(tokenizer, _section_text(section))


def _parent_path(section: Section) -> list[str]:
    return section.heading_path[:-1]


def _merge_into(target: Section, extra: Section) -> Section:
    """Merge `extra`'s content into `target`, keeping `target`'s heading identity."""
    return Section(
        heading_path=target.heading_path,
        heading_anchor=target.heading_anchor,
        source_type=target.source_type if target.blocks else extra.source_type,
        blocks=list(target.blocks) + list(extra.blocks),
    )


def merge_tiny(sections: list[Section], min_tokens: int, tokenizer) -> list[Section]:
    """Merge sections below `min_tokens` into a sibling or the parent's intro.

    Precedence: previous sibling/parent-intro (backward pass) -> next section
    (forward pass) -> root-level orphan, kept as-is with a warning.
    """
    # Infobox mini-chunks are a deliberate, always-standalone citation unit
    # (spec: "dedicated mini-chunk per page") regardless of size; never merge
    # them away or merge other content into them.
    def is_mergeable(s: Section) -> bool:
        return s.source_type != "infobox" and _section_tokens(s, tokenizer) < min_tokens

    # Pass 1: backward merge into the previous compatible section (same parent,
    # or the immediately preceding section being this one's own parent-intro).
    backward: list[Section] = []
    for section in sections:
        if backward and is_mergeable(section) and backward[-1].source_type != "infobox":
            last = backward[-1]
            compatible = _parent_path(section) == _parent_path(last) or last.heading_path == _parent_path(
                section
            )
            if compatible:
                backward[-1] = _merge_into(last, section)
                continue
        backward.append(section)

    # Pass 2: forward merge any still-tiny section into the next one (handles a
    # tiny section that opens a new subtree with no compatible predecessor).
    forward: list[Section] = []
    i = 0
    while i < len(backward):
        section = backward[i]
        is_last = i == len(backward) - 1
        if not is_last and is_mergeable(section) and backward[i + 1].source_type != "infobox":
            next_section = backward[i + 1]
            # keep the deeper heading identity (the next section is usually the
            # more specific one being introduced) unless this section is the
            # shallower parent-intro, in which case its own identity is kept
            if len(next_section.heading_path) >= len(section.heading_path):
                merged = Section(
                    heading_path=next_section.heading_path,
                    heading_anchor=next_section.heading_anchor,
                    source_type=next_section.source_type,
                    blocks=list(section.blocks) + list(next_section.blocks),
                )
            else:
                merged = _merge_into(section, next_section)
            forward.append(merged)
            i += 2
            continue
        if is_last and is_mergeable(section) and forward and forward[-1].source_type != "infobox":
            forward[-1] = _merge_into(forward[-1], section)
            i += 1
            continue
        if is_last and is_mergeable(section):
            logger.warning(
                "Orphan tiny section kept as-is (no compatible merge target): %s",
                " > ".join(section.heading_path) or "(root)",
            )
        forward.append(section)
        i += 1

    return forward


def _tail_by_tokens(tokenizer, text: str, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= n_tokens:
        return text
    return tokenizer.decode(ids[-n_tokens:])


def _hard_split_by_tokens(tokenizer, text: str, max_tokens: int) -> list[str]:
    """Last-resort token-window split, guaranteeing the max_tokens invariant."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [
        tokenizer.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)
    ]


def split_oversized(section: Section, params: ChunkParams, tokenizer) -> list[Section]:
    """Paragraph-boundary greedy packing under max_tokens, with trailing overlap."""
    expanded: list[str] = []
    for block in section.blocks:
        if count_tokens(tokenizer, block) > params.max_tokens:
            sentences = [s for s in _SENTENCE_SPLIT_RE.split(block) if s.strip()]
            expanded.extend(sentences if sentences else [block])
        else:
            expanded.append(block)

    # guarantee every individual part fits before packing: a sentence-split
    # part (or a block with no sentence punctuation, e.g. a dense table row)
    # can still exceed max_tokens on its own
    guaranteed: list[str] = []
    for part in expanded:
        if count_tokens(tokenizer, part) > params.max_tokens:
            guaranteed.extend(_hard_split_by_tokens(tokenizer, part, params.max_tokens))
        else:
            guaranteed.append(part)
    expanded = guaranteed

    packed: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for part in expanded:
        part_tokens = count_tokens(tokenizer, part)
        if current and current_tokens + part_tokens > params.max_tokens:
            packed.append(current)
            current, current_tokens = [], 0
        current.append(part)
        current_tokens += part_tokens
    if current:
        packed.append(current)

    result: list[Section] = []
    prev_text: str | None = None
    for parts in packed:
        base_text = "\n\n".join(parts)
        text = base_text
        if prev_text is not None and params.overlap_tokens > 0:
            # verify-and-shrink rather than a precomputed budget: the "\n\n"
            # separator and the decode round-trip in _tail_by_tokens can each
            # add a token or two beyond what a naive budget subtraction
            # predicts, so shrink until the assembled text actually fits
            overlap_len = params.overlap_tokens
            while overlap_len > 0:
                overlap = _tail_by_tokens(tokenizer, prev_text, overlap_len)
                candidate = f"{overlap}\n\n{base_text}" if overlap else base_text
                if count_tokens(tokenizer, candidate) <= params.max_tokens:
                    text = candidate
                    break
                overlap_len -= 5
            else:
                text = base_text
        result.append(
            Section(
                heading_path=section.heading_path,
                heading_anchor=section.heading_anchor,
                source_type=section.source_type,
                blocks=[text],
            )
        )
        prev_text = "\n\n".join(parts)
    return result


def chunk_page(
    parsed: ParsedPage,
    page_title: str,
    url: str,
    tokenizer,
    params: ChunkParams,
) -> list[Chunk]:
    page_slug = slugify(page_title)
    merged = merge_tiny(parsed.sections, params.min_tokens, tokenizer)

    final_sections: list[Section] = []
    for section in merged:
        if _section_tokens(section, tokenizer) > params.max_tokens:
            final_sections.extend(split_oversized(section, params, tokenizer))
        else:
            final_sections.append(section)

    chunks: list[Chunk] = []
    for index, section in enumerate(final_sections):
        text = _section_text(section)
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(page_slug, index),
                page_title=page_title,
                url=url,
                section_path=" > ".join(section.heading_path),
                heading_anchor=section.heading_anchor,
                token_count=count_tokens(tokenizer, text),
                source_type=section.source_type,
                text=text,
            )
        )
    return chunks
