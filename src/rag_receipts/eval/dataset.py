"""Loads and validates data/eval/qa_pairs.json - the hand-authored gold Q/A set.

This is the first thing that runs against the user's hand-written file, so schema
violations raise ValueError with a message specific enough to fix without guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_receipts.eval.models import EvalQuestion

_VALID_TYPES = {"single_hop", "multi_hop"}


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - write your gold Q/A pairs there first "
            f"(see EvalQuestion in rag_receipts.eval.models for the schema)."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON array of questions, got {type(raw).__name__}")

    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        _validate_entry(entry, index=i, path=path)

        qid = entry["id"]
        if qid in seen_ids:
            raise ValueError(f"{path}: duplicate question id {qid!r}")
        seen_ids.add(qid)

        answerable = entry["answerable"]
        gold_chunk_ids = entry.get("gold_chunk_ids", [])
        gold_answer = entry.get("gold_answer")

        if answerable and not gold_chunk_ids:
            raise ValueError(
                f"{path}: question {qid!r} has answerable=true but no gold_chunk_ids"
            )
        if answerable and not gold_answer:
            raise ValueError(f"{path}: question {qid!r} has answerable=true but no gold_answer")
        if not answerable and gold_chunk_ids:
            raise ValueError(
                f"{path}: question {qid!r} has answerable=false but non-empty "
                f"gold_chunk_ids {gold_chunk_ids!r}"
            )

        questions.append(
            EvalQuestion(
                id=qid,
                question=entry["question"],
                type=entry["type"],
                answerable=answerable,
                gold_chunk_ids=gold_chunk_ids,
                gold_answer=gold_answer,
            )
        )

    return questions


def _validate_entry(entry: dict, *, index: int, path: Path) -> None:
    for field_name in ("id", "question", "type", "answerable"):
        if field_name not in entry:
            raise ValueError(f"{path}: question at index {index} is missing required field {field_name!r}")

    if entry["type"] not in _VALID_TYPES:
        raise ValueError(
            f"{path}: question {entry['id']!r} has type={entry['type']!r}, "
            f"expected one of {sorted(_VALID_TYPES)}"
        )
    if not isinstance(entry["answerable"], bool):
        raise ValueError(
            f"{path}: question {entry['id']!r} has non-boolean answerable={entry['answerable']!r}"
        )
