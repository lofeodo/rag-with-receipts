from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_receipts.eval.dataset import load_eval_questions

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "qa_pairs_small.json"


def test_loads_valid_fixture_set():
    questions = load_eval_questions(FIXTURE_PATH)

    assert len(questions) == 5
    ids = [q.id for q in questions]
    assert ids == ["q001", "q002", "q003", "q004", "q005"]

    q001 = questions[0]
    assert q001.question == "What items are required to start Monkey Madness I?"
    assert q001.type == "single_hop"
    assert q001.answerable is True
    assert q001.gold_chunk_ids == ["Monkey_Madness_I__002"]
    assert q001.gold_answer

    q003 = questions[2]
    assert q003.type == "multi_hop"
    assert len(q003.gold_chunk_ids) == 2

    q004 = questions[3]
    assert q004.answerable is False
    assert q004.type is None
    assert q004.gold_chunk_ids == []
    assert q004.gold_answer is None


def test_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_questions(tmp_path / "does_not_exist.json")


def test_non_list_json_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    with pytest.raises(ValueError, match="expected a JSON array"):
        load_eval_questions(path)


def test_missing_required_field_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps([{"id": "q1", "question": "x", "type": "single_hop"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field"):
        load_eval_questions(path)


def test_invalid_type_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": "triple_hop",
                    "answerable": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="type=" ):
        load_eval_questions(path)


def test_duplicate_id_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    entry = {
        "id": "q1",
        "question": "x",
        "type": "single_hop",
        "answerable": False,
        "gold_chunk_ids": [],
    }
    path.write_text(json.dumps([entry, dict(entry)]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate question id"):
        load_eval_questions(path)


def test_answerable_true_without_gold_chunk_ids_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": "single_hop",
                    "answerable": True,
                    "gold_chunk_ids": [],
                    "gold_answer": "some answer",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no gold_chunk_ids"):
        load_eval_questions(path)


def test_answerable_true_without_gold_answer_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": "single_hop",
                    "answerable": True,
                    "gold_chunk_ids": ["Some_page__000"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no gold_answer"):
        load_eval_questions(path)


def test_type_null_allowed_when_answerable_false(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": None,
                    "answerable": False,
                    "gold_chunk_ids": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    questions = load_eval_questions(path)

    assert questions[0].type is None


def test_type_null_rejected_when_answerable_true(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": None,
                    "answerable": True,
                    "gold_chunk_ids": ["Some_page__000"],
                    "gold_answer": "some answer",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="type=None"):
        load_eval_questions(path)


def test_answerable_false_with_gold_chunk_ids_raises_value_error(tmp_path):
    path = tmp_path / "qa_pairs.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "x",
                    "type": "single_hop",
                    "answerable": False,
                    "gold_chunk_ids": ["Some_page__000"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty"):
        load_eval_questions(path)
