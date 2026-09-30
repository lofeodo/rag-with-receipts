import numpy as np
import pytest

from rag_receipts.generation.models import Citation
from rag_receipts.grounding.checker import GroundingChecker, summarize_grounding
from rag_receipts.grounding.config import GroundingConfig
from rag_receipts.grounding.entailment import EntailmentChecker
from rag_receipts.retrieval.models import RetrievedChunk

from .test_entailment import FakeCrossEncoderModel

DEFAULT_LABEL_INDEX = {"entailment": 0, "contradiction": 1, "neutral": 2}


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_title="Combat triangle",
        url="https://oldschool.runescape.wiki/w/Combat_triangle",
        section_path="",
        heading_anchor="",
        source_type="prose",
        text=text,
        token_count=10,
        dense_score=0.9,
        rerank_score=0.9,
        dense_rank=1,
        final_rank=1,
    )


def make_citation(chunk_id: str, claim: str, chunk_text: str) -> Citation:
    return Citation(chunk_id=chunk_id, claim=claim, chunk=make_chunk(chunk_id, chunk_text))


def make_checker(fixed_scores: np.ndarray, config: GroundingConfig | None = None) -> GroundingChecker:
    config = config or GroundingConfig()
    fake_model = FakeCrossEncoderModel(fixed_scores)
    entailment = EntailmentChecker(config=config, model=fake_model, label_index=DEFAULT_LABEL_INDEX)
    return GroundingChecker(config=config, entailment=entailment)


def test_grounded_via_high_entailment():
    citation = make_citation("c1", "melee beats rangers", "melee is strong against rangers")
    checker = make_checker(np.array([[0.9, 0.05, 0.05]], dtype="float32"))

    verdicts = checker.check([citation])

    assert verdicts[0].label == "grounded"
    assert verdicts[0].entailment_prob == pytest.approx(0.9)


def test_grounded_via_overlap_when_entailment_is_low():
    # low entailment/neutral-ish score, but claim text is verbatim in the chunk
    citation = make_citation("c1", "the gold bar is required", "the gold bar is required for this quest")
    checker = make_checker(np.array([[0.1, 0.1, 0.8]], dtype="float32"))

    verdicts = checker.check([citation])

    assert verdicts[0].overlap_score == 1.0
    assert verdicts[0].label == "grounded"


def test_ungrounded_when_both_signals_low():
    citation = make_citation("c1", "dragons breathe fire", "goblins carry bronze swords")
    checker = make_checker(np.array([[0.1, 0.1, 0.8]], dtype="float32"))

    verdicts = checker.check([citation])

    assert verdicts[0].overlap_score == 0.0
    assert verdicts[0].label == "ungrounded"


def test_contradicted_overrides_high_overlap():
    # verbatim overlap, but the model flags contradiction - contradiction wins
    citation = make_citation("c1", "melee beats rangers", "melee is weak against rangers, not melee beats rangers")
    checker = make_checker(np.array([[0.05, 0.9, 0.05]], dtype="float32"))

    verdicts = checker.check([citation])

    assert verdicts[0].contradiction_prob == pytest.approx(0.9)
    assert verdicts[0].label == "contradicted"


def test_threshold_boundaries_are_configurable():
    citation = make_citation("c1", "claim text", "unrelated chunk text")
    config = GroundingConfig(entailment_threshold=0.3, overlap_threshold=0.9)
    checker = make_checker(np.array([[0.35, 0.1, 0.55]], dtype="float32"), config=config)

    verdicts = checker.check([citation])

    assert verdicts[0].label == "grounded"  # entailment 0.35 clears the lowered 0.3 bar


def test_check_handles_multiple_citations_in_one_batch():
    citations = [
        make_citation("c1", "melee beats rangers", "melee is strong against rangers"),
        make_citation("c2", "dragons breathe fire", "goblins carry bronze swords"),
    ]
    checker = make_checker(
        np.array([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]], dtype="float32")
    )

    verdicts = checker.check(citations)

    assert len(verdicts) == 2
    assert verdicts[0].label == "grounded"
    assert verdicts[1].label == "ungrounded"


def test_check_returns_empty_list_for_no_citations():
    checker = make_checker(np.zeros((0, 3), dtype="float32"))

    assert checker.check([]) == []


def test_summarize_grounding_all_grounded():
    citations = [
        make_citation("c1", "claim one", "claim one supported"),
        make_citation("c2", "claim two", "claim two supported"),
    ]
    checker = make_checker(np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]], dtype="float32"))

    summary = summarize_grounding(checker.check(citations))

    assert summary.all_grounded is True
    assert summary.any_contradicted is False


def test_summarize_grounding_vacuously_true_for_empty():
    summary = summarize_grounding([])

    assert summary.all_grounded is True
    assert summary.any_contradicted is False


def test_summarize_grounding_flags_contradiction():
    citation = make_citation("c1", "melee beats rangers", "melee is weak against rangers")
    checker = make_checker(np.array([[0.05, 0.9, 0.05]], dtype="float32"))

    summary = summarize_grounding(checker.check([citation]))

    assert summary.all_grounded is False
    assert summary.any_contradicted is True
