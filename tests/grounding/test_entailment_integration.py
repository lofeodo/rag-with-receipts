"""The one real-model test in the grounding suite. Loads the actual
cross-encoder/nli-deberta-v3-base checkpoint and confirms it separates entailed,
contradicted, and neutral/unrelated (premise, hypothesis) pairs correctly, and that
EntailmentChecker.from_config()'s label-name remapping (see entailment.py's docstring)
produces the expected fixed (entailment, contradiction, neutral) column order end-to-end
against the real model's native id2label.

Excluded from the default `pytest -q` run via the `slow` marker. Run explicitly with:

    pytest -q -m slow tests/grounding/test_entailment_integration.py
"""

from __future__ import annotations

import pytest

from rag_receipts.grounding.config import GroundingConfig
from rag_receipts.grounding.entailment import EntailmentChecker

pytestmark = pytest.mark.slow

ENTAILED_PAIR = (
    "Old School RuneScape follows the combat triangle: melee is strong against "
    "rangers, ranged is strong against mages, and magic is strong against warriors.",
    "Melee is effective against rangers.",
)
CONTRADICTED_PAIR = (
    "Old School RuneScape follows the combat triangle: melee is strong against "
    "rangers, ranged is strong against mages, and magic is strong against warriors.",
    "Melee is weak against rangers.",
)
NEUTRAL_PAIR = (
    "Old School RuneScape follows the combat triangle: melee is strong against "
    "rangers, ranged is strong against mages, and magic is strong against warriors.",
    "The Abyssal whip is a popular melee weapon with a fast attack speed.",
)


def test_real_entailment_model_separates_labels():
    checker = EntailmentChecker.from_config(GroundingConfig())

    scores = checker.score([ENTAILED_PAIR, CONTRADICTED_PAIR, NEUTRAL_PAIR])

    entailment_probs = scores[:, 0]
    contradiction_probs = scores[:, 1]
    neutral_probs = scores[:, 2]

    assert entailment_probs[0] == max(scores[0])
    assert contradiction_probs[1] == max(scores[1])
    assert neutral_probs[2] == max(scores[2])
