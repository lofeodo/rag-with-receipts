import numpy as np

from rag_receipts.grounding.config import GroundingConfig
from rag_receipts.grounding.entailment import EntailmentChecker, resolve_label_index


class FakeCrossEncoderModel:
    """Structural fake for entailment.CrossEncoderModel - returns a fixed (n, 3) array
    in whatever native column order the test wants to simulate."""

    def __init__(self, fixed_output: np.ndarray):
        self.fixed_output = fixed_output
        self.last_call_kwargs: dict = {}

    def predict(self, sentences, *, batch_size, show_progress_bar, apply_softmax, convert_to_numpy):
        self.last_call_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "apply_softmax": apply_softmax,
            "convert_to_numpy": convert_to_numpy,
        }
        return self.fixed_output


def test_resolve_label_index_lowercases_names():
    id2label = {0: "CONTRADICTION", 1: "Entailment", 2: "neutral"}
    assert resolve_label_index(id2label) == {"contradiction": 0, "entailment": 1, "neutral": 2}


def test_score_remaps_contradiction_first_native_order():
    # native order: contradiction, entailment, neutral (matches
    # cross-encoder/nli-deberta-v3-base's real id2label)
    native = np.array([[0.1, 0.8, 0.1]], dtype="float32")
    fake_model = FakeCrossEncoderModel(native)
    label_index = {"contradiction": 0, "entailment": 1, "neutral": 2}
    checker = EntailmentChecker(config=GroundingConfig(), model=fake_model, label_index=label_index)

    result = checker.score([("premise", "hypothesis")])

    # fixed output order: (entailment, contradiction, neutral)
    np.testing.assert_allclose(result[0], [0.8, 0.1, 0.1])


def test_score_remaps_entailment_first_native_order():
    # a different checkpoint's native order: entailment, neutral, contradiction
    native = np.array([[0.7, 0.2, 0.1]], dtype="float32")
    fake_model = FakeCrossEncoderModel(native)
    label_index = {"entailment": 0, "neutral": 1, "contradiction": 2}
    checker = EntailmentChecker(config=GroundingConfig(), model=fake_model, label_index=label_index)

    result = checker.score([("premise", "hypothesis")])

    np.testing.assert_allclose(result[0], [0.7, 0.1, 0.2])


def test_score_passes_config_through_to_predict():
    native = np.array([[0.1, 0.2, 0.7]], dtype="float32")
    fake_model = FakeCrossEncoderModel(native)
    label_index = {"entailment": 0, "contradiction": 1, "neutral": 2}
    config = GroundingConfig(nli_batch_size=8, nli_show_progress_bar=True)
    checker = EntailmentChecker(config=config, model=fake_model, label_index=label_index)

    checker.score([("premise", "hypothesis")])

    assert fake_model.last_call_kwargs == {
        "batch_size": 8,
        "show_progress_bar": True,
        "apply_softmax": True,
        "convert_to_numpy": True,
    }


def test_score_handles_multiple_pairs():
    native = np.array(
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.1, 0.1, 0.8]],
        dtype="float32",
    )
    fake_model = FakeCrossEncoderModel(native)
    label_index = {"contradiction": 0, "entailment": 1, "neutral": 2}
    checker = EntailmentChecker(config=GroundingConfig(), model=fake_model, label_index=label_index)

    result = checker.score([("p1", "h1"), ("p2", "h2"), ("p3", "h3")])

    assert result.shape == (3, 3)
    np.testing.assert_allclose(result[0], [0.05, 0.9, 0.05])  # entailment, contradiction, neutral
    np.testing.assert_allclose(result[1], [0.9, 0.05, 0.05])
    np.testing.assert_allclose(result[2], [0.1, 0.1, 0.8])
