from rag_receipts.grounding.overlap import token_overlap


def test_exact_match_is_full_overlap():
    assert token_overlap("the sky is blue", "the sky is blue") == 1.0


def test_partial_overlap():
    # claim tokens: {the, sky, is, blue} (4); chunk has {the, sky} (2 match)
    score = token_overlap("the sky is blue", "the sky has clouds")
    assert score == 2 / 4


def test_no_overlap():
    assert token_overlap("dragons breathe fire", "goblins carry bronze swords") == 0.0


def test_case_insensitive():
    assert token_overlap("Dragon Slayer", "the dragon slayer quest") == 1.0


def test_punctuation_stripped():
    # \w+ splits "Zamorak's" into {zamorak, s} - claim tokens: {zamorak, s, brew} (3),
    # chunk matches {zamorak, brew} (2)
    score = token_overlap("Zamorak's brew!", "a zamorak brew potion")
    assert score == 2 / 3


def test_empty_claim_returns_zero():
    assert token_overlap("", "some chunk text") == 0.0


def test_extra_chunk_tokens_do_not_inflate_score():
    # overlap ratio is over claim tokens only, not chunk tokens
    score = token_overlap("gold bar", "you need a gold bar and many other items too")
    assert score == 1.0
