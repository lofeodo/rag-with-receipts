"""Tokenizer access for chunk sizing, matched to the embedding model (Step 2)."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import logging as hf_logging

# We deliberately count/encode text far longer than the model's 512-token limit
# before it's been chunked down; suppress the resulting "sequence length is
# longer than..." warning, which is expected noise here, not a real problem.
hf_logging.set_verbosity_error()


@lru_cache(maxsize=1)
def get_tokenizer(name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(name)


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
