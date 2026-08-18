from functools import cache

import tiktoken
from transformers import AutoTokenizer


@cache
def _hf_tokenizer(name: str):
    """Cached because embed() tokenizes once per document and loading is not free."""
    return AutoTokenizer.from_pretrained(name)


def tokenize(text: str, llm_tokenizer: str = "gpt-4", max_length: int = 10, padding_token: int = 0):
    """Tokenize text with tiktoken, or with a HuggingFace tokenizer if the name is not an
    OpenAI model.

    Returns a padded/truncated list of ints for tiktoken, or a tensor mapping for
    HuggingFace models -- the caller knows which backend it asked for.
    """
    try:
        enc = tiktoken.encoding_for_model(llm_tokenizer)
    except KeyError:
        # Not an OpenAI model name, so treat it as a HuggingFace repo id. truncation=True
        # applies the tokenizer's own model_max_length (512 for BERT), without which any
        # longer document blows up inside the model's position embeddings.
        return _hf_tokenizer(llm_tokenizer)(text, return_tensors="pt", truncation=True)

    tokens = enc.encode(text)
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + [padding_token] * (max_length - len(tokens))


def decode(tokens: list[int], llm_tokenizer: str = "gpt-4", padding_token: int = 0) -> str:
    """Decode tiktoken ids back into text, dropping padding."""
    enc = tiktoken.encoding_for_model(llm_tokenizer)
    return enc.decode([token for token in tokens if token != padding_token])
