## Currently performing tokenization via tiktoken library and the gpt-4 tokenizer.
## Future work -> implement a tokenizer from scratch

import tiktoken
from transformers import GPT2Tokenizer, AutoTokenizer

def tokenize(text, llm_tokenizer="gpt-4", max_length=10, padding_token=0):
    """
    Tokenizes the input text using tiktoken, and pads or truncates the token sequence to a fixed length.

    Args:
        text (str): The input text to tokenize.
        llm_tokenizer (str): The name of the tokenizer model (default is "gpt-4").
        max_length (int): The desired length for each token sequence (default is 10).
        padding_token (int): The token to use for padding if the sequence is shorter than max_length (default is 0).

    Returns:
        List[int]: A list of tokens of length `max_length`.
    """

    if llm_tokenizer == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        tokens = tokenizer(text, return_tensors="pt")
        return tokens

    # Tokenizer
    enc = tiktoken.encoding_for_model(llm_tokenizer)

    # Tokenize the text
    tokens = enc.encode(text)

    # Adjust the length of tokens to max_length
    if len(tokens) > max_length:
        # Truncate if the sequence is longer than max_length
        tokens = tokens[:max_length]
    else:
        # Pad with the padding_token if the sequence is shorter than max_length
        tokens += [padding_token] * (max_length - len(tokens))

    return tokens

def decode(tokens, llm_tokenizer="gpt-4", padding_token=0):
    """
    Decodes a list of tokens back into text using the tiktoken tokenizer.

    Args:
        tokens (List[int]): The list of tokens to decode.
        llm_tokenizer (str): The name of the tokenizer model (default is "gpt-4").
        padding_token (int): The padding token used in the sequence (default is 0).

    Returns:
        str: The decoded text string.
    """
    # Tokenizer
    enc = tiktoken.encoding_for_model(llm_tokenizer)
    
    # Remove padding tokens
    tokens = [token for token in tokens if token != padding_token]
    
    # Decode tokens back into text
    text = enc.decode(tokens)
    
    return text