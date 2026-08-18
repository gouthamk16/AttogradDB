import numpy as np
from transformers import AutoModel

from attogradDB.tokenizer import tokenize


class BertEmbedding:
    def __init__(self, model: str = "bert-base-uncased", tokenizer: str | None = None):
        self.model = AutoModel.from_pretrained(model)
        # Defaulting the tokenizer to the model keeps the two from silently diverging.
        self.llm_tokenizer = tokenizer or model

    def embed(self, text: str) -> np.ndarray:
        inputs = tokenize(text, llm_tokenizer=self.llm_tokenizer)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
