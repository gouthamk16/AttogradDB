# attoDB/embedding.py
from transformers import AutoModel
from attogradDB.tokenizer import tokenize

## Embedding function that uses huggungface embeddings to produce embeddings for thr given token
class BertEmbedding():
    def __init__(self, model="bert-base-uncased", tokenizer="bert-base-uncased"):
        self.model = model
        self.model = AutoModel.from_pretrained(model)
        self.llm_tokenizer = tokenizer
        self.embedding_map = {}
    
    def embed(self, text):
        inputs = tokenize(text, llm_tokenizer=self.llm_tokenizer)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        self.embedding_map[tuple(embedding)] = text
        return embedding

    def reverse_embedding(self, embedding):
        text = self.embedding_map.get(tuple(embedding))
        return text
    
    
# text = "Hello im the price of russia."
# embeddings = BertEmbedding()
# print(embeddings.embed(text))
# print(len(embeddings.embed(text)))
# print(embeddings.reverse_embedding(embeddings.embed(text)))
