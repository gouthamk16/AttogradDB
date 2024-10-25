import sys
import os

# Add the directory containing attodb to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attogradDB.attodb import VectorStore
from attogradDB.utils import read_pdf
from attogradDB.io import TextSplitter


## Document store implementation (vectorstore on top of textsplitter)

pdf_path = "../data/leclerc_sample.pdf"
text = read_pdf(pdf_path)
splitter = TextSplitter()
splitter.split_text(text)
docs = splitter.get_docs(extract=True)

store = VectorStore(embedding_model="bert")
store.add_documents(docs)

query = "fia formula 2"
results = store.get_similar(query, top_n=2)
print(results)


## Plaintext Vector store implementation

# Initialize the vector store
store = VectorStore(embedding_model="bert")

# Add vectors using text input (which will be tokenized using the 'gpt-4' tokenizer)
store.add_text("vec1", "The quick brown fox")
store.add_text("vec2", "The lazy dog")
store.add_text("vec3", "A quick fox jumps over")

# Query the store with a text input and find the top 2 similar vectors
results = store.get_similar("The brown fox is quick", tokenizer="gpt-4", max_length=10, top_n=2)

# Output results
print(store.vector)
print(store.index)
print(results)
