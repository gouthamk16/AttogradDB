import sys
import os
import time

# Add the directory containing attodb to sys.path

from attogradDB.attodb import VectorStore
from attogradDB.utils import read_pdf
from attogradDB.io import TextSplitter


## Document store implementation (vectorstore on top of textsplitter)
## Brute Force Indexing

# print("\nDocument store and retrieval using brute-force index\n")

# pdf_path = "../sample_data/dpo_paper.pdf"
# text = read_pdf(pdf_path)
# splitter = TextSplitter(chunk_size=300, chunk_overlap=20)
# splitter.split_text(text)
# docs = splitter.get_docs(extract=True)

# store = VectorStore(indexing="brute-force", embedding_model="bert") 
# add_start = time.perf_counter()
# store.add_documents(docs)
# add_end = time.perf_counter()
# print(f"Time taken to add documents: {add_end - add_start} seconds")

# query = "reward model"
# search_start = time.perf_counter()
# results = store.get_similar(query, top_n=2)
# search_end = time.perf_counter()
# print(f"Time taken to search: {search_end - search_start} seconds")
# print(results)


# ## Plaintext Vector store implementation

# # Initialize the vector store
# store = VectorStore(embedding_model="bert")

# # Add vectors using text input (which will be tokenized using the 'gpt-4' tokenizer)
# store.add_text("vec1", "The quick brown fox")
# store.add_text("vec2", "The lazy dog")
# store.add_text("vec3", "A quick fox jumps over")

# # Query the store with a text input and find the top 2 similar vectors
# results = store.get_similar("The brown fox is quick", max_length=10, top_n=2)

# # Output results
# print(store.vector)
# print(store.index)
# print(results)


# # Sample usage for hnsw index

# # Initialize the vector store with the 'hnsw' indexing method
# store = VectorStore()

# # Add vectors using text input (which will be tokenized using the 'gpt-4' tokenizer)
# store.add_text("vec1", "The quick brown fox")
# store.add_text("vec2", "The lazy dog")
# store.add_text("vec3", "A quick fox jumps over")

# # Query the store with a text input and find the top 2 similar vectors
# results = store.get_similar("The brown fox is quick", top_n=1)

# # Output results
# print(store.vector)
# print(store.index)
# print(results)

## Document store and retrieval using HNSW index

print("\nDocument store and retrieval using HNSW index\n")

pdf_path = "../sample_data/dpo_paper.pdf"
text = read_pdf(pdf_path)
splitter = TextSplitter(chunk_size=350, chunk_overlap=20)
splitter.split_text(text)
docs = splitter.get_docs(extract=True)

store2 = VectorStore(indexing="hnsw")
addhnsw_start = time.perf_counter()
store2.add_documents(docs)
addhnsw_end = time.perf_counter()
print(f"Time taken to add documents: {addhnsw_end - addhnsw_start} seconds")

query = "reward model"
searchhnsw_start = time.perf_counter()
results = store2.get_similar(query, top_n=4)
searchhnsw_end = time.perf_counter()
print(f"Time taken to search: {searchhnsw_end - searchhnsw_start} seconds")
for result in results:
    print(result)