from attogradDB.attodb import keyValueStore

# Initialize the store with the path to the JSON file
# Creates a new json file if it doesn't exist
store = keyValueStore(json_path="data.json")

# Loading a sample data into the store - ignore if data is loaded directly from a json file
sample_data = [
    {
        "id": 101,
        "name": "Alice",
        "contact": "alice@example.com"
    },
    {
        "id": 102,
        "name": "Bob",
        "contact": "bob@example.com"
    },
    {
        "id": 103,
        "name": "Charlie",
        "contact": None
    },
    {
        "id": 104,
        "name": "Diana",
        "contact": "diana@example.com"
    },
    {
        "id": 105,
        "name": "Diana",
        "contact": "diana5678@example.com"
    }
]

store.add(sample_data)

# Example retrieval for a specific key:
print(store[0])       # Accesses the first dictionary in the store

print(store.search("id", 104))  # Access the data instace with "id" value 2

semantic_store = store.toVector(indexing="hnsw", embedding_model="bert")

query = "Diana diana5678"
results = semantic_store.get_similar(query, top_n=1) # (vectorid, similarity, instance)
print(results)