from attogradDB.attodb import keyValueStore


store = keyValueStore(save_path="data.json")

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

print(store.search("id", 2))  # Access the data instace with "id" value 2

semantic_store = store.toVector()

query = "Diana diana5678"
results = semantic_store.get_similar(query, top_n=1) # (vectorid, similarity, instance)
print(results)