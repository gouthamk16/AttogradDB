from attogradDB.attodb import keyValueStore

# Initialize the store with the path to the JSON file
# Creates a new json file if it doesn't exist with default master collection and collection
store = keyValueStore(json_path="data.json")

# Create a new master collection (optional)
store.create_master_collection("users")

# Create a new collection within the master collection (optional) 
store.create_collection("employees", master_collection="users")

# Switch to the new collection
store.use_collection("employees", master_collection="users")

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

# Add documents with auto-generated UUIDs
store.add(sample_data)

# Example for adding multiple json documents into a collection
# create a new master collection and a collection
store.create_master_collection("test1")
store.create_collection("sample_data", master_collection="test1")

# switch to the new collection
store.use_collection("sample_data", master_collection="test1")

# add multiple json documents into the collection
store.add_json("input_data/example_1.json")
store.add_json("input_data/example_2.json")


# Example retrieval for a specific key:
print("First document in the store:", store[0])       # Accesses the first dictionary in the store

print("Documents with id 104:", store.search("id", 104))  # Access the data instance with "id" value 104

# Convert current collection to vector store
semantic_store = store.toVector(indexing="hnsw", embedding_model="bert", collection="employees", master_collection="users")

# Or specify a different collection to convert
# semantic_store = store.toVector(indexing="hnsw", embedding_model="bert", 
#                                collection="employees", master_collection="users")

query = "Diana diana5678"
results = semantic_store.get_similar(query, top_n=1) # (vectorid, similarity, instance)
print("Similar documents to query:", results)