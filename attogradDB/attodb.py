import numpy as np
import json
import os
from attogradDB.embedding import BertEmbedding
from attogradDB.indexing import HNSW

## Add support for custom tokenization and tiktoken
## Log perfomance for both brute force and hnsw indexing

class VectorStore:
    def __init__(self, indexing="hnsw", embedding_model="bert", save_index=False):
        self.vector = {}  
        self.index = {}
        if embedding_model == "bert":   
            self.embedding_model = BertEmbedding()
        if indexing == "hnsw":
            self.indexing = "hnsw"
            self.index = HNSW()
        elif indexing == "brute-force":
            self.indexing = "brute-force"

    def get_key_by_value(self, value):
        for key, val in self.vector.items():
            # Use np.array_equal for array comparison
            if np.array_equal(val, value):
                return key
        return None
    
    @staticmethod
    def similarity(vector_a, vector_b, method="cosine"):
        """Calculate similarity between two vectors."""
        if method == "cosine":
            return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        else:
            raise ValueError("Invalid similarity method")
    
    def add_text(self, vector_id, input_data):
        """
        Add a tokenized vector to the store.
        
        Args:
            vector_id (str): Identifier for the vector.
            input_data (str): Text input to be tokenized and stored.
            tokenizer (str): Tokenizer model to be used (default is "gpt-4").
        """

        tokenized_vector = np.array(self.embedding_model.embed(input_data))
        
        if self.indexing == "hnsw":
            self.vector[vector_id] = tokenized_vector
            self.index.add_node(tokenized_vector)
        else:
            self.vector[vector_id] = tokenized_vector
            self.update_index(vector_id, tokenized_vector)

    def add_documents(self, docs):
        """
        Create id's for each of the document
        Add the document to the store

        Args:
            docs (List[str]): List of documents to be added to the store.
        """
        idx = 0
        for doc in docs:
            self.add_text(f"doc_{idx}", doc)
            idx += 1
    
    def get_vector(self, vector_id, decode_results=False):
        """Retrieve vector by its ID."""
        if decode_results:
            # Return the n similar results decoded back to text
            decoded_text = self.embedding_model.reverse_embedding(self.vector.get(vector_id))
            return decoded_text
        else:
            return self.vector.get(vector_id)
    
    def update_index(self, vector_id, vector):
        """Update the similarity index with new vectors."""
        for existing_id, existing_vector in self.vector.items():
            if existing_id == vector_id:
                continue  # Skip if same vector
            cosine_similarity = self.similarity(vector, existing_vector)
            if existing_id not in self.index:
                self.index[existing_id] = {}
            self.index[existing_id][vector_id] = cosine_similarity
            if vector_id not in self.index:
                self.index[vector_id] = {}
            self.index[vector_id][existing_id] = cosine_similarity

    def get_similar(self, query_text, top_n=5, decode_results=True):
        """
        Get top N similar vectors to the query text.

        Args:
            query_text (str): Text input for the query to find similar vectors.
            top_n (int): Number of top similar vectors to return (default is 5).
            decode_results (bool): Whether to decode the results back to text (default is True).

        Returns:
            List[Tuple[str, float]]: List of vector IDs and their similarity scores.
        """
        query_vector = np.array(self.embedding_model.embed(query_text))
        
        results = []

        if self.indexing == "hnsw":
            nearest_vectors = self.index.search(query_vector, top_n)
            for vector in nearest_vectors:
                cosine_similarity = self.similarity(query_vector, vector)
                results.append((self.get_key_by_value(value=vector), cosine_similarity))
        
        else:
            for existing_id, existing_vector in self.vector.items():
                cosine_similarity = self.similarity(query_vector, existing_vector)
                results.append((existing_id, cosine_similarity))
        
            results.sort(key=lambda x: x[1], reverse=True)
                
        if decode_results:
            # Return the n similar results decoded back to text
            decoded_results = []
            for result in results[:top_n]:
                decoded_text = self.embedding_model.reverse_embedding(self.vector[result[0]])
                decoded_results.append((result[0], result[1], decoded_text))
            return decoded_results
        
        return results[:top_n]
    
    
class keyValueStore:
    def __init__(self, save_path = "data.json"):
        '''
        A json document intialized to store the key value pairs.
        Create an empty list if the file doesnt exist
        '''
        self.save_path = save_path
        # Ensure that the file exists
        if not os.path.exists(self.save_path):
            with open(self.save_path, "w") as outfile:
                json.dump([], outfile)

    def add(self, data):
        '''
        Adds a new dictionary to the JSON file
        '''
        with open(self.save_path, "r") as infile:
            try:
                existing_data = json.load(infile)
            except json.JSONDecodeError:
                existing_data = []
        
        if isinstance(data, list):
            for instance in data:
                existing_data.append(instance)
        else:
            existing_data.append(data)

        with open(self.save_path, "w") as outfile:
            json.dump(existing_data, outfile, indent=4)

    
    def __getitem__(self, key):
        '''
        Retrieves an item by key - searches for the dictionary containing a key
        '''
        with open(self.save_path, "r") as infile:
            existing_data = json.load(infile)

        return existing_data[key]
    
    def search(self, key, value):
        with open(self.save_path, "r") as infile:
            existing_data = json.load(infile)
        result = []
        for data in existing_data:
            if value == data[key]:
                result.append(data)
        return result
    
    def toVector(self, indexing="brute-force", embedding_model="bert"):
        '''
        Converts the key value store to Vector DB docs
        '''
        with open(self.save_path, "r") as infile:
            json_data = json.load(infile)
        
        # Convert the json docs into docs of strings
        docs = [json.dumps(entry, separators=(',', ':')) for entry in json_data]

        vectorStore = VectorStore(indexing=indexing, embedding_model=embedding_model)
        vectorStore.add_documents(docs)

        return vectorStore