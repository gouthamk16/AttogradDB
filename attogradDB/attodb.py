import numpy as np
from attogradDB.embedding import BertEmbedding

## Add support for custome tokenization and tiktoken

class VectorStore:
    def __init__(self, embedding_model="bert"):
        self.vector = {}  
        self.index = {}
        if embedding_model == "bert":   
            self.embedding_model = BertEmbedding()
    
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
            max_length (int): Maximum length of the tokenized vector (default is 10).
            padding_token (int): Padding token to be used if the sequence is shorter than max_length (default is 0).
        """

        tokenized_vector = np.array(self.embedding_model.embed(input_data))
        
        self.vector[vector_id] = tokenized_vector
        
        self.update_index(vector_id, tokenized_vector)

    def add_documents(self, docs):
        ## Create id's for each of the document
        idx = 0
        for doc in docs:
            self.add_text(f"doc_{idx}", doc)
            idx += 1
    
    def get_vector(self, vector_id):
        """Retrieve vector by its ID."""
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
            tokenizer (str): Tokenizer model to be used for tokenizing the query (default is "gpt-4").
            max_length (int): Maximum length of the tokenized vector (default is 10).
            padding_token (int): Padding token to be used if the sequence is shorter than max_length (default is 0).
            top_n (int): Number of top similar vectors to return (default is 5).

        Returns:
            List[Tuple[str, float]]: List of vector IDs and their similarity scores.
        """
        query_vector = np.array(self.embedding_model.embed(query_text))
        
        results = []
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