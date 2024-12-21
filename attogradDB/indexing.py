import numpy as np
import hnswlib

class HNSW:
    def __init__(self, space="cosine", dim=768, max_elements=1000, ef_construction=200, M=16):
        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_construction)
        self.label = 0
        self.label_to_vector = {}
    
    def add_node(self, vector):
        vector = np.array(vector, dtype=np.float32)
        self.index.add_items([vector], [self.label])
        self.label_to_vector[self.label] = vector
        self.label += 1

    def search(self, query_vector, top_n=5):
        query_vector = np.array(query_vector, dtype=np.float32)
        labels, distances = self.index.knn_query(query_vector, k=top_n)
        vectors = [self.label_to_vector[label] for label in labels[0]]
        return vectors
    
    def save_index(self, filepath):
        self.index.save_index(filepath)
    
    def load_index(self, filepath):
        self.index.load_index(filepath, max_elements=self.max_elements)
