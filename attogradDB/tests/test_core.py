import unittest
import numpy as np
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attogradDB.attodb import VectorStore

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore()
        self.vector_1 = "Hey there, Im attoDB"
        self.vector_2 = "Hey there, Im attoDB"
        self.vector_3 = "Hey there, Im Meta"
        self.cosine_example_1 = [0.983, 0.0123, 0.6412, 0.5633, 1.6743]
        self.cosine_example_2 = [0.2133, 0.4532, 0.9876, 0.1954, 3.2453]

    def test_add_and_get_vector(self):
        self.store.add_text("vec1", self.vector_1)
        retrieved_vector = self.store.get_vector("vec1", decode_results=True)
        self.assertEqual(self.vector_1, retrieved_vector)

    def test_similarity(self):
        cosine_sim = self.store.similarity(self.cosine_example_1, self.cosine_example_2)
        self.assertAlmostEqual(cosine_sim, 0.877518170548048)

    def test_get_similar(self):
        self.store.add_text("vec1", self.vector_1)
        self.store.add_text("vec2", self.vector_2)
        self.store.add_text("vec3", self.vector_3)

        query = "Hey there, Im Meta"
        results = self.store.get_similar(query, top_n=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "vec3")  # vec3 should be the most similar

if __name__ == "__main__":
    unittest.main()
