import unittest
import numpy as np
from attogradDB.attodb import VectorStore

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore()
        self.vector_1 = np.array([1.0, 0.0, 0.0])
        self.vector_2 = np.array([0.0, 1.0, 0.0])
        self.vector_3 = np.array([1.0, 1.0, 0.0])

    def test_add_and_get_vector(self):
        self.store.add_text("vec1", self.vector_1)
        retrieved_vector = self.store.get_vector("vec1")
        np.testing.assert_array_equal(retrieved_vector, self.vector_1)

    def test_similarity(self):
        cosine_sim = self.store.similarity(self.vector_1, self.vector_2)
        self.assertAlmostEqual(cosine_sim, 0.0)

        cosine_sim = self.store.similarity(self.vector_1, self.vector_3)
        self.assertAlmostEqual(cosine_sim, 0.7071, places=4)

    def test_get_similar(self):
        self.store.add_text("vec1", self.vector_1)
        self.store.add_text("vec2", self.vector_2)
        self.store.add_text("vec3", self.vector_3)

        query = np.array([1.0, 0.5, 0.0])
        results = self.store.get_similar(query, top_n=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "vec3")  # vec3 should be the most similar

if __name__ == "__main__":
    unittest.main()
