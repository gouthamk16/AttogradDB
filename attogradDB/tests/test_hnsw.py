# Tests for the hnsw indexing as a docstore

import unittest
import numpy as np
import sys
import os

from attogradDB.attodb import VectorStore

class TestVectorStore_HNSW(unittest.TestCase):
    
        def setUp(self):
            self.vectorStore = VectorStore(indexing="hnsw")
            self.input_vector1 = "The quick brown fox"
            self.input_vector2 = "The cat sat on the mat"
            self.input_vector3 = "The rabbit hole is very deep"
            self.test_query1 = "The quick black fox"
            self.text_query2 = "The cat sat on the floor"
    
        def test_add_and_get_vector(self):
            self.vectorStore.add_text("vec1", self.input_vector1)
            result = self.vectorStore.get_vector("vec1", decode_results=True)
            self.assertEqual(result, self.input_vector1)

        def test_hnsw(self):
             self.vectorStore.add_text("vec1", self.input_vector1)
             self.vectorStore.add_text("vec2", self.input_vector2)
             self.vectorStore.add_text("vec3", self.input_vector3)
             results1 = self.vectorStore.get_similar(self.test_query1, top_n=1)
             results2 = self.vectorStore.get_similar(self.text_query2, top_n=1)
             self.assertEqual(results1[0][0], "vec1")
             self.assertEqual(results2[0][0], "vec2")

if __name__ == "__main__":
    unittest.main()