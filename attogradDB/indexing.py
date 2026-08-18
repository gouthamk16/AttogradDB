import hnswlib
import numpy as np


class HNSW:
    def __init__(
        self,
        space: str = "cosine",
        dim: int = 768,
        max_elements: int = 1000,
        ef_construction: int = 200,
        M: int = 16,
    ):
        self.dim = dim
        self.ef_construction = ef_construction
        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(ef_construction)
        self.count = 0

    def add_node(self, vector) -> int:
        """Insert a vector and return the label it was stored under."""
        capacity = self.index.get_max_elements()
        if self.count >= capacity:
            self.index.resize_index(capacity * 2)
        label = self.count
        self.index.add_items(np.asarray([vector], dtype=np.float32), [label])
        self.count += 1
        return label

    def search(self, query_vector, top_n: int = 5) -> list[tuple[int, float]]:
        """Return (label, cosine similarity) pairs, nearest first."""
        if self.count == 0:
            return []

        # hnswlib raises if k exceeds the element count, and ef below k silently hurts recall.
        k = min(top_n, self.count)
        self.index.set_ef(max(self.ef_construction, k))
        labels, distances = self.index.knn_query(
            np.asarray(query_vector, dtype=np.float32), k=k
        )
        # The 'cosine' space stores 1 - cosine_similarity as its distance.
        return [
            (int(label), 1.0 - float(distance))
            for label, distance in zip(labels[0], distances[0])
        ]
