from random import random
from math import floor, log
import networkx as nx
import numpy as np

## Use cosine distance instead of euclidean distance - not working - need fix
## Can also use hsnwlib - no fun

class HNSW:
    def __init__(self, layers=10, maxk=5, ef_construction=5, ml=3.0):
        self.layers = layers
        self.maxk = maxk
        self.ef_construction = ef_construction
        self.ml = ml
        self.hnsw = {
            "entrance": 0,
            "layers": [nx.Graph() for _ in range(layers)],
        }
    
    @staticmethod
    def similarity(vector_a, vector_b, method="cosine"):
        """Calculate similarity between two vectors."""
        if method == "cosine":
            return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        else:
            raise ValueError("Invalid similarity method")

    def add_node(self, vector):
        """Insert a new vector (node) into the HNSW index."""
        q = tuple(vector)  # Convert vector to a tuple if it’s an array
        self._insert(self.hnsw, q)

    def search(self, query_vector, top_n=5):
        """Search for the top N most similar vectors to the query_vector."""
        results = []
        for i in range(len(self.hnsw["layers"])):
            layer = self.hnsw["layers"][i]
            eP = self.hnsw["entrance"]
            nearest_neighbors = self._search_layer(layer, query_vector, eP, top_n)
            results.append(sorted(nearest_neighbors, key=lambda x: self._distance(x, query_vector)))
        results = sum(results, [])
        return results[:top_n]

    def _insert(self, HNSW, q):
        """Insert a new node into the HNSW index."""
        W = set()
        eP = self._entrance_point(HNSW)
        topL = self._top_layer(HNSW)

        layer_i = floor(-1 * log(random()) * self.ml)
        layer_i = min(layer_i, topL)

        for lc in range(topL, layer_i, -1):
            layer = self._get_layer(HNSW, lc)
            W = self._search_layer(layer, q, eP, ef=1)
            eP = self._nearest(W, q)

        for lc in range(layer_i, -1, -1):
            layer = self._get_layer(HNSW, lc)
            W = self._search_layer(layer, q, eP, self.ef_construction)
            neighbors = self._select_neighbors(q, W, self.maxk)
            self._add_edges(layer, q, neighbors)

            for e in neighbors:
                e_neighbors = self._neighborhood(layer, e)
                if len(e_neighbors) > self.maxk:
                    e_new_edges = self._select_neighbors(e, e_neighbors, self.maxk)
                    self._shrink_edges(layer, e, e_new_edges)

            eP = self._nearest(W, q)
        self._set_entrance_point(HNSW, eP)

    def _search_layer(self, layer, q, eP, ef):
        """Search for the top N most similar vectors to the query q in the given layer."""
        visited = set([eP])
        cands = set([eP])
        nearestN = set([eP])

        while cands:
            c = self._nearest(cands, q)
            cands.remove(c)
            f = self._furthest(nearestN, q)
            if self._distance(c, q) > self._distance(f, q):
                break

            for e in self._neighborhood(layer, c):
                if e not in visited:
                    visited.add(e)
                    f = self._furthest(nearestN, q)
                    if self._distance(e, q) < self._distance(f, q) or len(nearestN) < ef:
                        cands.add(e)
                        nearestN.add(e)
                        if len(nearestN) > ef:
                            nearestN.remove(self._furthest(nearestN, q))
        return nearestN

    def _select_neighbors(self, q, cands, M):
        """Select M nearest neighbors for the query q from the candidates."""
        q = tuple(q) if isinstance(q, (list, np.ndarray)) else q  # Ensure q is a tuple if it's an array or list
        # Convert elements in `cands` to tuples if they are arrays, otherwise keep as is
        cands = {tuple(c) if isinstance(c, (list, np.ndarray)) else c for c in cands}
        
        if q in cands:
            cands.remove(q)
        return set(sorted(cands, key=lambda x: self._distance(x, q))[:M])

    def _entrance_point(self, HNSW):
        return HNSW["entrance"]

    def _set_entrance_point(self, HNSW, eP):
        HNSW["entrance"] = eP

    def _get_layer(self, HNSW, lc):
        return HNSW["layers"][lc]

    def _top_layer(self, HNSW):
        return len(HNSW["layers"]) - 1

    def _distance(self, u, v):
        """Compute the Euclidean distance between two vectors `u` and `v`."""
        u, v = np.array(u), np.array(v)
        return np.linalg.norm(u - v)

    def _furthest(self, W, q):
        return max(W, key=lambda w: self._distance(w, q))

    def _nearest(self, W, q):
        return min(W, key=lambda w: self._distance(w, q))

    def _neighborhood(self, layer, u):
        if u not in layer:
            return set()
        return set(layer[u])

    def _add_edges(self, layer, u, neighbors):
        for n in neighbors:
            layer.add_edge(u, n)

    def _shrink_edges(self, layer, u, new_edges):
        removes = [(u, n) for n in layer[u] if n not in new_edges]
        layer.remove_edges_from(removes)