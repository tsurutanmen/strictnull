"""A thin wrapper around igraph.Graph that remembers what it was built from."""

from __future__ import annotations

import csv
from typing import Iterable, Optional

import numpy as np

try:
    import igraph as ig
except ImportError as exc:  # pragma: no cover
    raise ImportError("strictnull needs python-igraph: pip install igraph") from exc


class Graph:
    """Directed or undirected simple graph with optional edge weights.

    Self-loops are dropped on construction and duplicate edges are merged
    (weights summed), because the null models are defined on simple graphs.
    """

    def __init__(self, g: "ig.Graph", dropped_self_loops: int = 0, merged_duplicates: int = 0):
        self.g = g
        self.directed = g.is_directed()
        self.dropped_self_loops = dropped_self_loops
        self.merged_duplicates = merged_duplicates

    # ------------------------------------------------------------ builders
    @classmethod
    def from_edges(cls, edges, weights=None, directed: bool = True, n_nodes: Optional[int] = None) -> "Graph":
        edges = np.asarray(edges, dtype=np.int64)
        if edges.size == 0:
            edges = edges.reshape(0, 2)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape (E, 2), got {edges.shape}")
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if len(weights) != len(edges):
                raise ValueError("weights length must match edges length")
        if n_nodes is None:
            n_nodes = int(edges.max()) + 1 if len(edges) else 0

        keep = edges[:, 0] != edges[:, 1]
        dropped = int((~keep).sum())
        edges = edges[keep]
        if weights is not None:
            weights = weights[keep]

        if not directed:
            edges = np.sort(edges, axis=1)
        merged = {}
        for i, (u, v) in enumerate(map(tuple, edges.tolist())):
            if (u, v) in merged:
                if weights is not None:
                    merged[(u, v)] += float(weights[i])
            else:
                merged[(u, v)] = float(weights[i]) if weights is not None else None
        n_dup = len(edges) - len(merged)

        g = ig.Graph(n=n_nodes, edges=list(merged.keys()), directed=directed)
        if weights is not None:
            g.es["weight"] = list(merged.values())
        return cls(g, dropped_self_loops=dropped, merged_duplicates=n_dup)

    @classmethod
    def from_adjacency(cls, A, directed: bool = True) -> "Graph":
        A = np.asarray(A)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adjacency must be square")
        src, dst = np.nonzero(A)
        if not directed:
            keep = src <= dst
            src, dst = src[keep], dst[keep]
        w = A[src, dst].astype(np.float64)
        return cls.from_edges(np.stack([src, dst], axis=1), w, directed=directed, n_nodes=A.shape[0])

    @classmethod
    def from_networkx(cls, G) -> "Graph":
        nodes = list(G.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        edges, weights, has_w = [], [], False
        for u, v, d in G.edges(data=True):
            edges.append((index[u], index[v]))
            if "weight" in d:
                has_w = True
            weights.append(float(d.get("weight", 1.0)))
        out = cls.from_edges(edges, weights if has_w else None, directed=G.is_directed(), n_nodes=len(nodes))
        out.labels = nodes
        return out

    @classmethod
    def from_csv(cls, path, directed: bool = True, delimiter: str = ",") -> "Graph":
        """Edge list with a header. Columns: source, target[, weight]. Node ids may be any strings."""
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            if len(header) < 2:
                raise ValueError("need at least two columns: source, target")
            has_w = len(header) >= 3
            for r in reader:
                if not r or not r[0].strip():
                    continue
                rows.append((r[0].strip(), r[1].strip(), float(r[2]) if has_w and len(r) > 2 and r[2].strip() else 1.0))
        labels = sorted({r[0] for r in rows} | {r[1] for r in rows})
        index = {n: i for i, n in enumerate(labels)}
        edges = [(index[a], index[b]) for a, b, _ in rows]
        weights = [w for _, _, w in rows] if has_w else None
        out = cls.from_edges(edges, weights, directed=directed, n_nodes=len(labels))
        out.labels = labels
        return out

    # ------------------------------------------------------------ accessors
    @property
    def n_nodes(self) -> int:
        return self.g.vcount()

    @property
    def n_edges(self) -> int:
        return self.g.ecount()

    @property
    def weighted(self) -> bool:
        return "weight" in self.g.edge_attributes()

    def edges(self) -> np.ndarray:
        e = np.array(self.g.get_edgelist(), dtype=np.int64)
        return e.reshape(-1, 2)

    def weights(self) -> Optional[np.ndarray]:
        return np.asarray(self.g.es["weight"], dtype=np.float64) if self.weighted else None

    def degrees(self):
        """(in, out) for directed graphs; (deg, deg) for undirected."""
        if self.directed:
            return (np.asarray(self.g.degree(mode="in"), dtype=np.int64),
                    np.asarray(self.g.degree(mode="out"), dtype=np.int64))
        d = np.asarray(self.g.degree(), dtype=np.int64)
        return d, d

    def copy(self) -> "Graph":
        out = Graph(self.g.copy(), self.dropped_self_loops, self.merged_duplicates)
        if hasattr(self, "labels"):
            out.labels = self.labels
        return out

    def __repr__(self) -> str:
        kind = "directed" if self.directed else "undirected"
        w = ", weighted" if self.weighted else ""
        return f"Graph({kind}{w}, {self.n_nodes} nodes, {self.n_edges} edges)"
