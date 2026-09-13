"""The two controls, and the check that makes them controls."""

from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import igraph as ig

from .graph import Graph


def weak_null(graph: Graph, seed: int = 0) -> Graph:
    """Erdős–Rényi control: same node count and edge count, nothing else.

    Weights, if present, are shuffled onto the new edges, so the weight
    distribution is preserved but its placement is random.  This is the
    control most papers use.  On its own it cannot tell wiring from degree.
    """
    rng = np.random.default_rng(seed)
    n, m = graph.n_nodes, graph.n_edges
    directed = graph.directed
    max_edges = n * (n - 1) if directed else n * (n - 1) // 2
    if m > max_edges:
        raise ValueError("more edges than a simple graph can hold")

    seen, out = set(), []
    while len(out) < m:
        need = m - len(out)
        a = rng.integers(0, n, size=need * 2 + 8)
        b = rng.integers(0, n, size=need * 2 + 8)
        for u, v in zip(a.tolist(), b.tolist()):
            if u == v:
                continue
            key = (u, v) if directed else (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) == m:
                break

    g = ig.Graph(n=n, edges=out, directed=directed)
    if graph.weighted:
        g.es["weight"] = rng.permutation(graph.weights()).tolist()
    return Graph(g)


def strict_null(graph: Graph, seed: int = 0, swaps_per_edge: int = 20, keep_weights: bool = True) -> Graph:
    """Degree-preserving control: every node keeps its exact degree.

    Directed graphs keep each node's in-degree and out-degree separately.
    Implemented with repeated double-edge swaps restricted to simple graphs
    (the configuration model conditioned on no self-loops or multi-edges).
    ``swaps_per_edge`` of 10-20 is enough to forget the original wiring; the
    report from ``verify`` says how much was actually rewired.

    ``keep_weights`` carries each weight with the edge slot it sat in, so the
    weight multiset is preserved.  ``keep_weights=False`` additionally
    shuffles weights across edges.
    """
    if swaps_per_edge <= 0:
        raise ValueError("swaps_per_edge must be positive")
    g = graph.g.copy()
    ig.set_random_number_generator(random.Random(seed))
    n_swaps = swaps_per_edge * max(graph.n_edges, 1)
    try:
        try:
            g.rewire(n=n_swaps, allowed_edge_types="simple")      # igraph >= 1.0
        except TypeError:
            g.rewire(n=n_swaps, mode="simple")                    # igraph 0.10 / 0.11
    finally:
        ig.set_random_number_generator(random)
    if graph.weighted:
        w = graph.weights()
        g.es["weight"] = w.tolist() if keep_weights else np.random.default_rng(seed + 1).permutation(w).tolist()
    return Graph(g)


def ensemble(graph: Graph, n: int = 20, seed: int = 0, kind: str = "strict", **kwargs) -> Iterator[Graph]:
    """Yield ``n`` independent nulls.  One draw is not a control; it is one draw."""
    fn = {"strict": strict_null, "weak": weak_null}[kind]
    for i in range(n):
        yield fn(graph, seed=seed + 1000 * i, **kwargs)


def verify(original: Graph, null: Graph, strict: bool = True) -> dict:
    """Check that a null is what it claims to be.  Raises if ``strict`` and degrees moved."""
    oi, oo = original.degrees()
    ni, no = null.degrees()
    el = null.g.get_edgelist()
    report = {
        "n_nodes_match": original.n_nodes == null.n_nodes,
        "n_edges_match": original.n_edges == null.n_edges,
        "in_degree_exact": bool(np.array_equal(oi, ni)),
        "out_degree_exact": bool(np.array_equal(oo, no)),
        "self_loops": int(sum(1 for u, v in el if u == v)),
        "multi_edges": null.n_edges - len(set(el)),
    }
    if not original.directed:
        report["degree_exact"] = report.pop("in_degree_exact")
        report.pop("out_degree_exact")
    orig_set = set(original.g.get_edgelist())
    shared = len(orig_set & set(el))
    report["edges_retained"] = shared
    report["fraction_rewired"] = 1.0 - shared / max(len(orig_set), 1)
    if original.weighted and null.weighted:
        report["weight_multiset_preserved"] = bool(np.allclose(np.sort(original.weights()), np.sort(null.weights())))
    if strict:
        deg_ok = report.get("degree_exact", report.get("in_degree_exact", False) and report.get("out_degree_exact", False))
        if not deg_ok:
            raise AssertionError("degree sequence was not preserved")
        if report["self_loops"] or report["multi_edges"]:
            raise AssertionError("null contains self-loops or multi-edges")
    return report
