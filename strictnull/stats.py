"""Graph statistics.  Each takes a Graph and returns a float.

Register your own with ``@register("name")`` or pass callables to ``compare``.
"""

from __future__ import annotations

from typing import Callable, Dict

from .graph import Graph

STATS: Dict[str, Callable[[Graph], float]] = {}


def register(name: str):
    def deco(fn):
        STATS[name] = fn
        return fn
    return deco


def _undirected(graph: Graph):
    if not graph.directed:
        return graph.g
    u = graph.g.copy()
    u.to_undirected(combine_edges="first")
    return u


@register("reciprocity")
def reciprocity(graph: Graph) -> float:
    """Fraction of edges whose reverse edge also exists.  Undirected graphs: 1.0."""
    return float(graph.g.reciprocity()) if graph.directed else 1.0


@register("clustering")
def clustering(graph: Graph) -> float:
    """Global transitivity of the underlying undirected graph."""
    return float(_undirected(graph).transitivity_undirected(mode="zero"))


@register("avg_clustering")
def avg_clustering(graph: Graph) -> float:
    return float(_undirected(graph).transitivity_avglocal_undirected(mode="zero"))


@register("avg_path")
def avg_path(graph: Graph) -> float:
    """Mean shortest path over connected pairs, ignoring direction."""
    return float(_undirected(graph).average_path_length(directed=False))


@register("assortativity")
def assortativity(graph: Graph) -> float:
    return float(_undirected(graph).assortativity_degree(directed=False))


@register("n_triangles")
def n_triangles(graph: Graph) -> float:
    return float(len(_undirected(graph).cliques(min=3, max=3)))


@register("max_core")
def max_core(graph: Graph) -> float:
    return float(max(_undirected(graph).coreness()) if graph.n_nodes else 0.0)


@register("modularity_leiden")
def modularity_leiden(graph: Graph) -> float:
    """Modularity of the best Leiden partition (undirected).  Seeded for reproducibility."""
    u = _undirected(graph)
    part = u.community_leiden(objective_function="modularity", n_iterations=-1)
    return float(part.modularity)


@register("mean_weight")
def mean_weight(graph: Graph) -> float:
    """Present so an invariant statistic is on hand: weights are carried by every null."""
    w = graph.weights()
    return float(w.mean()) if w is not None and len(w) else float("nan")


DEFAULT = ["reciprocity", "clustering", "avg_path", "assortativity"]
