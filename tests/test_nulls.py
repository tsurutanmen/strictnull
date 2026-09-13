import numpy as np
import pytest

import strictnull as sn


def heavy_tailed(n=600, m=6000, seed=7, directed=True):
    """Directed graph with heavy-tailed degrees, like a real connectome."""
    rng = np.random.default_rng(seed)
    pref = rng.pareto(1.5, size=n) + 1.0
    p_out = pref / pref.sum()
    p_in = rng.permutation(pref); p_in = p_in / p_in.sum()
    seen = set()
    while len(seen) < m:
        u = rng.choice(n, size=(m - len(seen)) * 2, p=p_out)
        v = rng.choice(n, size=(m - len(seen)) * 2, p=p_in)
        for a, b in zip(u.tolist(), v.tolist()):
            if a != b:
                seen.add((a, b) if directed else (min(a, b), max(a, b)))
            if len(seen) == m:
                break
    edges = np.array(sorted(seen), dtype=np.int64)
    weights = rng.lognormal(1.0, 1.0, size=len(edges)).round() + 1
    return sn.Graph.from_edges(edges, weights, directed=directed, n_nodes=n)


@pytest.mark.parametrize("directed", [True, False])
def test_strict_null_preserves_degrees_exactly(directed):
    g = heavy_tailed(directed=directed)
    h = sn.strict_null(g, seed=1)
    rep = sn.verify(g, h, strict=True)
    assert rep["n_edges_match"] and rep["n_nodes_match"]
    assert rep["self_loops"] == 0 and rep["multi_edges"] == 0
    assert rep["fraction_rewired"] > 0.85
    assert rep["weight_multiset_preserved"]
    oi, oo = g.degrees(); ni, no = h.degrees()
    assert np.array_equal(oi, ni) and np.array_equal(oo, no)


@pytest.mark.parametrize("directed", [True, False])
def test_weak_null_keeps_counts_only(directed):
    g = heavy_tailed(directed=directed)
    h = sn.weak_null(g, seed=1)
    rep = sn.verify(g, h, strict=False)
    assert rep["n_edges_match"] and rep["n_nodes_match"]
    assert rep["self_loops"] == 0 and rep["multi_edges"] == 0
    key = "degree_exact" if not directed else "in_degree_exact"
    assert not rep[key]
    assert rep["weight_multiset_preserved"]


def test_verify_raises_on_a_fake_strict_null():
    g = heavy_tailed()
    h = sn.weak_null(g, seed=3)
    with pytest.raises(AssertionError):
        sn.verify(g, h, strict=True)


def test_seeds_reproduce_and_differ():
    g = heavy_tailed()
    a = sn.strict_null(g, seed=5).edges()
    b = sn.strict_null(g, seed=5).edges()
    c = sn.strict_null(g, seed=6).edges()
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_ensemble_draws_are_independent():
    g = heavy_tailed()
    draws = [h.edges() for h in sn.ensemble(g, n=3, seed=0)]
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])


def test_shuffle_weights_option():
    g = heavy_tailed()
    h = sn.strict_null(g, seed=1, keep_weights=False)
    assert sn.verify(g, h)["weight_multiset_preserved"]
    # same edge slots, different weights than the kept version
    k = sn.strict_null(g, seed=1, keep_weights=True)
    assert np.array_equal(h.edges(), k.edges())
    assert not np.array_equal(h.weights(), k.weights())


def test_construction_cleans_self_loops_and_duplicates():
    g = sn.Graph.from_edges([(0, 1), (1, 2), (2, 2), (0, 1)], [1, 1, 5, 2], directed=True, n_nodes=3)
    assert g.n_edges == 2 and g.dropped_self_loops == 1 and g.merged_duplicates == 1
    assert sorted(g.weights().tolist()) == [1.0, 3.0]


def test_undirected_edges_are_canonical():
    g = sn.Graph.from_edges([(1, 0), (0, 1), (2, 1)], directed=False)
    assert g.n_edges == 2 and g.merged_duplicates == 1


def test_from_adjacency_and_networkx():
    A = np.array([[0, 2, 0], [0, 0, 1], [3, 0, 0]])
    g = sn.Graph.from_adjacency(A, directed=True)
    assert g.n_edges == 3 and sorted(g.weights().tolist()) == [1.0, 2.0, 3.0]
    nx = pytest.importorskip("networkx")
    G = nx.DiGraph(); G.add_edge("a", "b", weight=2.0); G.add_edge("b", "c")
    h = sn.Graph.from_networkx(G)
    assert h.n_nodes == 3 and h.n_edges == 2 and h.labels == ["a", "b", "c"]
