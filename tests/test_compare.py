import json

import numpy as np
import pytest

import strictnull as sn
from tests.test_nulls import heavy_tailed


def test_compare_shows_degree_explains_clustering():
    """On a heavy-tailed graph with no planted structure, the weak control
    reports a large clustering gap; the strict control shows it was the
    degree sequence all along."""
    g = heavy_tailed(n=800, m=12000)
    rep = sn.compare(g, stats=["clustering", "reciprocity"], n_null=6, seed=0)
    row = {r.stat: r for r in rep.rows}["clustering"]
    assert row.real > row.weak_mean * 1.5              # looks structured against the weak control
    assert row.fraction_explained_by_degree > 0.8      # but it is the degree sequence
    assert abs(row.z_strict) < abs(row.z_weak)
    assert 0.85 < rep.fraction_rewired <= 1.0


def test_planted_structure_survives_the_strict_control():
    """Reciprocity planted on top of the degree sequence must stay visible."""
    g = heavy_tailed(n=500, m=5000)
    e = g.edges()
    rng = np.random.default_rng(0)
    pick = e[rng.choice(len(e), size=1500, replace=False)]
    both = np.concatenate([e, pick[:, ::-1]])
    h = sn.Graph.from_edges(both, directed=True, n_nodes=g.n_nodes)
    rep = sn.compare(h, stats=["reciprocity"], n_null=6, seed=1)
    row = rep.rows[0]
    assert row.z_strict > 10
    assert row.fraction_explained_by_degree < 0.5


def test_invariant_statistic_is_flagged():
    g = heavy_tailed(n=300, m=2000)
    rep = sn.compare(g, stats=["mean_weight", "clustering"], n_null=3, seed=0)
    inv = {r.stat: r.invariant_under_strict for r in rep.rows}
    assert inv["mean_weight"] is True and inv["clustering"] is False
    assert any("mean_weight" in w and "invariant" in w for w in rep.warnings)


def test_degree_only_statistic_is_flagged_under_strict_only():
    g = heavy_tailed(n=300, m=2000)

    def max_in_degree(graph):
        return float(graph.degrees()[0].max())

    rep = sn.compare(g, stats=[max_in_degree], n_null=3, seed=0)
    row = rep.rows[0]
    assert row.invariant_under_strict and not row.invariant_under_weak
    assert any("function of the degree sequence" in w for w in rep.warnings)


def test_one_draw_is_refused():
    g = heavy_tailed(n=100, m=500)
    with pytest.raises(ValueError):
        sn.compare(g, n_null=1)


def test_custom_stat_and_json_roundtrip(tmp_path):
    g = heavy_tailed(n=200, m=1200)

    @sn.register("edge_density")
    def edge_density(graph):
        n = graph.n_nodes
        return graph.n_edges / (n * (n - 1))

    rep = sn.compare(g, stats=["edge_density", "clustering"], n_null=2, seed=0)
    p = tmp_path / "r.json"
    rep.to_json(str(p))
    d = json.loads(p.read_text())
    assert {r["stat"] for r in d["rows"]} == {"edge_density", "clustering"}
    assert d["n_null"] == 2
    assert "INVARIANT" in str(rep)          # density is fixed by the edge count


def test_cli_runs_on_toy_csv(tmp_path, capsys):
    from strictnull.cli import main
    out = tmp_path / "out.json"
    rc = main(["tests/data/toy.csv", "--n-null", "3", "--stats", "clustering,reciprocity", "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "strict null" in text and out.exists()
