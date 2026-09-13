"""strictnull EDGES.csv [--undirected] [--n-null 20] [--stats a,b,c] [--json out.json]"""

from __future__ import annotations

import argparse
import sys

from .graph import Graph
from .stats import STATS, DEFAULT
from .compare import compare


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="strictnull",
        description="Compare a graph against a weak (edge-count) and a strict (degree-preserving) null model.")
    ap.add_argument("edges", help="CSV edge list with header: source,target[,weight]")
    ap.add_argument("--undirected", action="store_true", help="treat edges as undirected")
    ap.add_argument("--n-null", type=int, default=20, help="draws per control (default 20)")
    ap.add_argument("--stats", default=",".join(DEFAULT),
                    help="comma-separated statistics; known: " + ", ".join(sorted(STATS)))
    ap.add_argument("--swaps-per-edge", type=int, default=20)
    ap.add_argument("--shuffle-weights", action="store_true",
                    help="also shuffle weights across edges in the strict null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the report as JSON to this path")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    g = Graph.from_csv(a.edges, directed=not a.undirected, delimiter=a.delimiter)
    stats = [s.strip() for s in a.stats.split(",") if s.strip()]
    rep = compare(g, stats=stats, n_null=a.n_null, seed=a.seed, swaps_per_edge=a.swaps_per_edge,
                  keep_weights=not a.shuffle_weights, verbose=a.verbose)
    print(rep)
    if a.json:
        rep.to_json(a.json)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
