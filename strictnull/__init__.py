"""strictnull: build the control before you compare.

A graph only looks special against the right control.  Two controls are
provided, deliberately ordered from weak to strict:

    weak_null    same node count and edge count, nothing else preserved
    strict_null  every node keeps its exact degree (in and out, if directed)

If an apparent property of a graph survives only against the weak control,
the property belongs to the degree sequence, not to the wiring.

    >>> import strictnull as sn
    >>> g = sn.Graph.from_edges(edges, directed=True)
    >>> print(sn.compare(g, n_null=20))
"""

from .graph import Graph
from .nulls import weak_null, strict_null, ensemble, verify
from .stats import STATS, register
from .compare import compare, Report

__version__ = "0.1.0"
__all__ = [
    "Graph", "weak_null", "strict_null", "ensemble", "verify",
    "STATS", "register", "compare", "Report",
]
