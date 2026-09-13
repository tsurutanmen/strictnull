# strictnull

Build the control before you compare.

A graph only looks special against the right control. Most papers compare a network against an
Erdős–Rényi graph with the same number of nodes and edges. That control cannot tell wiring from
degree: a heavy-tailed degree distribution on its own produces clustering, short paths, reciprocity,
and communities. `strictnull` puts the second control next to the first, verifies both, and tells you
how much of the "structure" was the degree sequence.

```
pip install git+https://github.com/tsurutanmen/strictnull
```

A PyPI release is planned; until then install from GitHub as above. Requires numpy and [igraph](https://python.igraph.org). `networkx` is optional (for `Graph.from_networkx`).

## Thirty seconds

```
$ strictnull tests/data/toy.csv --n-null 20

Graph(directed, weighted, 60 nodes, 400 edges)
controls: 20 draws each, strict null rewired 73.4% of edges

statistic              real          weak null        strict null   z(weak)  z(strict)  by degree
-------------------------------------------------------------------------------------------------
reciprocity          0.1250    0.1150+-0.0251      0.1153+-0.0171        +0.4       +0.6         2%
clustering           0.2835    0.2154+-0.0075      0.2817+-0.0091        +9.0       +0.2        97%
avg_path             1.8113    1.8391+-0.0098      1.8178+-0.0075        -2.8       -0.9        77%
assortativity       -0.1818   -0.0292+-0.0311     -0.1695+-0.0172        -4.9       -0.7        92%

warning: strict null rewired only 73% of edges; raise swaps_per_edge or accept that this graph has little room to rewire
```

Against the weak control this toy graph is nine standard deviations more clustered than chance.
Against the strict control it is 0.2. The degree sequence explains 97% of the gap. Nothing about the
wiring is special here; the paper that reported `z = 9` would have been reporting its degree
distribution.

## The three numbers

For each statistic the report gives:

| column | meaning |
|---|---|
| `weak null` | mean and sd over draws of an Erdős–Rényi graph with the same node and edge count (weights shuffled). The control most papers use. |
| `strict null` | mean and sd over draws of a degree-preserving rewiring: every node keeps its exact in-degree and out-degree, everything else is randomised. |
| `z(weak)`, `z(strict)` | distance of the real value from each control, in control standard deviations. |
| `by degree` | `(strict − weak) / (real − weak)`: the share of the weak-control gap that the degree sequence alone reproduces. Near 100% means the structure is the degree distribution. |

## Python

```python
import strictnull as sn

g = sn.Graph.from_csv("edges.csv", directed=True)      # header: source,target[,weight]
g = sn.Graph.from_edges(edges, weights, directed=True)  # (E, 2) int array
g = sn.Graph.from_adjacency(A, directed=True)
g = sn.Graph.from_networkx(G)

rep = sn.compare(g, stats=["clustering", "reciprocity", "avg_path", "assortativity"], n_null=20, seed=0)
print(rep)
rep.to_json("report.json")

for row in rep.rows:
    print(row.stat, row.z_strict, row.fraction_explained_by_degree)
```

Built-in statistics: `reciprocity`, `clustering`, `avg_clustering`, `avg_path`, `assortativity`,
`n_triangles`, `max_core`, `modularity_leiden`, `mean_weight`. Add your own:

```python
@sn.register("rich_club_100")
def rich_club_100(graph):
    ...
    return value

rep = sn.compare(g, stats=["rich_club_100", "clustering"])
# or pass the callable directly: sn.compare(g, stats=[rich_club_100])
```

### Swapping the graph inside a model

When the claim is not about a statistic but about a model that uses the graph (a connectome as a
fixed layer, a knowledge graph as a prior), draw nulls and retrain:

```python
for i, h in enumerate(sn.ensemble(g, n=5, seed=0)):        # independent degree-preserving draws
    sn.verify(g, h, strict=True)                            # raises if a degree moved
    A_null = h.g.get_adjacency(attribute="weight")          # or h.edges(), h.weights()
    acc = train_and_evaluate(A_null)
```

Compare the real graph's outcome against the mean and spread over draws, not against one draw.

## What the tool refuses or flags, and why

- **One draw is not a control.** `compare` raises on `n_null < 2`.
- **Every null is verified before use.** Exact degree match, no self-loops, no multi-edges, weight
  multiset preserved, and the fraction of edges actually rewired. Below 80% rewired, the report warns.
- **Invariant statistics are flagged.** If a statistic is identical, to the last digit, on the graph
  and on every null, the control never touched what it measures. That is not "no effect"; it is
  "nothing was tested". The row is marked `INVARIANT` and a warning explains which control it is
  invariant under. Functions of the degree sequence (max degree, degree variance) are invariant under
  the strict null. Functions of the edge count or weight multiset are invariant under both.

## Worked example: a connectome

The larval *Drosophila* connectome released by Winding et al. (2023): 2,952 neurons, 110,140
directed edges. Ten draws per control, 25 seconds.

```
statistic              real          weak null        strict null   z(weak)  z(strict)  by degree
reciprocity          0.2569    0.0126+-0.0004      0.0250+-0.0006      +677.1     +380.7         5%
clustering           0.2350    0.0252+-0.0000      0.0537+-0.0001     +5103.3    +1469.6        14%
avg_path             2.7466    2.1262+-0.0002      2.2596+-0.0008     +2840.9     +631.7        22%
assortativity        0.2365   -0.0012+-0.0020     -0.0138+-0.0021      +116.1     +119.7        -5%
```

Here the wiring is real: the degree sequence explains only 5–22% of each gap and the strict-control
z stays in the hundreds. That is the case where the strict control changes nothing about the
conclusion, and it still had to be run to know that.

The same connectome, used as a fixed recurrent layer in an MNIST classifier, gave the opposite
result: swapping the layer for a strict null moved accuracy by +0.002, within seed noise. Structure
exists, and the task does not use it. The two findings, and the code that produced them, are in the
[Tsuruta Lab research notes](https://tsurutalab.org/notes/).

## Claude Code skill

`skill/strictnull/SKILL.md` teaches Claude Code when to reach for this tool and how to write up the
result. Install it by copying the folder:

```
cp -r skill/strictnull ~/.claude/skills/strictnull
```

After that, a sentence like "this network is more clustered than random" in a session triggers the
comparison against both controls and a write-up that names the control.

## Method

The strict null is the directed configuration model conditioned on simple graphs, sampled by
double-edge swaps (Maslov & Sneppen 2002), `swaps_per_edge` swaps per edge (default 20). For
undirected graphs the same procedure preserves each node's degree. The weak null samples uniformly
among simple graphs with the given edge count. See Fosdick, Larremore, Nishimura & Ugander (2018),
*Configuring random graph models with fixed degree sequences*, SIAM Review 60(2), for the space of
choices this tool does not cover (multigraphs, self-loops, stub matching).

## License

MIT.
