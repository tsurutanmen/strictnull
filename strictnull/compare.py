"""Compare a graph against the weak and the strict control, and say what survived."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .graph import Graph
from .nulls import ensemble, verify
from .stats import STATS, DEFAULT

StatSpec = Union[str, Callable[[Graph], float]]


@dataclass
class Row:
    stat: str
    real: float
    weak_mean: float
    weak_sd: float
    strict_mean: float
    strict_sd: float
    z_weak: float
    z_strict: float
    fraction_explained_by_degree: float
    invariant_under_strict: bool
    invariant_under_weak: bool


@dataclass
class Report:
    graph: str
    n_null: int
    swaps_per_edge: int
    fraction_rewired: float
    rows: List[Row] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "graph": self.graph, "n_null": self.n_null, "swaps_per_edge": self.swaps_per_edge,
            "fraction_rewired": self.fraction_rewired,
            "rows": [r.__dict__ for r in self.rows], "warnings": list(self.warnings),
        }

    def to_json(self, path: Optional[str] = None, indent: int = 1) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def __str__(self) -> str:
        lines = [f"{self.graph}", f"controls: {self.n_null} draws each, strict null rewired {self.fraction_rewired:.1%} of edges", ""]
        head = f"{'statistic':16s} {'real':>10s} {'weak null':>18s} {'strict null':>18s} {'z(weak)':>9s} {'z(strict)':>10s} {'by degree':>10s}"
        lines.append(head)
        lines.append("-" * len(head))
        for r in self.rows:
            fe = "-" if math.isnan(r.fraction_explained_by_degree) else f"{r.fraction_explained_by_degree:9.0%}"
            flag = "  INVARIANT" if r.invariant_under_strict else ""
            lines.append(
                f"{r.stat:16s} {r.real:10.4f} {r.weak_mean:9.4f}+-{r.weak_sd:<7.4f} "
                f"{r.strict_mean:9.4f}+-{r.strict_sd:<7.4f} {_fz(r.z_weak):>9s} {_fz(r.z_strict):>10s} {fe:>10s}{flag}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"warning: {w}")
        return "\n".join(lines)


def _fz(z: float) -> str:
    if math.isnan(z):
        return "nan"
    if math.isinf(z):
        return "inf" if z > 0 else "-inf"
    return f"{z:+.1f}"


def _z(real: float, mean: float, sd: float) -> float:
    if sd == 0:
        return 0.0 if real == mean else math.copysign(math.inf, real - mean)
    return (real - mean) / sd


def _resolve(stats: Sequence[StatSpec]) -> Dict[str, Callable[[Graph], float]]:
    out = {}
    for s in stats:
        if callable(s):
            out[getattr(s, "__name__", f"stat{len(out)}")] = s
        elif s in STATS:
            out[s] = STATS[s]
        else:
            raise KeyError(f"unknown statistic {s!r}; known: {sorted(STATS)}")
    return out


def compare(graph: Graph, stats: Sequence[StatSpec] = DEFAULT, n_null: int = 20, seed: int = 0,
            swaps_per_edge: int = 20, keep_weights: bool = True, verbose: bool = False) -> Report:
    """Measure ``stats`` on the graph and on ``n_null`` weak and strict nulls.

    ``fraction_explained_by_degree`` is (strict_mean - weak_mean) / (real - weak_mean):
    the share of the gap between the graph and the weak control that the
    degree sequence alone reproduces.  Near 1 means the "structure" is the
    degree distribution.  ``invariant_under_strict`` flags a statistic that is
    identical, to the last digit, on the graph and on every strict null: that
    control did not touch what the statistic measures, so the comparison is
    vacuous for it.
    """
    if n_null < 2:
        raise ValueError("n_null must be at least 2; one draw is not a control")
    fns = _resolve(stats)
    real = {k: float(f(graph)) for k, f in fns.items()}

    weak_vals = {k: [] for k in fns}
    strict_vals = {k: [] for k in fns}
    rewired = []
    for i, h in enumerate(ensemble(graph, n=n_null, seed=seed, kind="weak")):
        rep = verify(graph, h, strict=False)
        if not rep["n_edges_match"]:
            raise AssertionError("weak null has the wrong edge count")
        for k, f in fns.items():
            weak_vals[k].append(float(f(h)))
        if verbose:
            print(f"  weak null {i + 1}/{n_null}")
    for i, h in enumerate(ensemble(graph, n=n_null, seed=seed + 7, kind="strict",
                                   swaps_per_edge=swaps_per_edge, keep_weights=keep_weights)):
        rep = verify(graph, h, strict=True)
        rewired.append(rep["fraction_rewired"])
        for k, f in fns.items():
            strict_vals[k].append(float(f(h)))
        if verbose:
            print(f"  strict null {i + 1}/{n_null}  rewired {rep['fraction_rewired']:.1%}")

    report = Report(graph=repr(graph), n_null=n_null, swaps_per_edge=swaps_per_edge,
                    fraction_rewired=float(np.mean(rewired)))
    if report.fraction_rewired < 0.8:
        report.warnings.append(
            f"strict null rewired only {report.fraction_rewired:.0%} of edges; raise swaps_per_edge "
            f"or accept that this graph has little room to rewire (very dense or very constrained degrees)")
    if graph.dropped_self_loops or graph.merged_duplicates:
        report.warnings.append(
            f"graph construction dropped {graph.dropped_self_loops} self-loops and merged "
            f"{graph.merged_duplicates} duplicate edges; nulls are defined on the simple graph")

    for k in fns:
        w = np.asarray(weak_vals[k]); s = np.asarray(strict_vals[k])
        wm, wsd, sm, ssd = float(w.mean()), float(w.std()), float(s.mean()), float(s.std())
        gap = real[k] - wm
        frac = (sm - wm) / gap if gap != 0 else float("nan")
        inv_s = bool(np.all(s == real[k]))
        inv_w = bool(np.all(w == real[k]))
        row = Row(stat=k, real=real[k], weak_mean=wm, weak_sd=wsd, strict_mean=sm, strict_sd=ssd,
                  z_weak=_z(real[k], wm, wsd), z_strict=_z(real[k], sm, ssd),
                  fraction_explained_by_degree=float(frac),
                  invariant_under_strict=inv_s, invariant_under_weak=inv_w)
        report.rows.append(row)
        if inv_s and inv_w:
            report.warnings.append(
                f"{k!r} is identical on the graph and on every null: this statistic is invariant under both "
                f"controls, so the comparison measures nothing for it")
        elif inv_s:
            report.warnings.append(
                f"{k!r} is identical on the graph and on every strict null: the degree-preserving control "
                f"does not touch what it measures (it is a function of the degree sequence)")
    return report
