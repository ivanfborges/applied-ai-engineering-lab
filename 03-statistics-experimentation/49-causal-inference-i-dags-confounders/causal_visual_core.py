"""Shared synthetic SCM, graphical explanations and numerical visual evidence."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd

from example import study_graph
from from_scratch import fwl_coefficient, standardized_difference


MODULES = ("dag", "confounding", "collider", "mediation", "intervention",
           "geometry", "adjustment", "sample-size")
TITLES = ("Understand the DAG", "Observe confounding", "Understand colliders",
          "Explore mediation", "Observation versus intervention", "Causal geometry in 3D",
          "Compare regression adjustments", "Bias versus sample size")
COLORS = {"A": "#2563a6", "Y": "#087e64", "C": "#b66a0b", "M": "#7954a1",
          "S": "#be4350", "U": "#596675"}
ROLES = {"A": "Treatment", "Y": "Outcome", "C": "Confounder", "M": "Mediator",
         "S": "Collider", "U": "Latent (simulator only)"}
POSITIONS = {"A": (-1.7, 0), "Y": (1.7, 0), "C": (0, 1.5),
             "M": (0, .6), "S": (-1.1, -1.25), "U": (1.1, -1.25)}
ADJUSTMENTS = {"None": (), "C": ("C",), "C + M": ("C", "M"), "C + S": ("C", "S")}


def random_tape(n: int, seed: int) -> dict[str, np.ndarray]:
    """NumPy exogenous draws, shared across parameter changes (common random numbers)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 100:
        raise ValueError("n must be an integer >= 100")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    c = rng.binomial(1, .5, n)
    a = rng.binomial(1, .2 + .6 * c)
    return {"C": c, "A": a, **{key: rng.normal(size=n) for key in ("U", "eM", "eS", "eY")}}


def model(tape: dict[str, np.ndarray], *, alpha=.7, beta=1., confounding=1.5,
          noise=.5, n: int | None = None) -> pd.DataFrame:
    """Regenerate all descendants using the declared equations; S never causes Y."""
    if not np.isfinite([alpha, beta, confounding, noise]).all() or noise < 0:
        raise ValueError("coefficients must be finite and collider noise nonnegative")
    size = len(tape["A"])
    if n is None:
        n = size
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 100 <= n <= size:
        raise ValueError("n must be an integer between 100 and the tape length")
    if set(tape) != {"A", "C", "U", "eM", "eS", "eY"}:
        raise ValueError("tape must contain A, C, U, eM, eS and eY")
    if any(np.asarray(v).ndim != 1 or len(v) != size or not np.isfinite(v).all()
           for v in tape.values()):
        raise ValueError("tape arrays must be finite, 1D and equally sized")
    if any(not np.isin(tape[k], [0, 1]).all() for k in ("A", "C")):
        raise ValueError("A and C must be binary")
    d = pd.DataFrame({k: v[:n] for k, v in tape.items()})
    d["M"] = alpha * d.A + d.eM
    d["S"] = d.A + d.U + noise * d.eS
    d["Y0"] = beta * d.eM + confounding * d.C + 1.2 * d.U + d.eY
    d["Y1"] = d.Y0 + 1.3 + alpha * beta
    d["Y"] = 1.3 * d.A + beta * d.M + confounding * d.C + 1.2 * d.U + d.eY
    return d


def population_values(alpha=.7, beta=1., confounding=1.5, noise=.5) -> dict[str, float]:
    if not np.isfinite([alpha, beta, confounding, noise]).all() or noise < 0:
        raise ValueError("invalid population parameters")
    total = 1.3 + alpha * beta
    return {"total": total, "direct": 1.3, "indirect": alpha * beta,
            "naive": total + .6 * confounding,
            "collider": total - 1.2 / (1 + noise ** 2)}


def slope(d: pd.DataFrame, controls=()) -> float:
    if len(d) < len(controls) + 4:
        raise ValueError("insufficient observations")
    x = np.column_stack([np.ones(len(d)), d.A, *(d[k] for k in controls)])
    coefficients, _, rank, _ = np.linalg.lstsq(x, d.Y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("rank-deficient design")
    return float(coefficients[1])


def residuals(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.column_stack([np.ones(len(d)), d.C])
    ra = d.A.to_numpy() - x @ np.linalg.lstsq(x, d.A, rcond=None)[0]
    ry = d.Y.to_numpy() - x @ np.linalg.lstsq(x, d.Y, rcond=None)[0]
    fwl = fwl_coefficient(d.A, d.Y, d[["C"]].to_numpy())
    np.testing.assert_allclose(fwl, slope(d, ("C",)), atol=1e-10)
    return ra, ry, fwl


def summary(d: pd.DataFrame) -> dict:
    estimates = {name: slope(d, controls) for name, controls in ADJUSTMENTS.items()}
    estimates["Standardized C"] = standardized_difference(d.A, d.Y, d.C)
    strata = [{"C": c, "n": int((d.C == c).sum()),
               "weight": float((d.C == c).mean()),
               "difference": float(d.loc[(d.C == c) & (d.A == 1), "Y"].mean()
                                   - d.loc[(d.C == c) & (d.A == 0), "Y"].mean())}
              for c in (0, 1)]
    means = [float(d.loc[d.A == a, "Y"].mean()) for a in (0, 1)]
    _, _, fwl = residuals(d)
    return {"estimates": estimates, "strata": strata, "means": means, "fwl": fwl,
            "p_c1_by_a": [float(d.loc[d.A == a, "C"].mean()) for a in (0, 1)]}


def collider_selection(d: pd.DataFrame, center=0.5, width=.35, mode="interval") -> dict:
    if not np.isfinite([center, width]).all() or width <= 0 or mode not in ("interval", "high"):
        raise ValueError("invalid selection rule")
    mask = abs(d.S - center) <= width if mode == "interval" else d.S >= center
    selected = d.loc[mask]

    def delta_u(frame):
        if frame.A.nunique() < 2:
            return None
        return float(frame.loc[frame.A == 1, "U"].mean() - frame.loc[frame.A == 0, "U"].mean())

    def safe(frame, controls):
        try:
            return slope(frame, controls)
        except ValueError:
            return None

    return {"indices": np.flatnonzero(mask).tolist(), "n": len(selected),
            "delta_u_full": delta_u(d), "delta_u_selected": delta_u(selected),
            "full_C": safe(d, ("C",)), "full_CS": safe(d, ("C", "S")),
            "selected_C": safe(selected, ("C",)), "selected_CS": safe(selected, ("C", "S"))}


def graph_payload() -> dict:
    graph = study_graph()
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(sorted(graph.nodes))
    nx_graph.add_edges_from(sorted(graph.edges))
    if not nx.is_directed_acyclic_graph(nx_graph):
        raise ValueError("study graph must be acyclic")
    cases = {}
    for label, controls in ADJUSTMENTS.items():
        paths = []
        for path in sorted(graph.paths("A", "Y")):
            blockers, openers = [], []
            for left, middle, right in zip(path, path[1:], path[2:]):
                collider = (left, middle) in graph.edges and (right, middle) in graph.edges
                if collider:
                    activated = set(controls) & ({middle} | graph.descendants(middle))
                    if activated:
                        openers.append(f"collider {middle} activated by {', '.join(sorted(activated))}")
                    else:
                        blockers.append(f"collider {middle} and its descendants are unconditioned")
                elif middle in controls:
                    blockers.append(f"conditioned noncollider {middle}")
            causal = all((u, v) in graph.edges for u, v in zip(path, path[1:]))
            active = not blockers
            assert active == graph._active(path, set(controls))
            text = path[0]
            for u, v in zip(path, path[1:]):
                text += (" → " if (u, v) in graph.edges else " ← ") + v
            paths.append({"nodes": path, "label": text, "causal": causal,
                          "active": active, "reason": "; ".join(blockers or openers)
                          or "No internal blocking node"})
        cases[label] = {"controls": controls, "paths": paths,
                        "backdoor": graph.satisfies_backdoor("A", "Y", controls),
                        "preserves_total": label in ("None", "C")}
    return {"nodes": sorted(nx_graph.nodes), "edges": sorted(nx_graph.edges),
            "positions": POSITIONS, "roles": ROLES, "colors": COLORS, "cases": cases}


def validate_graph_algorithm() -> int:
    """Independently compare path enumeration with NetworkX over all conditioning sets."""
    graph = study_graph()
    reference = nx.DiGraph(list(graph.edges))
    comparisons = 0
    for x, y in itertools.combinations(sorted(graph.nodes), 2):
        others = sorted(graph.nodes - {x, y})
        for bits in itertools.product((False, True), repeat=len(others)):
            z = {node for node, chosen in zip(others, bits) if chosen}
            assert graph.is_d_separated(x, y, z) == nx.is_d_separator(reference, {x}, {y}, z)
            comparisons += 1
    return comparisons


def bias_study(seed: int, sizes=(200, 600, 1800, 5400), repetitions=24) -> list[dict]:
    """Independent samples per size/repetition; quantiles are sampling ranges, not CIs."""
    if repetitions < 8 or repetitions > 200 or any(n < 100 for n in sizes):
        raise ValueError("require 8..200 repetitions and sample sizes >= 100")
    records = []
    for i, n in enumerate(sizes):
        draws = {name: [] for name in ("None", "C", "C + S")}
        for r in range(repetitions):
            d = model(random_tape(int(n), seed + 100_000 + i * 1000 + r))
            for name in draws:
                draws[name].append(slope(d, ADJUSTMENTS[name]))
        for name, values in draws.items():
            v = np.array(values)
            records.append({"n": int(n), "estimator": name, "mean": float(v.mean()),
                            "sd": float(v.std(ddof=1)), "mcse": float(v.std(ddof=1) / np.sqrt(repetitions)),
                            "low": float(np.quantile(v, .1)), "high": float(np.quantile(v, .9))})
    return records
