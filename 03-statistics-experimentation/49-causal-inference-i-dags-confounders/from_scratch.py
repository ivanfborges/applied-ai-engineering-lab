"""Tiny-DAG path rules and linear adjustment for teaching, not production use."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class DAG:
    """Validate a DAG and enumerate simple paths; cost can grow exponentially."""

    def __init__(self, nodes: Iterable[str], edges: Iterable[tuple[str, str]]):
        node_list = tuple(nodes)
        if not node_list or any(not isinstance(n, str) or not n for n in node_list):
            raise ValueError("nodes must be nonempty strings")
        if len(set(node_list)) != len(node_list):
            raise ValueError("nodes must be unique")
        self.nodes = frozenset(node_list)
        self.edges = frozenset(tuple(edge) for edge in edges)
        if any(len(e) != 2 or not set(e) <= self.nodes or e[0] == e[1]
               for e in self.edges):
            raise ValueError("edges must connect two distinct declared nodes")
        self.children = {n: {v for u, v in self.edges if u == n} for n in self.nodes}
        self.neighbors = {n: {v for u, v in self.edges if u == n}
                          | {u for u, v in self.edges if v == n} for n in self.nodes}
        remaining = set(self.nodes)
        while remaining:
            roots = {n for n in remaining
                     if not any(v == n and u in remaining for u, v in self.edges)}
            if not roots:
                raise ValueError("directed cycles are not allowed")
            remaining -= roots

    def _query(self, x: str, y: str, conditioned: Iterable[str]) -> set[str]:
        z = set(conditioned)
        if x not in self.nodes or y not in self.nodes or not z <= self.nodes:
            raise ValueError("query contains unknown nodes")
        if x == y or {x, y} & z:
            raise ValueError("endpoints must be distinct and outside the conditioning set")
        return z

    def descendants(self, node: str) -> set[str]:
        """Return strict descendants (the node itself is excluded)."""
        if node not in self.nodes:
            raise ValueError("unknown node")
        found: set[str] = set()
        pending = list(self.children[node])
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.children[current])
        return found

    def paths(self, x: str, y: str):
        """Yield simple undirected paths, retaining edge directions in the DAG."""
        self._query(x, y, ())
        pending = [(x,)]
        while pending:
            path = pending.pop()
            if path[-1] == y:
                yield path
            else:
                pending.extend(path + (n,) for n in sorted(self.neighbors[path[-1]])
                               if n not in path)

    def _active(self, path: tuple[str, ...], z: set[str]) -> bool:
        for left, middle, right in zip(path, path[1:], path[2:]):
            collider = (left, middle) in self.edges and (right, middle) in self.edges
            if collider:
                if middle not in z and not self.descendants(middle) & z:
                    return False
            elif middle in z:
                return False
        return True

    def is_d_separated(self, x: str, y: str, conditioned: Iterable[str] = ()) -> bool:
        """Check all paths using noncollider blocking and collider activation."""
        z = self._query(x, y, conditioned)
        return not any(self._active(path, z) for path in self.paths(x, y))

    def satisfies_backdoor(self, treatment: str, outcome: str,
                           adjusted: Iterable[str] = ()) -> bool:
        """Original sufficient backdoor criterion for a single treatment/outcome."""
        z = self._query(treatment, outcome, adjusted)
        if z & self.descendants(treatment):
            return False
        # Only paths whose first edge points INTO treatment count as backdoor paths.
        return not any(self._active(path, z) for path in self.paths(treatment, outcome)
                       if (path[1], treatment) in self.edges)


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1D array with at least two values")
    return array


def fwl_coefficient(treatment, outcome, controls) -> float:
    """OLS treatment slope via FWL, including an intercept in both projections.

    Controls must be an n-by-k matrix (n-by-0 for no controls). This function
    rejects rank deficiency rather than silently choosing a nonunique slope.
    It provides a point estimate, not causal identification or standard errors.
    """
    a, y = _vector(treatment, "treatment"), _vector(outcome, "outcome")
    c = np.asarray(controls, dtype=float)
    if c.ndim != 2 or c.shape[0] != a.size or y.size != a.size or not np.isfinite(c).all():
        raise ValueError("controls must be finite n-by-k and all row counts must match")
    design = np.column_stack([np.ones(a.size), c])
    full = np.column_stack([design, a])
    if full.shape[0] <= full.shape[1] or np.linalg.matrix_rank(full) < full.shape[1]:
        raise ValueError("design must have full column rank and positive residual degrees of freedom")
    residual_a = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(residual_a @ residual_y / (residual_a @ residual_a))


def standardized_difference(treatment, outcome, strata) -> float:
    """Empirical g-formula for one discrete stratum variable and binary treatment.

    Weight by the entire sample's stratum frequencies, not arm frequencies.
    Both arms must occur in every observed stratum; this finite-sample check
    cannot establish population positivity. Continuous controls are unsuitable.
    """
    a, y, c = (_vector(treatment, "treatment"), _vector(outcome, "outcome"),
               _vector(strata, "strata"))
    if a.size != y.size or a.size != c.size or not np.isin(a, [0, 1]).all():
        raise ValueError("row counts must match and treatment must be binary 0/1")
    result = 0.0
    for value in np.unique(c):
        in_stratum = c == value
        treated, untreated = in_stratum & (a == 1), in_stratum & (a == 0)
        if not treated.any() or not untreated.any():
            raise ValueError("both treatment arms are required in every observed stratum")
        result += in_stratum.mean() * (y[treated].mean() - y[untreated].mean())
    return float(result)
