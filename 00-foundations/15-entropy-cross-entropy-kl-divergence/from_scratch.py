"""Educational information-theory and logit-loss utilities built with NumPy."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np


Reduction = Literal["none", "sum", "mean"]


def validate_distribution(values: np.ndarray, *, name: str = "distribution") -> np.ndarray:
    """Return a validated one-dimensional discrete probability distribution."""
    try:
        distribution = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric probabilities.") from error
    if distribution.ndim != 1 or distribution.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(distribution)):
        raise ValueError(f"{name} must contain only finite probabilities.")
    if np.any(distribution < 0.0):
        raise ValueError(f"{name} cannot contain negative probabilities.")
    if not np.isclose(distribution.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} probabilities must sum to 1.")
    return distribution


def _validate_pair(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = validate_distribution(p, name="p")
    q = validate_distribution(q, name="q")
    if p.shape != q.shape:
        raise ValueError("p and q must describe the same number of outcomes.")
    return p, q


def entropy(p: np.ndarray) -> float:
    """Compute discrete Shannon entropy H(P) in nats."""
    p = validate_distribution(p, name="p")
    positive = p > 0.0
    return float(-np.sum(p[positive] * np.log(p[positive])))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Compute H(P, Q) in nats, returning infinity for a support mismatch."""
    p, q = _validate_pair(p, q)
    positive = p > 0.0
    if np.any(q[positive] == 0.0):
        return math.inf
    return float(-np.sum(p[positive] * np.log(q[positive])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute D_KL(P || Q) in nats, including its infinite case."""
    p, q = _validate_pair(p, q)
    positive = p > 0.0
    if np.any(q[positive] == 0.0):
        return math.inf
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute log-softmax without exponentiating the original logits."""
    try:
        array = np.asarray(logits, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("logits must contain numeric values.") from error
    if array.ndim == 0 or array.size == 0:
        raise ValueError("logits must be a non-empty array with at least one axis.")
    if np.any(~np.isfinite(array)):
        raise ValueError("logits must contain only finite values.")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer.")
    if not -array.ndim <= axis < array.ndim:
        raise ValueError("axis is outside the logits dimensions.")
    shifted = array - np.max(array, axis=axis, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_normalizer


def softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax along one axis."""
    return np.exp(log_softmax(logits, axis=axis))


def categorical_nll_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    reduction: Reduction = "mean",
) -> float | np.ndarray:
    """Compute categorical NLL from logits, optionally masking positions."""
    try:
        logits_array = np.asarray(logits, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("logits must contain numeric values.") from error
    if logits_array.ndim < 2 or logits_array.shape[-1] < 2:
        raise ValueError("logits must have shape (..., classes) with at least 2 classes.")
    if np.any(~np.isfinite(logits_array)):
        raise ValueError("logits must contain only finite values.")
    targets_array = np.asarray(targets)
    if targets_array.dtype == np.bool_ or not np.issubdtype(
        targets_array.dtype, np.integer
    ):
        raise TypeError("targets must contain integer class indices.")
    if targets_array.shape != logits_array.shape[:-1]:
        raise ValueError("targets must match every non-class logits dimension.")
    if np.any((targets_array < 0) | (targets_array >= logits_array.shape[-1])):
        raise ValueError("targets contain a class index outside the logits range.")
    if mask is None:
        valid = np.ones(targets_array.shape, dtype=bool)
    else:
        valid = np.asarray(mask)
        if valid.dtype != np.bool_:
            raise TypeError("mask must contain Boolean values.")
        if valid.shape != targets_array.shape:
            raise ValueError("mask must have the same shape as targets.")
    if not np.any(valid):
        raise ValueError("mask must retain at least one target position.")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")
    log_probabilities = log_softmax(logits_array)
    selected = np.take_along_axis(
        log_probabilities, targets_array[..., np.newaxis], axis=-1
    )[..., 0]
    losses = np.where(valid, -selected, 0.0)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses))
    return float(np.sum(losses) / np.count_nonzero(valid))


def main() -> None:
    """Print the core identity and a stable categorical-loss calculation."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.6, 0.3, 0.1])
    h_p = entropy(p)
    h_pq = cross_entropy(p, q)
    kl_pq = kl_divergence(p, q)
    logits = np.array([[1_000.0, 999.0, 998.0]])
    target = np.array([0])
    print("Information is measured in nats (natural logarithms).")
    print(f"H(P):       {h_p:.6f}")
    print(f"H(P, Q):    {h_pq:.6f}")
    print(f"KL(P || Q): {kl_pq:.6f}")
    print(f"Identity holds: {np.isclose(h_pq, h_p + kl_pq)}")
    loss = categorical_nll_from_logits(logits, target)
    print(f"Stable NLL from large logits: {loss:.6f}")


if __name__ == "__main__":
    main()
