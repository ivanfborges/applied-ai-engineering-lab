"""Educational, unbatched single-head attention in NumPy; no training or dropout."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a nonempty 2D matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numbers")
    return array


def softmax(scores: ArrayLike, allowed: ArrayLike | None = None) -> NDArray[np.float64]:
    """Normalize each row across keys. A boolean True permits an interaction.

    Masks may broadcast to the 2D score shape. Fully blocked rows are errors,
    because there is no probability distribution over an empty set of keys.
    """
    scores = _matrix(scores, "scores")
    if allowed is None:
        mask = np.ones(scores.shape, dtype=bool)
    else:
        mask = np.asarray(allowed)
        if mask.dtype != np.bool_:
            raise ValueError("allowed must be a boolean mask (True means permitted)")
        try:
            mask = np.broadcast_to(mask, scores.shape)
        except ValueError as error:
            raise ValueError("allowed must broadcast to the score shape") from error
    if not mask.any(axis=1).all():
        raise ValueError("each query must have at least one permitted key")
    masked = np.where(mask, scores, -np.inf)
    # Shift by the permitted maximum, so no exponential has a positive argument.
    # Extreme finite differences may underflow to zero, which is acceptable here.
    with np.errstate(over="ignore", under="ignore"):
        exponentials = np.exp(masked - masked.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def scaled_dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    allowed: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (output, weights) for Q:(n,d_k), K:(m,d_k), V:(m,d_v).

    Supports self- or cross-attention; rejects nonfinite intermediates. It does
    not support batch/head axes, additive biases, autograd, or implicit causality.
    """
    query, key, value = (
        _matrix(array, name)
        for array, name in ((query, "query"), (key, "key"), (value, "value"))
    )
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key must share d_k")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value must share the number of positions")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (query @ key.T) / np.sqrt(query.shape[1])
    if not np.isfinite(scores).all():
        raise ValueError("query-key dot products overflowed; reduce input magnitudes")
    weights = softmax(scores, allowed)
    with np.errstate(over="ignore", invalid="ignore"):
        output = weights @ value
    if not np.isfinite(output).all():
        raise ValueError("weighted sum produced nonfinite output")
    return output, weights


if __name__ == "__main__":
    q = np.array([[1.0, 0.0]])
    k = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    v = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    result, coefficients = scaled_dot_product_attention(q, k, v)
    print("Synthetic weights:", coefficients)
    print("Weighted output:", result)
