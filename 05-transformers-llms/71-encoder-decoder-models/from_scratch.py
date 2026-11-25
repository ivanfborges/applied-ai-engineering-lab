"""Unbatched attention and target alignment; no training or language model."""

import json

import numpy as np


def visibility(query_length, key_length, *, causal=False, key_valid=None):
    """Return [query, key] visibility: True permits attention.

    Causality here requires equal lengths and positions starting at zero.
    Padding hides keys, not query outputs. Fully blocked rows are rejected.
    """
    for length in (query_length, key_length):
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or length < 1:
            raise ValueError("sequence lengths must be positive integers")
    if not isinstance(causal, (bool, np.bool_)):
        raise ValueError("causal must be Boolean")
    if causal and query_length != key_length:
        raise ValueError("causal visibility requires equal lengths; cache offsets are unsupported")
    allowed = np.ones((query_length, key_length), dtype=bool)
    if causal:
        allowed &= np.tri(query_length, dtype=bool)
    if key_valid is not None:
        valid = np.asarray(key_valid)
        if valid.dtype != np.bool_ or valid.shape != (key_length,):
            raise ValueError("key_valid must be a Boolean vector of key_length")
        allowed &= valid[None, :]
    if not allowed.any(axis=1).all():
        raise ValueError("each query must have at least one visible key")
    return allowed


def attention(q, k, v, allowed):
    """Single-head scaled dot-product attention on already projected Q/K/V."""
    arrays = []
    for name, value in (("q", q), ("k", k), ("v", v)):
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or raw.ndim != 2 or min(raw.shape) < 1:
            raise ValueError(f"{name} must be a nonempty real matrix")
        matrix = raw.astype(np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be finite")
        arrays.append(matrix)
    q, k, v = arrays
    if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q/K feature widths and K/V sequence lengths must match")
    allowed = np.asarray(allowed)
    if allowed.dtype != np.bool_ or allowed.shape != (len(q), len(k)):
        raise ValueError("allowed must be a Boolean [query_length, key_length] matrix")
    if not allowed.any(axis=1).all():
        raise ValueError("each query must have at least one visible key")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (q / np.sqrt(q.shape[1])) @ k.T
    if not np.isfinite(scores).all():
        raise ValueError("attention scores overflowed; rescale Q/K")
    scores = np.where(allowed, scores, -np.inf)
    with np.errstate(over="ignore"):
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore"):
        output = weights @ v
    if not np.isfinite(output).all():
        raise ValueError("attention output overflowed; rescale V")
    return output, weights


def shift_right(labels, start_id):
    """Return [start, y_0, ..., y_(m-2)] for one unpadded target vector.

    The start ID is explicit: T5 commonly uses its pad token to start decoding.
    Labels are unchanged; callers include EOS when their task requires it.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu" or (labels < 0).any():
        raise ValueError("labels must be a nonempty vector of nonnegative integer IDs")
    if isinstance(start_id, (bool, np.bool_)) or not isinstance(start_id, (int, np.integer)) or start_id < 0:
        raise ValueError("start_id must be a nonnegative integer")
    # Avoid narrowing a start ID to a small or unsigned label dtype.
    if start_id > np.iinfo(np.int64).max or (labels > np.iinfo(np.int64).max).any():
        raise ValueError("token IDs must fit int64")
    return np.concatenate((np.array([start_id], dtype=np.int64), labels[:-1].astype(np.int64)))


def demo():
    rng = np.random.default_rng(71)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    target_q = rng.normal(size=(3, 4))
    result = {"seed": 71, "numpy": np.__version__, "data": "synthetic projected Q/K/V"}
    for name, queries, mask in (
        ("bidirectional", q, visibility(5, 5)),
        ("causal", q, visibility(5, 5, causal=True)),
        ("cross", target_q, visibility(3, 5, key_valid=[True, True, True, True, False])),
    ):
        output, weights = attention(queries, k, v, mask)
        result[name] = {
            "output_shape": list(output.shape),
            "weight_shape": list(weights.shape),
            "row_sum_max_error": float(np.max(np.abs(weights.sum(axis=1) - 1))),
            "blocked_probability_max": float(weights[~mask].max()) if (~mask).any() else None,
        }
    labels = np.array([4, 7, 2])  # Synthetic IDs, with 2 standing for EOS.
    result["seq2seq_alignment"] = {"inputs": shift_right(labels, 0).tolist(), "labels": labels.tolist()}
    return result


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
