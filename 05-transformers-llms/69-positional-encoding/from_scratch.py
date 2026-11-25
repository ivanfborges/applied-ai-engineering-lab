"""Educational float64 positional mechanisms; no training or fused kernels."""

import json

import numpy as np


def _positions(values, name="positions"):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty 1D integer array.")
    if np.any(values < 0) or np.any(values > 2**53 - 1):
        raise ValueError(f"{name} must be nonnegative and exactly representable in float64.")
    return values.astype(np.float64)


def _frequencies(dim, base):
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise ValueError("Dimension must be a positive even integer.")
    if dim <= 0 or dim % 2:
        raise ValueError("Dimension must be a positive even integer.")
    if isinstance(base, (bool, np.bool_)) or not np.isscalar(base):
        raise ValueError("base must be finite and greater than one.")
    try:
        base = float(base)
    except (TypeError, ValueError) as exc:
        raise ValueError("base must be finite and greater than one.") from exc
    if not np.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than one.")
    return np.exp(-np.arange(0, dim, 2, dtype=np.float64) * np.log(base) / dim)


def _matrix(values, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{name} must be a nonempty 2D real array.")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values.")
    return result


def sinusoidal_encoding(positions, d_model, base=10000.0):
    """Return (tokens, d_model), interleaving sin then cos; even widths only."""
    position = _positions(positions)
    angles = position[:, None] * _frequencies(d_model, base)
    encoding = np.empty((position.size, d_model), dtype=np.float64)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding


def absolute_position_lookup(table, positions):
    """Read a supplied position table. This function does not learn its entries."""
    table = _matrix(table, "table")
    position = _positions(positions)
    if np.any(position >= table.shape[0]):
        raise ValueError("Position exceeds the supplied table; no extrapolation is defined.")
    return table[position.astype(np.int64)].copy()


def apply_rope(x, positions, base=10000.0):
    """Rotate adjacent pairs of a (tokens, even_dim) query or key matrix."""
    x = _matrix(x, "x")
    position = _positions(positions)
    if position.size != x.shape[0]:
        raise ValueError("One position is required per row of x.")
    angles = position[:, None] * _frequencies(x.shape[1], base)
    cosine, sine = np.cos(angles), np.sin(angles)
    rotated = np.empty_like(x)
    rotated[:, 0::2] = x[:, 0::2] * cosine - x[:, 1::2] * sine
    rotated[:, 1::2] = x[:, 0::2] * sine + x[:, 1::2] * cosine
    return rotated


def causal_alibi_bias(query_positions, key_positions, slopes):
    """Return (heads, queries, keys); future keys have -inf, self has zero bias.

    Slopes are supplied explicitly, not generated with the paper's head schedule.
    Positions may include offsets for rectangular prefix-cache attention.
    """
    query = _positions(query_positions, "query_positions")
    key = _positions(key_positions, "key_positions")
    raw = np.asarray(slopes)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ValueError("slopes must be a nonempty 1D array of positive finite values.")
    slopes = raw.astype(np.float64)
    if not np.isfinite(slopes).all() or np.any(slopes <= 0):
        raise ValueError("slopes must be positive and finite.")
    distance = query[:, None] - key[None, :]
    if np.any(np.all(distance < 0, axis=-1)):
        raise ValueError("Every query must have at least one visible key.")
    with np.errstate(over="ignore"):
        bias = -slopes[:, None, None] * np.maximum(distance, 0)
    if not np.isfinite(bias).all():
        raise ValueError("Slope times distance overflowed float64.")
    return np.where(distance[None, :, :] >= 0, bias, -np.inf)


def run_checks():
    """Measure algebraic properties on fixed synthetic vectors, not model quality."""
    rng = np.random.default_rng(69)
    dim, base = 16, 10000.0
    pe = sinusoidal_encoding(np.arange(12), dim, base)
    gram = pe @ pe.T
    gram_shift_error = float(np.max(np.abs(gram[1:, 1:] - gram[:-1, :-1])))
    q, k = rng.normal(size=(2, 1, dim))
    reference = float((apply_rope(q, [10]) @ apply_rope(k, [15]).T).item())
    shifts = []
    for position in [10, 100, 1000, 10000, 1000000]:
        dot = float((apply_rope(q, [position]) @ apply_rope(k, [position + 5]).T).item())
        shifts.append({"query_position": position, "key_position": position + 5,
                       "dot_product": dot, "absolute_error_vs_10_15": abs(dot - reference)})
    rotated = apply_rope(q, [10000])
    norm_error = float(np.max(np.abs(np.linalg.norm(q, axis=-1) - np.linalg.norm(rotated, axis=-1))))
    slopes = np.array([0.02, 0.1, 0.5, 1.0])
    bias = causal_alibi_bias([15], np.arange(16), slopes)[:, 0]
    weights = np.exp(bias - bias.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return {
        "configuration": {"seed": 69, "d_model": dim, "base": base, "dtype": "float64",
                          "alibi_keys": 16, "content_logits": "all zero; no trained model"},
        "sinusoidal_shifted_gram_max_error": gram_shift_error,
        "rope_norm_max_error": norm_error,
        "rope_common_shift": shifts,
        "alibi_flat_content": [
            {"slope": float(slope), "self_weight": float(weight[-1]),
             "expected_distance": float(weight @ np.arange(15, -1, -1))}
            for slope, weight in zip(slopes, weights)
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
