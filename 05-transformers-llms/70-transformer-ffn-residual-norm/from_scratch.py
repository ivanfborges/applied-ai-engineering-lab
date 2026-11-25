"""Educational float64 FFN and last-axis LayerNorm; no training or attention."""

from __future__ import annotations

import numpy as np


def _array(value, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _features(x) -> np.ndarray:
    x = _array(x, "x")
    if x.ndim < 1:
        raise ValueError("x must have a hidden-feature axis")
    return x


def _eps(eps: float) -> float:
    if isinstance(eps, (bool, np.bool_)) or not np.isscalar(eps):
        raise ValueError("eps must be a positive finite real scalar")
    if not isinstance(eps, (int, float, np.integer, np.floating)):
        raise ValueError("eps must be a positive finite real scalar")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite real scalar")
    return float(eps)


def layer_norm(x, gamma, beta, eps: float = 1e-5) -> np.ndarray:
    """Normalize only the final axis using population variance (ddof=0)."""
    x = _features(x)
    gamma, beta = _array(gamma, "gamma"), _array(beta, "beta")
    if gamma.shape != (x.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("gamma and beta must have shape (hidden_features,)")
    eps = _eps(eps)
    centered = x - x.mean(axis=-1, keepdims=True)
    variance = np.mean(centered**2, axis=-1, keepdims=True)
    return gamma * centered / np.sqrt(variance + eps) + beta


def layer_norm_vjp(x, gamma, upstream, eps: float = 1e-5) -> np.ndarray:
    """Return J_LN(x).T @ upstream, including the stabilizing epsilon."""
    x = _features(x)
    gamma, upstream = _array(gamma, "gamma"), _array(upstream, "upstream")
    if gamma.shape != (x.shape[-1],) or upstream.shape != x.shape:
        raise ValueError("gamma must match features and upstream must match x")
    eps = _eps(eps)
    centered = x - x.mean(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + eps)
    normalized = centered * inv_std
    g = upstream * gamma
    return inv_std * (
        g - g.mean(axis=-1, keepdims=True)
        - normalized * np.mean(g * normalized, axis=-1, keepdims=True)
    )


def gelu(x) -> np.ndarray:
    """Tanh approximation; compare against PyTorch approximate='tanh'."""
    x = _array(x, "x")
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def feed_forward(x, w1, b1, w2, b2) -> np.ndarray:
    """Apply the same two-projection GELU FFN independently at each position."""
    x = _features(x)
    w1, b1 = _array(w1, "w1"), _array(b1, "b1")
    w2, b2 = _array(w2, "w2"), _array(b2, "b2")
    d = x.shape[-1]
    if w1.ndim != 2 or w1.shape[0] != d:
        raise ValueError("w1 must have shape (d_model, d_ff)")
    h = w1.shape[1]
    if b1.shape != (h,) or w2.shape != (h, d) or b2.shape != (d,):
        raise ValueError("FFN parameters must have shapes (d,h), (h,), (h,d), (d,)")
    return gelu(x @ w1 + b1) @ w2 + b2


def residual_sublayer(x, transform, gamma, beta, order: str = "pre", eps: float = 1e-5):
    """Compose one FFN-like branch with an exact-shape residual addition."""
    x = _features(x)
    if order not in ("pre", "post"):
        raise ValueError("order must be 'pre' or 'post'")
    branch_input = layer_norm(x, gamma, beta, eps) if order == "pre" else x
    update = _array(transform(branch_input), "transform output")
    if update.shape != x.shape:
        raise ValueError("transform output must have exactly the same shape as x")
    summed = x + update
    return summed if order == "pre" else layer_norm(summed, gamma, beta, eps)


def main() -> None:
    rng = np.random.default_rng(70)
    x = rng.normal(size=(2, 3, 8))  # Synthetic token features, not text embeddings.
    gamma, beta = np.ones(8), np.zeros(8)
    w1, w2 = rng.normal(size=(8, 32)) / np.sqrt(8), rng.normal(size=(32, 8)) / np.sqrt(32)
    branch = lambda z: feed_forward(z, w1, np.zeros(32), w2, np.zeros(8))
    normalized = layer_norm(x, gamma, beta)
    pre = residual_sublayer(x, branch, gamma, beta, "pre")
    post = residual_sublayer(x, branch, gamma, beta, "post")
    print("Synthetic seed=70; forward-only educational operations; no training")
    print(f"input/FFN/residual shapes: {x.shape}, {branch(x).shape}, {pre.shape}")
    print(f"standardized mean max abs: {np.abs(normalized.mean(axis=-1)).max():.6e}")
    print(f"standardized variance range: {normalized.var(axis=-1).min():.6f}..{normalized.var(axis=-1).max():.6f}")
    print(f"pre/post output RMS: {np.sqrt(np.mean(pre**2)):.6f}, {np.sqrt(np.mean(post**2)):.6f}")


if __name__ == "__main__":
    main()
