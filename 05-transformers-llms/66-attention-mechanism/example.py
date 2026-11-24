"""Deterministic synthetic attention controls; prints results, writes no assets."""

from __future__ import annotations

import argparse
import platform

import numpy as np

from from_scratch import scaled_dot_product_attention, softmax


def entropy(weights: np.ndarray) -> float:
    """Mean row entropy in nats, with the convention 0 log(0) = 0."""
    positive = weights > 0
    terms = np.zeros_like(weights)
    terms[positive] = weights[positive] * np.log(weights[positive])
    return float(-terms.sum(axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", action="store_true", help="compare CPU float64 SDPA")
    args = parser.parse_args()
    np.set_printoptions(precision=6, suppress=True)
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; CPU float64")
    print("Synthetic fixtures only; no learned representations or task benchmark.")
    q = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    k = q.copy()
    v = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    causal = np.tri(3, dtype=bool)
    output, weights = scaled_dot_product_attention(q, k, v)
    causal_output, causal_weights = scaled_dot_product_attention(q, k, v, causal)
    print("Full weights:\n", weights)
    print("Full output:\n", output)
    print("Causal weights:\n", causal_weights)
    print("Causal output:\n", causal_output)

    changed_output, changed_weights = scaled_dot_product_attention(q, k, v + [3, -2])
    print("Value-only change: max weight delta =", np.max(abs(weights - changed_weights)))
    print("Value-only change: output shift =", changed_output[0] - output[0])
    future_k, future_v = k.copy(), v.copy()
    future_k[-1], future_v[-1] = [20, -20], [100, -100]
    changed_full, _ = scaled_dot_product_attention(q, future_k, future_v)
    changed_causal, _ = scaled_dot_product_attention(q, future_k, future_v, causal)
    print("Future K/V change: max earlier full-output delta =",
          np.max(abs(output[:2] - changed_full[:2])))
    print("Future K/V change: max earlier causal-output delta =",
          np.max(abs(causal_output[:2] - changed_causal[:2])))

    rng = np.random.default_rng(66)
    print("Scaling: seed=66; 256 independent queries, 64 independent keys per d_k")
    print("d_k raw_std scaled_std raw_entropy scaled_entropy (nats)")
    for dimension in (4, 16, 64, 256):
        random_q = rng.normal(size=(256, dimension))
        random_k = rng.normal(size=(64, dimension))
        raw = random_q @ random_k.T
        scaled = raw / np.sqrt(dimension)
        print(f"{dimension:3} {raw.std():.6f} {scaled.std():.6f} "
              f"{entropy(softmax(raw)):.6f} {entropy(softmax(scaled)):.6f}")

    if args.torch:
        try:
            import torch
            from torch.nn.functional import scaled_dot_product_attention as sdpa
        except ImportError as error:
            raise SystemExit('PyTorch comparison requires: pip install -e ".[deep-learning]"') from error
        torch.set_num_threads(1)
        print(f"PyTorch {torch.__version__}; CPU float64; dropout_p=0")
        for name, mask, expected in (
            ("full", None, output), ("causal", causal, causal_output)
        ):
            tensors = [torch.from_numpy(x)[None, None] for x in (q, k, v)]
            torch_mask = None if mask is None else torch.from_numpy(mask)
            with torch.inference_mode():
                actual = sdpa(*tensors, attn_mask=torch_mask, dropout_p=0.0).numpy()[0, 0]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            print(f"{name} NumPy/SDPA max absolute error = {np.max(abs(actual - expected)):.3e}")
    print("Interpretation candidates require author review; no production claim.")


if __name__ == "__main__":
    main()
