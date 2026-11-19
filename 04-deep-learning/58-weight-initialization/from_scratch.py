"""Educational initialization scales and paired NumPy ReLU propagation."""

import math

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def initialization_std(fan_in, fan_out, method="he", *, mode="fan_in", gain=1.0):
    """Normal-distribution standard deviation; gain multiplies the base scale."""
    fan_in = positive_int(fan_in, "fan_in")
    fan_out = positive_int(fan_out, "fan_out")
    if method not in ("xavier", "he") or mode not in ("fan_in", "fan_out"):
        raise ValueError("Use method='xavier'/'he' and mode='fan_in'/'fan_out'")
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    denominator = fan_in + fan_out if method == "xavier" else (fan_in if mode == "fan_in" else fan_out)
    return float(gain * math.sqrt(2.0 / denominator))


def propagate_relu(inputs, weights):
    """Return input and layer second moments, using W shaped (out, in)."""
    h = np.asarray(inputs, dtype=np.float64)
    if h.ndim != 2 or 0 in h.shape or not np.isfinite(h).all():
        raise ValueError("inputs must be a nonempty finite batch-by-feature matrix")
    weights = list(weights)
    if not weights:
        raise ValueError("Provide at least one weight matrix")
    moments = []
    with np.errstate(over="raise", invalid="raise"):
        moments.append(float(np.mean(h * h)))
        for layer, matrix in enumerate(weights, 1):
            matrix = np.asarray(matrix, dtype=np.float64)
            if (matrix.ndim != 2 or 0 in matrix.shape or matrix.shape[1] != h.shape[1]
                    or not np.isfinite(matrix).all()):
                raise ValueError(f"Invalid weight matrix at layer {layer}; expected (out, {h.shape[1]})")
            h = np.maximum(h @ matrix.T, 0.0)
            moments.append(float(np.mean(h * h)))
    return np.array(moments)


def paired_demo(*, width=128, depth=20, batch=128, seed=58):
    width = positive_int(width, "width")
    depth = positive_int(depth, "depth")
    batch = positive_int(batch, "batch")
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((batch, width))
    bases = [rng.standard_normal((width, width)) for _ in range(depth)]
    return {
        name: propagate_relu(inputs, [base * initialization_std(width, width, name) for base in bases])
        for name in ("xavier", "he")
    }


if __name__ == "__main__":
    print("Synthetic Gaussian inputs; seed=58, batch=128, width=128, depth=20; float64")
    print("Shared Gaussian matrices, different scales; no training or backward pass.")
    for name, moments in paired_demo().items():
        print(f"{name:7s} q0={moments[0]:.8e} q20={moments[-1]:.8e} q20/q0={moments[-1]/moments[0]:.8e}")
