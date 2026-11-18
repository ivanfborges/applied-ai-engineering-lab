"""Educational float64 activations and derivatives; no autodiff or training engine."""

import json

import numpy as np
from scipy.special import ndtr


ELEMENTWISE = ("sigmoid", "tanh", "relu", "leaky_relu", "gelu")


def _finite_array(values):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("inputs must contain real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("inputs must be nonempty and finite")
    return array


def _positive_scalar(value, name):
    array = _finite_array(value)
    if array.ndim != 0 or array <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return float(array)


def activation_and_derivative(values, name, negative_slope=0.01):
    """Return f(x), f'(x), preserving shape, including scalar inputs.

    At zero, select ReLU derivative 0 and Leaky ReLU derivative alpha.
    GELU uses x * Phi(x), not the tanh approximation.
    """
    x = _finite_array(values)
    if name == "sigmoid":
        # exp(-abs(x)) avoids overflow on both branches, even in np.where.
        e = np.exp(-np.abs(x))
        y = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
        return y, e / (1 + e) ** 2
    if name == "tanh":
        y = np.tanh(x)
        return y, 1 - y * y
    if name == "relu":
        return np.maximum(x, 0), (x > 0).astype(np.float64)
    if name == "leaky_relu":
        alpha = _positive_scalar(negative_slope, "negative_slope")
        if alpha >= 1:
            raise ValueError("negative_slope must be strictly between 0 and 1")
        return np.where(x > 0, x, alpha * x), np.where(x > 0, 1.0, alpha)
    if name == "gelu":
        cdf = ndtr(x)
        # Beyond |x|=40 the normal density is negligible in float64.
        # Clipping only this density calculation also prevents x*x overflow.
        density = np.exp(-0.5 * np.minimum(np.abs(x), 40) ** 2) / np.sqrt(2 * np.pi)
        return x * cdf, cdf + x * density
    raise ValueError(f"unknown activation {name!r}; choose one of {ELEMENTWISE}")


def _shifted_logits(logits, axis, temperature):
    z = _finite_array(logits)
    if z.ndim == 0:
        raise ValueError("logits must have at least one dimension")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError("axis must be an integer")
    if not -z.ndim <= axis < z.ndim:
        raise ValueError("axis is out of range")
    t = _positive_scalar(temperature, "temperature")
    # Shift before dividing so a large common offset is not amplified by 1/T.
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = (z - np.max(z, axis=axis, keepdims=True)) / t
    if not np.all(np.isfinite(shifted)):
        raise ValueError("logit differences / temperature exceed float64 range")
    return shifted, t


def softmax(logits, axis=-1, temperature=1.0):
    """Normalize along one explicit axis; reject infinities and attention masks."""
    shifted, _ = _shifted_logits(logits, axis, temperature)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_jacobian(logits, temperature=1.0):
    """Dense educational Jacobian d p_i / d z_j for a single vector."""
    z = _finite_array(logits)
    if z.ndim != 1:
        raise ValueError("Jacobian requires one logit vector")
    t = _positive_scalar(temperature, "temperature")
    p = softmax(z, temperature=t)
    return (np.diag(p) - np.outer(p, p)) / t


def cross_entropy_with_logits(logits, targets, temperature=1.0):
    """Mean categorical loss and gradient for (batch, classes) inputs.

    Targets are fixed nonnegative distributions, one per row, summing to one.
    The gradient includes both mean reduction and temperature scaling.
    """
    z, y = _finite_array(logits), _finite_array(targets)
    if z.ndim != 2 or y.shape != z.shape:
        raise ValueError("logits and targets must have matching (batch, classes) shapes")
    if np.any(y < 0) or not np.allclose(y.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("target rows must be nonnegative and sum to one")
    shifted, t = _shifted_logits(z, -1, temperature)
    exp_values = np.exp(shifted)
    normalizers = np.sum(exp_values, axis=1, keepdims=True)
    # Work with log probabilities directly: log(softmax(z)) can produce log(0).
    log_p = shifted - np.log(normalizers)
    loss = float(np.mean(-np.sum(y * log_p, axis=1)))
    gradient = (exp_values / normalizers - y) / (z.shape[0] * t)
    if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
        raise ValueError("loss or gradient exceeds float64 range")
    return loss, gradient


def finite_difference_jacobian(function, values, epsilon=1e-5):
    """Central differences for vector-to-vector functions; columns perturb inputs."""
    x = _finite_array(values)
    if x.ndim != 1:
        raise ValueError("finite differences require one input vector")
    step = _positive_scalar(epsilon, "epsilon")
    baseline = _finite_array(function(x.copy()))
    if baseline.ndim != 1:
        raise ValueError("function must return a vector")
    columns = []
    for j in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[j] += step
        minus[j] -= step
        if plus[j] == x[j] or minus[j] == x[j]:
            raise ValueError("epsilon is too small at this input scale")
        upper, lower = _finite_array(function(plus)), _finite_array(function(minus))
        if upper.shape != baseline.shape or lower.shape != baseline.shape:
            raise ValueError("function output shape changed under perturbation")
        columns.append((upper - lower) / (2 * step))
    return np.column_stack(columns)


def main():
    logits = np.array([2.0, 1.0, -1.0])
    errors = {}
    for t in (0.5, 1.0, 2.0):
        analytical = softmax_jacobian(logits, temperature=t)
        numerical = finite_difference_jacobian(
            lambda z: softmax(z, temperature=t), logits
        )
        errors[str(t)] = float(np.max(np.abs(analytical - numerical)))
        np.testing.assert_allclose(analytical, numerical, rtol=1e-7, atol=1e-9)
    print(json.dumps({"data": "fixed synthetic logits", "epsilon": 1e-5,
                      "jacobian_max_absolute_errors": errors}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
