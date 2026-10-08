"""One educational, L2-regularized Newton stump for binary logistic loss."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty, finite one-dimensional array")
    return array


def nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def sigmoid(logits) -> np.ndarray:
    z = finite_vector(logits, "logits")
    return np.exp(-np.logaddexp(0.0, -z))


def logistic_derivatives(y, logits) -> tuple[np.ndarray, np.ndarray]:
    labels = finite_vector(y, "y")
    z = finite_vector(logits, "logits")
    if labels.shape != z.shape or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y must contain binary labels and match logits")
    p = sigmoid(z)
    # Compute sigmoid(-z) separately to retain curvature at large positive logits.
    return p - labels, p * sigmoid(-z)


def leaf_weight(gradient: float, hessian: float, reg_lambda: float = 1.0) -> float:
    penalty = nonnegative(reg_lambda, "reg_lambda")
    curvature = nonnegative(hessian, "hessian")
    if not np.isfinite(gradient) or curvature + penalty <= 0:
        raise ValueError("gradient must be finite and hessian + reg_lambda positive")
    return float(-gradient / (curvature + penalty))


def split_gain(g_left, h_left, g_right, h_right, reg_lambda=1.0, gamma=0.0) -> float:
    penalty = nonnegative(reg_lambda, "reg_lambda")
    complexity = nonnegative(gamma, "gamma")
    # Validation also rejects zero-curvature leaves without L2 regularization.
    wl = leaf_weight(g_left, h_left, penalty)
    wr = leaf_weight(g_right, h_right, penalty)
    wp = leaf_weight(g_left + g_right, h_left + h_right, penalty)
    return float(0.5 * (-g_left * wl - g_right * wr + (g_left + g_right) * wp) - complexity)


@dataclass(frozen=True)
class NewtonStump:
    threshold: float | None
    left_weight: float
    right_weight: float
    gain: float

    def predict(self, x) -> np.ndarray:
        values = finite_vector(x, "x")
        if self.threshold is None:
            return np.full(values.shape, self.left_weight)
        return np.where(values <= self.threshold, self.left_weight, self.right_weight)


def fit_newton_stump(x, y, logits, reg_lambda=1.0, gamma=0.0, min_child_weight=0.0) -> NewtonStump:
    values = finite_vector(x, "x")
    gradients, hessians = logistic_derivatives(y, logits)
    if values.shape != gradients.shape:
        raise ValueError("x, y and logits must have the same shape")
    penalty = nonnegative(reg_lambda, "reg_lambda")
    complexity = nonnegative(gamma, "gamma")
    minimum = nonnegative(min_child_weight, "min_child_weight")
    parent = leaf_weight(float(gradients.sum()), float(hessians.sum()), penalty)
    best = NewtonStump(None, parent, parent, 0.0)
    # Observed left endpoints define every distinct partition without midpoint overflow.
    for threshold in np.unique(values)[:-1]:
        left = values <= threshold
        gl, hl = float(gradients[left].sum()), float(hessians[left].sum())
        gr, hr = float(gradients[~left].sum()), float(hessians[~left].sum())
        if hl < minimum or hr < minimum or hl + penalty <= 0 or hr + penalty <= 0:
            continue
        gain = split_gain(gl, hl, gr, hr, penalty, complexity)
        # Only a positive gain justifies an extra leaf; ties retain the first split.
        if gain > best.gain:
            best = NewtonStump(float(threshold), leaf_weight(gl, hl, penalty),
                               leaf_weight(gr, hr, penalty), gain)
    return best


def main() -> None:
    x = np.array([1.0, 2.0, 2.5, 3.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([0.0] * 4 + [1.0] * 4)
    z = np.zeros_like(y)
    print("Synthetic eight-row binary example; one Newton step, learning rate 1.")
    for penalty in (0.0, 1.0, 10.0):
        stump = fit_newton_stump(x, y, z, reg_lambda=penalty)
        updated = z + stump.predict(x)
        before = float(np.mean(np.logaddexp(0.0, z) - y * z))
        after = float(np.mean(np.logaddexp(0.0, updated) - y * updated))
        print(f"lambda={penalty:g}, threshold={stump.threshold}, gain={stump.gain:.6f}, "
              f"weights=({stump.left_weight:.6f}, {stump.right_weight:.6f}), "
              f"mean_log_loss={before:.6f}->{after:.6f}")
    print("In-sample arithmetic demonstration; interpretation pending author review.")


if __name__ == "__main__":
    main()
