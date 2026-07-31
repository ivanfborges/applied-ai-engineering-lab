"""Check analytical gradients and fit a line to synthetic data with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitResult:
    """Parameters and loss history returned by gradient descent."""

    weight: float
    bias: float
    losses: list[float]


def make_synthetic_data(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic observations from y = 3.2x - 1.5 + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    noise = rng.normal(loc=0.0, scale=0.8, size=n_samples)
    y = 3.2 * x - 1.5 + noise
    return x, y


def mean_squared_error(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
) -> float:
    """Return mean((weight * x + bias - y) ** 2)."""
    residuals = weight * x + bias - y
    return float(np.mean(residuals**2))


def analytical_gradients(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
) -> tuple[float, float]:
    """Compute the closed-form MSE partial derivatives for weight and bias."""
    residuals = weight * x + bias - y
    d_loss_d_weight = 2.0 * np.mean(residuals * x)
    d_loss_d_bias = 2.0 * np.mean(residuals)
    return float(d_loss_d_weight), float(d_loss_d_bias)


def numerical_gradients(
    x: np.ndarray,
    y: np.ndarray,
    weight: float,
    bias: float,
    epsilon: float = 1e-6,
) -> tuple[float, float]:
    """Estimate both derivatives with centered finite differences."""
    d_loss_d_weight = (
        mean_squared_error(x, y, weight + epsilon, bias)
        - mean_squared_error(x, y, weight - epsilon, bias)
    ) / (2.0 * epsilon)

    d_loss_d_bias = (
        mean_squared_error(x, y, weight, bias + epsilon)
        - mean_squared_error(x, y, weight, bias - epsilon)
    ) / (2.0 * epsilon)

    return d_loss_d_weight, d_loss_d_bias


def fit_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    steps: int = 200,
) -> FitResult:
    """Minimize MSE using the analytical gradient."""
    weight = 0.0
    bias = 0.0
    losses = [mean_squared_error(x, y, weight, bias)]

    for _ in range(steps):
        d_weight, d_bias = analytical_gradients(x, y, weight, bias)
        weight -= learning_rate * d_weight
        bias -= learning_rate * d_bias
        losses.append(mean_squared_error(x, y, weight, bias))

    return FitResult(weight=weight, bias=bias, losses=losses)


def main() -> None:
    x, y = make_synthetic_data()
    initial_weight = 0.0
    initial_bias = 0.0

    analytical = analytical_gradients(
        x,
        y,
        initial_weight,
        initial_bias,
    )
    numerical = numerical_gradients(
        x,
        y,
        initial_weight,
        initial_bias,
    )

    print("Dataset: synthetic linear observations (fixed random seed)")
    print("\nInitial gradient check")
    print(f"Analytical dL/dw: {analytical[0]: .8f}")
    print(f"Numerical  dL/dw: {numerical[0]: .8f}")
    print(f"Analytical dL/db: {analytical[1]: .8f}")
    print(f"Numerical  dL/db: {numerical[1]: .8f}")

    if not np.allclose(analytical, numerical, rtol=1e-6, atol=1e-7):
        raise RuntimeError("Analytical and numerical gradients do not agree.")

    result = fit_linear_model(x, y)

    print("\nGradient descent")
    print(f"Initial loss:     {result.losses[0]:.6f}")
    print(f"Final loss:       {result.losses[-1]:.6f}")
    print(f"Estimated weight: {result.weight:.4f} (generating value: 3.2000)")
    print(f"Estimated bias:   {result.bias:.4f} (generating value: -1.5000)")
    print(
        "Loss was non-increasing (within floating-point tolerance):",
        all(
            next_loss <= loss + 1e-12
            for loss, next_loss in zip(result.losses, result.losses[1:])
        ),
    )


if __name__ == "__main__":
    main()
