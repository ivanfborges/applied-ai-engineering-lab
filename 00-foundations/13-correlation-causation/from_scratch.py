"""Educational correlation and linear-adjustment helpers built with NumPy."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _as_finite_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    """Return a validated one-dimensional float array."""
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two observations.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite observations.")
    return array


def pearson_correlation(x: Iterable[float], y: Iterable[float]) -> float:
    """Calculate sample Pearson correlation from centered vectors."""
    x_array = _as_finite_vector(x, name="x")
    y_array = _as_finite_vector(y, name="y")
    if x_array.size != y_array.size:
        raise ValueError("x and y must have the same number of observations.")

    x_centered = x_array - np.mean(x_array)
    y_centered = y_array - np.mean(y_array)
    denominator = np.sqrt(
        np.sum(x_centered**2) * np.sum(y_centered**2)
    )
    if denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a constant input.")
    return float(np.sum(x_centered * y_centered) / denominator)


def ols_coefficients(
    features: Iterable[Iterable[float]] | Iterable[float],
    outcome: Iterable[float],
    *,
    add_intercept: bool = True,
) -> np.ndarray:
    """Estimate ordinary least-squares coefficients with ``numpy.linalg.lstsq``.

    This is an educational helper for small, dense examples. It is not a
    replacement for a statistical package that reports uncertainty and model
    diagnostics.
    """
    feature_array = np.asarray(list(features), dtype=float)
    outcome_array = _as_finite_vector(outcome, name="outcome")

    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(-1, 1)
    if feature_array.ndim != 2:
        raise ValueError("features must be a one- or two-dimensional array.")
    if feature_array.shape[0] != outcome_array.size:
        raise ValueError("features and outcome must have the same number of rows.")
    if feature_array.shape[1] == 0:
        raise ValueError("features must contain at least one column.")
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("features must contain only finite observations.")

    design = feature_array
    if add_intercept:
        design = np.column_stack((np.ones(outcome_array.size), feature_array))
    if design.shape[0] < design.shape[1]:
        raise ValueError("at least as many observations as coefficients are required.")

    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome_array, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("the regression design matrix is rank deficient.")
    return coefficients


def residualize(
    target: Iterable[float], controls: Iterable[Iterable[float]] | Iterable[float]
) -> np.ndarray:
    """Remove the fitted linear association between a target and controls."""
    target_array = _as_finite_vector(target, name="target")
    control_array = np.asarray(list(controls), dtype=float)
    if control_array.ndim == 1:
        control_array = control_array.reshape(-1, 1)
    coefficients = ols_coefficients(control_array, target_array)
    design = np.column_stack((np.ones(target_array.size), control_array))
    return target_array - design @ coefficients


def partial_correlation(
    x: Iterable[float],
    y: Iterable[float],
    controls: Iterable[Iterable[float]] | Iterable[float],
) -> float:
    """Calculate linear partial correlation by correlating residuals."""
    return pearson_correlation(
        residualize(x, controls),
        residualize(y, controls),
    )


def main() -> None:
    """Show the helpers on a small exact linear example."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 1.0 + 2.0 * x
    coefficients = ols_coefficients(x, y)

    print("First-principles correlation and OLS demonstration")
    print(f"Pearson correlation: {pearson_correlation(x, y):.4f}")
    print(f"OLS intercept: {coefficients[0]:.4f}")
    print(f"OLS slope: {coefficients[1]:.4f}")
    print("Educational implementation: use statistical libraries for inference.")


if __name__ == "__main__":
    main()
