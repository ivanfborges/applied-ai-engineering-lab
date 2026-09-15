"""Synthetic OLS diagnostics: training geometry does not prove correct specification."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return observed minus predicted values for matching finite 1-D arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.size == 0 or y_pred.shape != y_true.shape:
        raise ValueError("Targets and predictions must be matching nonempty 1-D arrays.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Targets and predictions must contain only finite values.")
    return y_true - y_pred


def residual_diagnostics(X: np.ndarray, errors: np.ndarray) -> dict[str, float]:
    """Summarize residuals against a design augmented with an intercept.

    Column-normalized inner products preserve residual units. They are
    descriptive quantities, not hypothesis tests or scale-free tolerances.
    """
    X = np.asarray(X, dtype=float)
    errors = np.asarray(errors, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty 2-D feature matrix.")
    if errors.shape != (X.shape[0],) or not np.isfinite(errors).all():
        raise ValueError("Residuals must be a finite 1-D array with one value per row.")
    design = np.column_stack((np.ones(len(X)), X))
    norms = np.linalg.norm(design, axis=0)
    # A zero column imposes no additional orthogonality constraint.
    unit_columns = design / np.where(norms > 0, norms, 1.0)
    return {
        "mean_residual": float(errors.mean()),
        "max_normalized_inner_product": float(np.max(np.abs(unit_columns.T @ errors))),
    }


def report_fit(
    name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LinearRegression, np.ndarray]:
    """Fit on training rows only; print evaluation and training diagnostics."""
    model = LinearRegression().fit(X_train, y_train)
    prediction = model.predict(X_test)
    train_errors = residuals(y_train, model.predict(X_train))
    test_errors = residuals(y_test, prediction)
    geometry = residual_diagnostics(X_train, train_errors)
    print(f"\n{name}")
    print(f"  intercept={model.intercept_:.6f}; coefficients={model.coef_.round(6)}")
    print(
        f"  test RMSE={np.sqrt(mean_squared_error(y_test, prediction)):.6f}; "
        f"MAE={mean_absolute_error(y_test, prediction):.6f}; "
        f"R2={r2_score(y_test, prediction):.6f}"
    )
    print(f"  train mean residual={geometry['mean_residual']:.3e}")
    print(
        "  train max |column-normalized X_design.T @ residuals|="
        f"{geometry['max_normalized_inner_product']:.3e}"
    )
    print(f"  test mean residual={test_errors.mean():.6f}")
    return model, test_errors


def main() -> None:
    print("Synthetic data only; arbitrary units. Interpretations require author review.")
    print("Each experiment: n=500, RNG seed=42, 400 train / 100 test, split seed=42.")

    rng = np.random.default_rng(42)
    X = np.column_stack((rng.normal(10, 2, 500), rng.normal(5, 1.5, 500)))
    y = 5 + 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(0, 2, 500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\nExperiment 1: y = 5 + 3*x1 - 2*x2 + Normal(0, std=2)")
    report_fit("Correctly specified OLS", X_train, X_test, y_train, y_test)
    baseline = np.full(y_test.shape, y_train.mean())
    print(
        "  training-mean baseline test RMSE="
        f"{np.sqrt(mean_squared_error(y_test, baseline)):.6f}"
    )

    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, 500)
    y = 3 * x + 2 * x**2 + rng.normal(0, 1, 500)
    train, test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=42)
    print("\nExperiment 2: y = 3*x + 2*x^2 + Normal(0, std=1)")
    # Both feature sets are specified before inspecting the held-out outcomes.
    for name, features in (
        ("Missing quadratic term", x[:, None]),
        ("Including quadratic term", np.column_stack((x, x**2))),
    ):
        _, errors = report_fit(name, features[train], features[test], y[train], y[test])
        print(
            "  test correlation(residual, x^2)="
            f"{np.corrcoef(errors, x[test] ** 2)[0, 1]:.6f}"
        )
        for label, lower, upper in (("left", -3, -1), ("center", -1, 1), ("right", 1, 3)):
            selected = (x[test] >= lower) & (x[test] < upper)
            print(
                f"  {label} test bin: n={selected.sum()}, "
                f"mean residual={errors[selected].mean():.6f}"
            )


if __name__ == "__main__":
    main()
