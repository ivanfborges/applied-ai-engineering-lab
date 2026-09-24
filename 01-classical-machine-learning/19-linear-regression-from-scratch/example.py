"""Reproducible synthetic solver comparison and feature-scaling experiment."""

import platform
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from from_scratch import fit_gradient_descent, fit_ols


def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 3))
    y = 4.0 + X @ np.array([3.5, -2.0, 1.2]) + rng.normal(scale=0.8, size=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
    ols = fit_ols(X_train, y_train)
    gd = fit_gradient_descent(Z_train, y_train)
    reference = LinearRegression().fit(X_train, y_train)
    predictions = {
        "NumPy OLS": ols.predict(X_test),
        "Batch GD": gd.predict(Z_test),
        "scikit-learn": reference.predict(X_test),
    }
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, sklearn {sklearn.__version__}")
    print("Synthetic data: seed=42, n=500, p=3, train/test=400/100, noise std=0.8")
    for name, prediction in predictions.items():
        print(f"{name:14} test MSE={mean_squared_error(y_test, prediction):.9f} "
              f"R2={r2_score(y_test, prediction):.9f}")
    # Convert both slope and intercept back to the original feature units.
    gd_coef = gd.coef / scaler.scale_
    gd_beta = np.r_[gd.intercept - scaler.mean_ @ gd_coef, gd_coef]
    print("Original-unit parameters [intercept, slopes]:")
    print("OLS:", ols.beta)
    print("GD: ", gd_beta)
    print("sklearn:", np.r_[reference.intercept_, reference.coef_])
    for name in ("NumPy OLS", "Batch GD"):
        delta = np.max(np.abs(predictions[name] - predictions["scikit-learn"]))
        print(f"{name} max prediction difference from sklearn={delta:.3e}")
    print(f"GD updates={gd.n_iter}, converged={gd.converged}, "
          f"half-MSE={gd.loss_history[0]:.9f} -> {gd.loss_history[-1]:.9f}")
    if not gd.converged:
        raise RuntimeError("The documented GD configuration did not converge.")
    np.testing.assert_allclose(predictions["NumPy OLS"], predictions["scikit-learn"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(predictions["Batch GD"], predictions["scikit-learn"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(gd_beta, ols.beta, atol=1e-6, rtol=0)

    # Change units without changing y, the split, or the underlying relationship.
    unscaled = X_train * np.array([1.0, 10000.0, 1.0])
    scaled = StandardScaler().fit_transform(unscaled)
    print("Feature-scale experiment: train-only, budget=2000, tol=1e-8, step=1/L")
    for name, features in (("Unscaled", unscaled), ("Standardized", scaled)):
        A = np.column_stack((np.ones(len(features)), features))
        L = np.linalg.norm(A, ord=2) ** 2 / len(A)
        fitted = fit_gradient_descent(features, y_train, learning_rate=1 / L, max_iter=2000)
        optimum = fit_ols(features, y_train)
        optimum_loss = mean_squared_error(y_train, optimum.predict(features)) / 2
        print(f"{name:12} step={1 / L:.3e}, cond(A)={np.linalg.cond(A):.3e}, "
              f"updates={fitted.n_iter}, converged={fitted.converged}, "
              f"half-MSE={fitted.loss_history[-1]:.9f}, gap={fitted.loss_history[-1] - optimum_loss:.3e}")


if __name__ == "__main__":
    main()
