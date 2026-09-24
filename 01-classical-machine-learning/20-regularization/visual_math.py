"""Deterministic numerical experiments for the Day 20 visual laboratory.

All penalties use half-MSE + alpha*rho*L1 + alpha*(1-rho)*L2_squared/2.
Ridge's native sklearn alpha is therefore n_train * alpha.
"""

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

RANDOM_STATE = 42


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _alphas(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("alphas must be a nonempty finite positive 1D array.")
    return values


def soft_threshold(z, alpha):
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("z must be finite and alpha finite and nonnegative.")
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0)


def create_dataset(seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(240, 8))
    X[:, 6] = X[:, 0] + rng.normal(scale=0.12, size=len(X))
    X[:, 7] = X[:, 1] + rng.normal(scale=0.12, size=len(X))
    beta = np.array([3, -2, 1.5, 0, 0, 0, 0, 0])
    y = 2 + X @ beta + rng.normal(scale=1.5, size=len(X))
    return X, y


def regularization_paths(seed=RANDOM_STATE, alphas=None):
    alphas = _alphas(np.logspace(-4, 3, 100) if alphas is None else alphas)
    X, y = create_dataset(seed)
    Z = StandardScaler().fit_transform(X)
    ridge, lasso = [], []
    for alpha in alphas:
        ridge.append(Ridge(alpha=len(Z) * alpha, solver="svd").fit(Z, y).coef_)
        lasso.append(Lasso(alpha=alpha, max_iter=100_000, tol=1e-9).fit(Z, y).coef_)
    return {
        "alphas": alphas, "ridge": np.asarray(ridge), "lasso": np.asarray(lasso),
        "ols": LinearRegression().fit(Z, y).coef_, "X": Z, "y": y,
        "alpha_max_lasso": np.max(np.abs(Z.T @ (y - y.mean()))) / len(y),
    }


def geometry_problem(seed=RANDOM_STATE):
    rng = np.random.default_rng(seed + 1)
    raw = rng.normal(size=(100, 2))
    Q, _ = np.linalg.qr(raw - raw.mean(axis=0))
    X = np.sqrt(len(raw)) * np.column_stack([Q[:, 0], 0.35 * Q[:, 0] + np.sqrt(1 - 0.35**2) * Q[:, 1]])
    y = X @ np.array([2.4, 0.4])
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    radius = 1.0

    def ridge_coef(alpha):
        return np.linalg.solve(X.T @ X / len(X) + alpha * np.eye(2), X.T @ y / len(X))

    ridge_alpha = brentq(lambda a: np.linalg.norm(ridge_coef(a)) - radius, 0, 100)
    alpha_max = np.max(np.abs(X.T @ y)) / len(X)

    def lasso_coef(alpha):
        return Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=100_000).fit(X, y).coef_

    lasso_alpha = brentq(lambda a: np.abs(lasso_coef(a)).sum() - radius, 1e-8, alpha_max)
    return {
        "X": X, "y": y, "ols": ols, "radius": radius,
        "ridge": ridge_coef(ridge_alpha), "lasso": lasso_coef(lasso_alpha),
        "ridge_alpha": ridge_alpha, "lasso_alpha": lasso_alpha,
    }


def objective_grid(X, y, beta1, beta2, alpha=0.0, l1_ratio=0.0):
    """Evaluate centered two-feature objectives without a giant residual cube."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    beta1, beta2 = np.broadcast_arrays(np.asarray(beta1, dtype=float), np.asarray(beta2, dtype=float))
    if X.ndim != 2 or X.shape[1] != 2 or len(X) == 0 or y.shape != (len(X),):
        raise ValueError("Expected nonempty X with two columns and a matching 1D y.")
    if not all(np.isfinite(a).all() for a in (X, y, beta1, beta2)):
        raise ValueError("All inputs must be finite.")
    if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(l1_ratio) or not 0 <= l1_ratio <= 1:
        raise ValueError("Require alpha >= 0 and 0 <= l1_ratio <= 1.")
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    weights = np.stack([beta1, beta2], axis=-1)
    gram, cross = Xc.T @ Xc / len(X), Xc.T @ yc / len(X)
    loss = 0.5 * np.einsum("...i,ij,...j->...", weights, gram, weights) - weights @ cross + yc @ yc / (2 * len(X))
    return loss + alpha * l1_ratio * np.abs(weights).sum(axis=-1) + alpha * (1 - l1_ratio) * (weights**2).sum(axis=-1) / 2


def ridge_prediction_path(X, y, X_eval, alphas):
    """One SVD per training sample, with train-only centering and scaling."""
    alphas = _alphas(alphas)
    scaler = StandardScaler().fit(X)
    Z, Z_eval = scaler.transform(X), scaler.transform(X_eval)
    U, singular, Vt = np.linalg.svd(Z, full_matrices=False)
    weights = Vt.T @ (
        singular[:, None] * (U.T @ (y - np.mean(y)))[:, None]
        / (singular[:, None] ** 2 + len(y) * alphas[None, :])
    )
    return (Z_eval @ weights + np.mean(y)).T


def truth(x):
    return np.sin(np.pi * x) + 0.3 * x


def simulate_bias_variance(seed=RANDOM_STATE, repeats=100, alphas=None):
    """Monte Carlo training-set distribution; fixed noiseless reference grid."""
    _positive_integer(repeats, "repeats", 2)
    alphas = _alphas(np.logspace(-6, 2, 29) if alphas is None else alphas)
    rng = np.random.default_rng(seed + 2)
    n, degree, noise_sd = 35, 12, 0.6
    grid = np.linspace(-1, 1, 201)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_eval = poly.fit_transform(grid[:, None])
    predictions = []
    for _ in range(repeats):
        x = rng.uniform(-1, 1, n)
        y = truth(x) + rng.normal(scale=noise_sd, size=n)
        predictions.append(ridge_prediction_path(poly.transform(x[:, None]), y, X_eval, alphas))
    predictions = np.asarray(predictions)
    mean_prediction = predictions.mean(axis=0)
    bias2 = ((mean_prediction - truth(grid)) ** 2).mean(axis=1)
    # ddof=0 makes the finite Monte Carlo bias/variance identity exact.
    variance = predictions.var(axis=0, ddof=0).mean(axis=1)
    signal_mse = ((predictions - truth(grid)) ** 2).mean(axis=(0, 2))
    return {
        "alphas": alphas, "bias2": bias2, "variance": variance,
        "risk": signal_mse + noise_sd**2, "signal_mse": signal_mse,
        "noise_variance": noise_sd**2, "repeats": repeats, "n_train": n,
        "degree": degree, "grid": grid, "mean_prediction": mean_prediction,
    }


def simulate_multicollinearity(seed=RANDOM_STATE, repeats=100):
    _positive_integer(repeats, "repeats", 2)
    rng = np.random.default_rng(seed + 3)
    sigmas = np.array([1.0, 0.5, 0.1, 0.01])
    coefficients = np.zeros((len(sigmas), repeats, 2, 2))
    rmse = np.zeros((len(sigmas), repeats, 2))
    correlations = np.zeros((len(sigmas), repeats))
    alpha, n = 0.1, 80
    for i, sigma in enumerate(sigmas):
        x_eval = rng.normal(size=1500)
        X_eval = np.column_stack([x_eval, x_eval + rng.normal(scale=sigma, size=len(x_eval))])
        y_eval = X_eval @ np.ones(2) + rng.normal(size=len(x_eval))
        for repetition in range(repeats):
            x = rng.normal(size=n)
            X = np.column_stack([x, x + rng.normal(scale=sigma, size=n)])
            y = X @ np.ones(2) + rng.normal(size=n)
            correlations[i, repetition] = np.corrcoef(X.T)[0, 1]
            for j, estimator in enumerate([LinearRegression(), Ridge(alpha=n * alpha, solver="svd")]):
                pipeline = make_pipeline(StandardScaler(), estimator).fit(X, y)
                coefficients[i, repetition, j] = estimator.coef_ / pipeline[0].scale_
                rmse[i, repetition, j] = np.sqrt(np.mean((pipeline.predict(X_eval) - y_eval) ** 2))
    return {
        "sigmas": sigmas, "coefficients": coefficients, "rmse": rmse,
        "correlations": correlations, "alpha": alpha, "repeats": repeats,
        "n_train": n, "n_eval": 1500,
    }


def analyze_lasso_stability(seed=RANDOM_STATE, repeats=100):
    _positive_integer(repeats, "repeats", 2)
    rng = np.random.default_rng(seed + 4)
    latent = rng.normal(size=160)
    X = latent[:, None] + rng.normal(scale=0.06, size=(160, 2))
    y = 2 * latent + rng.normal(scale=0.8, size=160)
    weights = []
    alpha = 0.2
    for _ in range(repeats):
        indices = rng.integers(0, len(X), len(X))
        pipeline = make_pipeline(
            StandardScaler(), Lasso(alpha=alpha, max_iter=100_000, tol=1e-8)
        ).fit(X[indices], y[indices])
        weights.append(pipeline[-1].coef_)
    weights = np.asarray(weights)
    active = weights != 0
    categories = np.array([
        np.mean(active[:, 0] & ~active[:, 1]), np.mean(~active[:, 0] & active[:, 1]),
        np.mean(active.all(axis=1)), np.mean(~active.any(axis=1)),
    ])
    return {
        "coefficients": weights, "frequency": active.mean(axis=0),
        "categories": categories, "correlation": np.corrcoef(X.T)[0, 1],
        "alpha": alpha, "repeats": repeats, "n_train": len(X),
    }


def elasticnet_grouping(seed=RANDOM_STATE, repeats=60):
    _positive_integer(repeats, "repeats", 2)
    rng = np.random.default_rng(seed + 5)
    latent = rng.normal(size=180)
    X = rng.normal(size=(180, 8))
    X[:, :3] = latent[:, None] + rng.normal(scale=0.08, size=(180, 3))
    y = 3 * latent - 2 * X[:, 3] + rng.normal(scale=1.0, size=len(X))
    ratios, alpha = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]), 0.15
    coefficients = np.zeros((repeats, len(ratios), X.shape[1]))
    for r in range(repeats):
        indices = rng.integers(0, len(X), len(X))
        Z = StandardScaler().fit_transform(X[indices])
        for j, ratio in enumerate(ratios):
            estimator = (
                Ridge(alpha=len(Z) * alpha, solver="svd") if ratio == 0
                else ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=100_000, tol=1e-8)
            )
            coefficients[r, j] = estimator.fit(Z, y[indices]).coef_
    return {
        "ratios": ratios, "alpha": alpha, "coefficients": coefficients,
        "mean": coefficients.mean(axis=0),
        "frequency": (coefficients != 0).mean(axis=0),
        "group_all_frequency": (coefficients[:, :, :3] != 0).all(axis=2).mean(axis=0),
        "repeats": repeats, "n_train": len(X),
    }


def polynomial_predictions(seed=RANDOM_STATE):
    rng = np.random.default_rng(seed + 6)
    x = np.sort(rng.uniform(-1, 1, 32))
    y = truth(x) + rng.normal(scale=0.35, size=len(x))
    grid = np.linspace(-1, 1, 250)
    pipeline = make_pipeline(PolynomialFeatures(12, include_bias=False), StandardScaler(), Ridge(solver="svd"))
    search = GridSearchCV(
        pipeline, {"ridge__alpha": np.logspace(-6, 3, 22)},
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="neg_root_mean_squared_error", error_score="raise",
    ).fit(x[:, None], y)
    alphas = [1e-9, search.best_params_["ridge__alpha"], 1e5]
    predictions = []
    for alpha in alphas:
        pipeline.set_params(ridge__alpha=alpha).fit(x[:, None], y)
        predictions.append(pipeline.predict(grid[:, None]))
    return {
        "x": x, "y": y, "grid": grid, "truth": truth(grid),
        "predictions": np.asarray(predictions), "native_alphas": np.asarray(alphas),
        "cv_rmse": -search.best_score_,
        "function_rmse": np.sqrt(np.mean((np.asarray(predictions) - truth(grid)) ** 2, axis=1)),
    }


def demonstrate_scaling_effect(seed=RANDOM_STATE):
    rng = np.random.default_rng(seed + 7)
    latent = rng.normal(size=(180, 2))
    scales = np.array([1.0, 100_000.0])
    X = latent * scales
    y = 2 * latent.sum(axis=1) + rng.normal(scale=0.4, size=len(X))
    eval_latent = rng.normal(size=(400, 2))
    X_eval = eval_latent * scales
    y_eval = 2 * eval_latent.sum(axis=1) + rng.normal(scale=0.4, size=len(X_eval))
    alpha = 1.0
    raw = Ridge(alpha=len(X) * alpha, solver="svd").fit(X, y)
    scaled = make_pipeline(StandardScaler(), Ridge(alpha=len(X) * alpha, solver="svd")).fit(X, y)
    coefficients = np.stack([raw.coef_, scaled[-1].coef_ / scaled[0].scale_])
    predictions = np.stack([raw.predict(X_eval), scaled.predict(X_eval)])
    return {
        "raw_coefficients": coefficients, "unit_effects": coefficients * scales,
        "predictions": predictions, "y_eval": y_eval,
        "rmse": np.sqrt(np.mean((predictions - y_eval) ** 2, axis=1)),
        "alpha": alpha, "scales": scales, "true_unit_effects": np.array([2.0, 2.0]),
    }
