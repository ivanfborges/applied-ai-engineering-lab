"""Fixed synthetic noisy-XOR comparison; prints metrics without creating assets."""

import json
import platform
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_noisy_xor(n_samples=800, noise=0.12, seed=42):
    """Labels come from latent binary corners, before Gaussian feature noise."""
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, (int, np.integer)) or n_samples < 8:
        raise ValueError("n_samples must be an integer >= 8")
    if not np.isscalar(noise) or not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    corners = rng.integers(0, 2, size=(n_samples, 2))
    X = corners + rng.normal(0, noise, size=corners.shape)
    y = np.logical_xor(corners[:, 0], corners[:, 1]).astype(int)
    return X, y


def run_comparison():
    X, y = make_noisy_xor()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )
    models = {"logistic": LogisticRegression(C=1.0, max_iter=1000)}
    for activation in ("identity", "relu"):
        models[f"mlp_{activation}"] = MLPClassifier(
            hidden_layer_sizes=(8, 8), activation=activation, solver="lbfgs",
            alpha=1e-3, max_iter=1000, max_fun=15000, tol=1e-4, random_state=42,
        )
    results = {}
    for name, estimator in models.items():
        pipeline = make_pipeline(StandardScaler(), estimator)
        # Retain convergence diagnostics rather than silently treating a cap as convergence.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)
        results[name] = {
            "train_accuracy": float(pipeline.score(X_train, y_train)),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "test_log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
            "iterations": int(np.max(np.atleast_1d(estimator.n_iter_))),
            "fit_warnings": [str(item.message) for item in caught],
        }
    return {
        "data": "synthetic latent XOR corners plus Gaussian feature noise",
        "config": {"n_samples": 800, "noise_sd": 0.12, "seed": 42, "split_seed": 42,
                   "test_fraction": 0.25, "stratified": True, "hidden_layers": [8, 8],
                   "solver": "lbfgs", "alpha": 1e-3, "max_iter": 1000,
                   "max_fun": 15000, "tol": 1e-4, "threshold": 0.5},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "train_samples": len(X_train), "test_samples": len(X_test),
        "test_class_counts": np.bincount(y_test, minlength=2).tolist(),
        "models": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_comparison(), indent=2))
