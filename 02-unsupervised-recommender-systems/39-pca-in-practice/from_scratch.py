"""Educational dense covariance PCA; use an SVD library for practical workloads."""

from numbers import Integral
import platform

import numpy as np


def _matrix(X, min_rows=1):
    if np.iscomplexobj(X):
        raise ValueError("X must contain real numbers")
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a rectangular numeric matrix") from exc
    if X.ndim != 2 or X.shape[0] < min_rows or X.shape[1] == 0:
        raise ValueError(f"X must be 2D with at least {min_rows} rows and one column")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    return X


class CovariancePCA:
    """Center, eigendecompose, project and reconstruct without standardization.

    Components are rows, matching sklearn's convention. Rank-deficient data
    are allowed, but zero total variance is rejected because EVR is undefined.
    No whitening, sparse inputs, missing-data handling or solver optimizations.
    """

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = _matrix(X, min_rows=2)
        limit = min(X.shape[0] - 1, X.shape[1])
        if (isinstance(self.n_components, (bool, np.bool_))
                or not isinstance(self.n_components, Integral)
                or not 1 <= self.n_components <= limit):
            raise ValueError(f"n_components must be an integer in [1, {limit}]")
        mean = X.mean(axis=0)
        centered = X - mean
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = centered.T @ centered / (len(X) - 1)
        if not np.isfinite(covariance).all():
            raise ValueError("covariance overflowed; rescale the input")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        # Covariance is positive semidefinite; tiny negative values are roundoff.
        eigenvalues = np.maximum(eigenvalues[order], 0)
        total = eigenvalues.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("PCA requires positive, finite total variance")
        components = eigenvectors[:, order[:self.n_components]].T.copy()
        # A display convention only: flipping an axis does not change its subspace.
        pivots = np.argmax(np.abs(components), axis=1)
        components *= np.sign(components[np.arange(len(components)), pivots])[:, None]
        self.mean_ = mean
        self.n_features_in_ = X.shape[1]
        self.components_ = components
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total
        return self

    def _require_fit(self):
        if not hasattr(self, "components_"):
            raise ValueError("fit PCA before transforming data")

    def transform(self, X):
        self._require_fit()
        X = _matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count must match the fitted data")
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, scores):
        self._require_fit()
        scores = _matrix(scores)
        if scores.shape[1] != len(self.components_):
            raise ValueError("scores must have one column per retained component")
        return scores @ self.components_ + self.mean_


def main():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(120, 3))
    mixing = np.array([[3., 2., 1.], [0., 0.5, 0.2], [0., 0., 0.1]])
    X = latent @ mixing + [10., -4., 2.]
    train, test = X[:90], X[90:]
    print(f"Python {platform.python_version()}; NumPy {np.__version__}")
    print("Synthetic Gaussian latent data: seed=42; 120x3; first 90 fit, last 30 held out")
    print("mixing=[[3,2,1],[0,0.5,0.2],[0,0,0.1]]; offset=[10,-4,2]; no scaling")
    print("k  train_EVR  train_MSE     heldout_MSE")
    for k in (1, 2, 3):
        model = CovariancePCA(k).fit(train)
        errors = [np.mean((part - model.inverse_transform(model.transform(part))) ** 2)
                  for part in (train, test)]
        print(f"{k}  {model.explained_variance_ratio_.sum():.8f}  "
              f"{errors[0]:.10f}  {errors[1]:.10f}")
    print("Interpretation candidates require author review; no files written.")


if __name__ == "__main__":
    main()
