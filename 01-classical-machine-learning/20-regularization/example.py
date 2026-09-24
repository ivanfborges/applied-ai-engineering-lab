"""Compare regularized linear models with fold-local scaling on synthetic data."""

import platform
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20


def make_synthetic_data():
    """500 rows, 30 features, five direct effects, two correlated proxies."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(500, 30))
    X[:, 28] = X[:, 0] + rng.normal(scale=0.05, size=len(X))
    X[:, 29] = X[:, 1] + rng.normal(scale=0.05, size=len(X))
    beta = np.zeros(30)
    beta[:5] = [4.0, -3.0, 2.0, 1.5, -1.0]
    # Define predictors before generating y so beta remains the true DGP.
    y = 8 + X @ beta + rng.normal(scale=4.0, size=len(X))
    scales = np.geomspace(0.1, 100, X.shape[1])
    return X * scales, y, beta / scales


def make_searches():
    """Tune the entire pipeline; no data or fitted scalers are captured here."""
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    alphas = np.logspace(-3, 1, 13)
    candidates = {
        "OLS": (LinearRegression(), {}),
        "Ridge": (Ridge(solver="svd"), {"model__alpha": np.logspace(-3, 3, 13)}),
        "Lasso": (Lasso(max_iter=100_000, tol=1e-7), {"model__alpha": alphas}),
        "ElasticNet": (
            ElasticNet(max_iter=100_000, tol=1e-7),
            {"model__alpha": alphas, "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
    }
    return {
        name: GridSearchCV(
            Pipeline([("scale", StandardScaler()), ("model", model)]),
            grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        for name, (model, grid) in candidates.items()
    }


def main():
    X, y, _ = make_synthetic_data()
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED
    )
    searches = make_searches()
    # Compare converged fits; a warning stops this demonstration.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for search in searches.values():
            search.fit(X_dev, y_dev)
    selected = max(searches, key=lambda name: searches[name].best_score_)
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; scikit-learn {sklearn.__version__}")
    print("Synthetic data only: seed=20, rows=500, features=30, noise SD=4.")
    print(f"Development/test rows: {len(X_dev)}/{len(X_test)}; CV: shuffled 5-fold, seed=20.")
    print(f"Selected by development CV before test evaluation: {selected}")
    print("model       cv_rmse train_rmse test_rmse exact_zeros coef_l2 alpha l1_ratio")
    for name, search in searches.items():
        pipeline = search.best_estimator_
        model = pipeline.named_steps["model"]
        train_rmse = np.sqrt(mean_squared_error(y_dev, pipeline.predict(X_dev)))
        test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))
        print(
            f"{name:11} {-search.best_score_:7.4f} {train_rmse:10.4f} {test_rmse:9.4f} "
            f"{np.count_nonzero(model.coef_ == 0):11d} {np.linalg.norm(model.coef_):7.4f} "
            f"{getattr(model, 'alpha', 0):.6g} {getattr(model, 'l1_ratio', '-')}"
        )
    print("Standardized coefficients for direct features and their proxies:")
    print("model       x0       x28      x1       x29")
    for name, search in searches.items():
        w = search.best_estimator_.named_steps["model"].coef_
        print(f"{name:11} " + " ".join(f"{w[j]:8.4f}" for j in [0, 28, 1, 29]))
    print("Interpretation requires author review; one split cannot establish stability or superiority.")


if __name__ == "__main__":
    main()
