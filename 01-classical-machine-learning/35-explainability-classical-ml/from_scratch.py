"""Educational permutation reliance and direct PDP/ICE for finite numeric inputs."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import check_array, check_consistent_length


def _index(value, width):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Feature indices must be integers.")
    if not 0 <= value < width:
        raise ValueError("Feature index is out of range.")
    return int(value)


def _predictions(predict, X):
    values = np.asarray(predict(X), dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError("predict must return one finite scalar per row.")
    return values


def permutation_importance(model, X, y, *, scoring, groups=None, repeats=10, seed=35):
    """Return baseline-minus-score for independent shuffles of columns or groups.

    scoring follows sklearn's higher-is-better convention, including negated
    losses. groups is a sequence of nonempty column-index sequences; groups may
    overlap to compare individual features with their joint importance.
    Repeat standard deviations use ddof=0, as in sklearn, and are not CIs.
    """
    X = check_array(X, dtype=float, ensure_min_samples=2)
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional target.")
    check_consistent_length(X, y)
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 2:
        raise ValueError("repeats must be an integer of at least 2.")
    if groups is None:
        groups = [(index,) for index in range(X.shape[1])]
    groups = list(groups)
    if not groups:
        raise ValueError("At least one group is required.")
    validated = []
    for group in groups:
        columns = tuple(_index(index, X.shape[1]) for index in group)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("Each group must contain distinct indices and be nonempty.")
        validated.append(columns)

    scorer = check_scoring(model, scoring=scoring)

    def score(data):
        result = float(scorer(model, data, y))
        if not np.isfinite(result):
            raise ValueError("The evaluation score must be finite.")
        return result

    baseline = score(X)
    rng = np.random.default_rng(seed)
    decreases = np.empty((len(validated), repeats))
    for group_index, columns in enumerate(validated):
        for repeat in range(repeats):
            permuted = X.copy()
            # One row permutation preserves dependence within a feature group.
            order = rng.permutation(len(X))
            permuted[:, columns] = X[order][:, columns]
            decreases[group_index, repeat] = baseline - score(permuted)
    return {
        "baseline": baseline,
        "groups": validated,
        "importances": decreases,
        "importances_mean": decreases.mean(axis=1),
        "importances_std": decreases.std(axis=1, ddof=0),
    }


def pdp_ice(predict, X, feature, grid):
    """Return direct ICE (rows x grid) and its PDP average in predict's units."""
    X = check_array(X, dtype=float)
    feature = _index(feature, X.shape[1])
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("grid must be a nonempty finite one-dimensional array.")
    ice = np.empty((len(X), len(grid)))
    for index, value in enumerate(grid):
        replaced = X.copy()
        replaced[:, feature] = value
        ice[:, index] = _predictions(predict, replaced)
    return {"grid": grid.copy(), "ice": ice, "pdp": ice.mean(axis=0)}


def main():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(35)
    X_train, X_test = rng.normal(size=(100, 2)), rng.normal(size=(60, 2))
    y_train, y_test = 2 * X_train[:, 0], 2 * X_test[:, 0]
    model = LinearRegression().fit(X_train, y_train)
    result = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_squared_error", repeats=10
    )
    curves = pdp_ice(model.predict, X_test, 0, [-1, 0, 1])
    print("Synthetic y = 2*x0; frozen linear model, seed 35")
    print("Mean MSE increase:", np.round(result["importances_mean"], 6))
    print("PDP predictions at x0 = -1, 0, 1:", np.round(curves["pdp"], 6))


if __name__ == "__main__":
    main()