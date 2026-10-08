"""Educational bootstrap ensemble of randomized one-split classifiers.

This exposes forest sampling and OOB voting, but it is not a full random forest:
each learner is a stump, so it samples features for its only split.
"""

import numpy as np


def _numeric_matrix(X):
    try:
        values = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite numeric matrix") from exc
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("X must be a nonempty finite numeric matrix")
    return values


def _binary_target(y, n_rows):
    try:
        values = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain both binary labels 0 and 1") from exc
    if (
        values.ndim != 1
        or len(values) != n_rows
        or not np.isfinite(values).all()
        or not np.array_equal(np.unique(values), [0, 1])
    ):
        raise ValueError("y must contain both binary labels 0 and 1, one per row")
    return values.astype(np.int8)


def _gini(y):
    proportion = np.mean(y)
    return 2.0 * proportion * (1.0 - proportion)


class RandomStump:
    """Fit the best Gini split among one sampled subset of features."""

    def fit(self, X, y, features):
        self.feature_ = None
        self.default_class_ = int(np.mean(y) >= 0.5)
        best_impurity = _gini(y)
        for feature in features:
            unique = np.unique(X[:, feature])
            for threshold in (unique[:-1] + unique[1:]) / 2.0:
                left = X[:, feature] <= threshold
                n_left = int(left.sum())
                n_right = len(y) - n_left
                impurity = (n_left * _gini(y[left]) + n_right * _gini(y[~left])) / len(y)
                if impurity < best_impurity:
                    best_impurity = impurity
                    self.feature_ = int(feature)
                    self.threshold_ = float(threshold)
                    self.left_class_ = int(np.mean(y[left]) >= 0.5)
                    self.right_class_ = int(np.mean(y[~left]) >= 0.5)
        return self

    def predict(self, X):
        if self.feature_ is None:
            return np.full(len(X), self.default_class_, dtype=np.int8)
        return np.where(
            X[:, self.feature_] <= self.threshold_, self.left_class_, self.right_class_
        ).astype(np.int8)


class RandomStumpForest:
    """Bootstrap stumps, randomly select candidate features, and vote."""

    def __init__(self, n_estimators=80, max_features=None, random_state=27):
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 1
        ):
            raise ValueError("max_features must be None or a positive integer")
        if isinstance(random_state, bool) or not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = _numeric_matrix(X)
        if len(X) < 2:
            raise ValueError("X must have at least two rows")
        y = _binary_target(y, len(X))
        n_rows, n_features = X.shape
        m = self.max_features or max(1, int(np.sqrt(n_features)))
        if m > n_features:
            raise ValueError("max_features cannot exceed the number of features")

        rng = np.random.default_rng(self.random_state)
        self.stumps_ = []
        self.bootstrap_indices_ = []
        self.feature_subsets_ = []
        self.n_features_in_ = n_features
        vote_sum = np.zeros(n_rows, dtype=int)
        vote_count = np.zeros(n_rows, dtype=int)

        for _ in range(self.n_estimators):
            indices = rng.integers(0, n_rows, size=n_rows)
            features = rng.choice(n_features, size=m, replace=False)
            stump = RandomStump().fit(X[indices], y[indices], features)
            self.stumps_.append(stump)
            self.bootstrap_indices_.append(indices)
            self.feature_subsets_.append(features)

            # Only trees that omitted a training row may contribute to its OOB vote.
            oob_mask = np.ones(n_rows, dtype=bool)
            oob_mask[indices] = False
            vote_sum[oob_mask] += stump.predict(X[oob_mask])
            vote_count[oob_mask] += 1

        covered = vote_count > 0
        self.oob_counts_ = vote_count
        self.oob_predictions_ = np.full(n_rows, np.nan)
        self.oob_predictions_[covered] = (vote_sum[covered] * 2 >= vote_count[covered]).astype(int)
        self.oob_coverage_ = float(np.mean(covered))
        self.oob_accuracy_ = (
            float(np.mean(self.oob_predictions_[covered] == y[covered]))
            if covered.any()
            else None
        )
        return self

    def predict(self, X):
        if not hasattr(self, "stumps_"):
            raise RuntimeError("fit must be called before predict")
        X = _numeric_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than the training data")
        votes = np.stack([stump.predict(X) for stump in self.stumps_])
        return (votes.sum(axis=0) * 2 >= len(self.stumps_)).astype(np.int8)


def main():
    rng = np.random.default_rng(27)
    X = rng.uniform(-1, 1, size=(240, 4))
    clean_y = ((X[:, 0] > 0.2) | ((X[:, 1] > 0.3) & (X[:, 2] < 0))).astype(int)
    y = clean_y ^ (rng.random(len(X)) < 0.10).astype(int)
    forest = RandomStumpForest(n_estimators=80, max_features=2, random_state=27).fit(X, y)
    print("Synthetic data: 240 rows; four numeric features; nonlinear rule; 10% label flips.")
    print("Educational model: 80 randomized stumps; two candidate features per stump.")
    print(f"OOB coverage: {forest.oob_coverage_:.3f}")
    print(f"OOB accuracy on covered training rows: {forest.oob_accuracy_:.3f}")
    print("This one-split ensemble is not equivalent to a full Random Forest.")


if __name__ == "__main__":
    main()
