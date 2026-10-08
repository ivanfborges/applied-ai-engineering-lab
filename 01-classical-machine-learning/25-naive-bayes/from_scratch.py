"""Educational Multinomial Naive Bayes for nonnegative token-count matrices."""

import numpy as np


class MultinomialNBFromScratch:
    """Fit empirical class priors and additively smoothed token likelihoods."""

    def __init__(self, alpha: float = 1.0):
        if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
            raise ValueError("alpha must be a finite positive number")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha must be a finite positive number") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = alpha

    @staticmethod
    def _counts(X):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a numeric token-count matrix") from exc
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D matrix with at least one feature")
        if not np.all(np.isfinite(X)) or np.any(X < 0) or np.any(X != np.floor(X)):
            raise ValueError("X must contain finite nonnegative integer counts")
        return X

    def fit(self, X, y):
        X = self._counts(X)
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one training row")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per training row")
        try:
            classes, indices = np.unique(y, return_inverse=True)
        except TypeError as exc:
            raise ValueError("y must contain comparable class labels") from exc

        class_counts = np.bincount(indices)
        token_counts = np.vstack([X[indices == i].sum(axis=0) for i in range(len(classes))])
        smoothed = token_counts + self.alpha
        token_totals = smoothed.sum(axis=1, keepdims=True)

        self.classes_ = classes
        self.class_log_prior_ = np.log(class_counts / len(y))
        self.feature_log_prob_ = np.log(smoothed) - np.log(token_totals)
        self.n_features_in_ = X.shape[1]
        return self

    def joint_log_likelihood(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError("fit must be called before scoring")
        X = self._counts(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X must have the fitted feature count")
        # The document's multinomial coefficient is common to all classes.
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        scores = self.joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]


if __name__ == "__main__":
    X_train = [[2, 1, 0], [1, 2, 0], [0, 0, 2], [0, 0, 1]]
    y_train = ["billing", "billing", "access", "access"]
    queries = [[1, 0, 0], [0, 0, 1]]
    model = MultinomialNBFromScratch(alpha=1.0).fit(X_train, y_train)
    print("Synthetic count queries:", model.predict(queries).tolist())
    print("Class order:", model.classes_.tolist())
    print("Joint log scores:\n", model.joint_log_likelihood(queries))
