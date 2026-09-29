"""Small, exact Euclidean KNN classifier for studying neighbor voting."""

import numpy as np


class KNNClassifier:
    """Educational KNN with uniform votes and sorted-label tie breaking."""

    def __init__(self, k: int = 5):
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = int(k)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty 2D feature matrix")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per training row")
        if self.k > len(X):
            raise ValueError("k cannot exceed the number of training rows")
        self.X_train_ = X.copy()
        self.classes_, self.y_indices_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        if not hasattr(self, "X_train_"):
            raise ValueError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X must be 2D with the fitted feature count")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        predictions = []
        for row in X:
            distances = np.linalg.norm(self.X_train_ - row, axis=1)
            # Stable sorting makes equal-distance neighbor selection reproducible.
            neighbors = np.argsort(distances, kind="stable")[: self.k]
            votes = np.bincount(self.y_indices_[neighbors], minlength=len(self.classes_))
            predictions.append(self.classes_[np.argmax(votes)])
        return np.asarray(predictions, dtype=self.classes_.dtype)


if __name__ == "__main__":
    X_train = np.array([[0.0], [0.2], [0.8], [1.0]])
    y_train = np.array(["low", "low", "high", "high"])
    model = KNNClassifier(k=3).fit(X_train, y_train)
    print("Synthetic one-dimensional queries:", model.predict([[0.1], [0.9]]).tolist())
