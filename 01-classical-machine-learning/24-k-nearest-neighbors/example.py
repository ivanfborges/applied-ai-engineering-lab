"""Compare fixed-k KNN with and without train-fitted scaling."""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 24
K = 5


def make_synthetic_data(n_samples: int = 400):
    """One predictive feature and one independent feature in larger units."""
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20")
    rng = np.random.default_rng(SEED)
    signal = rng.normal(size=n_samples)
    nuisance = rng.normal(scale=100.0, size=n_samples)
    y = (signal + rng.normal(scale=0.35, size=n_samples) > 0).astype(int)
    return np.column_stack((signal, nuisance)), y


def main():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    raw = KNeighborsClassifier(n_neighbors=K)
    scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    raw.fit(X_train, y_train)
    scaled.fit(X_train, y_train)
    print("Synthetic data: 400 rows; signal SD=1; independent feature SD=100.")
    print(f"Split: {len(X_train)} train / {len(X_test)} test; seed={SEED}; k={K}.")
    print(f"Raw Euclidean KNN test accuracy: {accuracy_score(y_test, raw.predict(X_test)):.3f}")
    print(f"Train-fitted scaled KNN test accuracy: {accuracy_score(y_test, scaled.predict(X_test)):.3f}")
    print("These are one-split synthetic results; interpretation requires author review.")


if __name__ == "__main__":
    main()
