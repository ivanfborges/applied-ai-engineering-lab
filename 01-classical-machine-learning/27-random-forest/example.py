"""Inspect OOB evaluation and importance on a seeded synthetic classification task."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 27
FEATURE_NAMES = ("signal_a", "signal_b", "signal_c", "noise_d", "noise_e")


def make_synthetic_data(n_samples=800):
    """Create a nonlinear rule with independent label flips and two unused features."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 100:
        raise ValueError("n_samples must be an integer of at least 100")
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, size=(n_samples, len(FEATURE_NAMES)))
    clean_y = ((X[:, 0] > 0.35) | ((X[:, 1] > 0.1) & (X[:, 2] < -0.2))).astype(int)
    flips = (rng.random(n_samples) < 0.10).astype(int)
    return X, clean_y ^ flips


def main():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        min_samples_leaf=2,
        bootstrap=True,
        oob_score=True,
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    perm = permutation_importance(
        model, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=SEED, n_jobs=1
    )

    print("Synthetic data: 800 rows; five independent uniform features; three in the rule; 10% label flips.")
    print(f"Stratified split: {len(X_train)} train / {len(X_test)} test; seed={SEED}.")
    print("Forest: 150 trees; max_features=sqrt; min_samples_leaf=2; bootstrap=True.")
    print(f"OOB accuracy (training rows): {model.oob_score_:.3f}")
    print(f"Held-out accuracy: {test_accuracy:.3f}")
    print("Feature importance (MDI; held-out permutation accuracy decrease +/- repeat SD):")
    for name, mdi, mean, sd in zip(
        FEATURE_NAMES, model.feature_importances_, perm.importances_mean, perm.importances_std
    ):
        print(f"  {name}: {mdi:.3f}; {mean:+.3f} +/- {sd:.3f}")
    print("Single seeded synthetic split; interpretation requires author review.")


if __name__ == "__main__":
    main()
