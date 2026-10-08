"""Compare tree growth controls on a seeded synthetic two-feature problem."""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SEED = 26


def make_synthetic_data(n_samples=600):
    """Axis-aligned class rule with independently flipped training labels."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 100:
        raise ValueError("n_samples must be an integer of at least 100")
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-1, 1, size=(n_samples, 2))
    clean_y = ((X[:, 0] < -0.45) | ((X[:, 0] > 0.1) & (X[:, 1] > -0.2))).astype(int)
    flips = rng.random(n_samples) < 0.12
    return X, clean_y ^ flips.astype(int)


def select_pruning_alpha(X_train, y_train, X_val, y_val):
    """Select from the training pruning path using validation accuracy only."""
    path = DecisionTreeClassifier(random_state=SEED).cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(path.ccp_alphas)
    best_alpha = float(alphas[0])
    best_score = -1.0
    for alpha in alphas:
        tree = DecisionTreeClassifier(ccp_alpha=float(alpha), random_state=SEED)
        tree.fit(X_train, y_train)
        score = accuracy_score(y_val, tree.predict(X_val))
        if score > best_score or (score == best_score and alpha > best_alpha):
            best_alpha, best_score = float(alpha), float(score)
    return best_alpha, best_score, len(alphas)


def main():
    X, y = make_synthetic_data()
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=SEED
    )
    alpha, validation_accuracy, candidates = select_pruning_alpha(
        X_train, y_train, X_val, y_val
    )
    models = {
        "unrestricted": DecisionTreeClassifier(random_state=SEED),
        "pre-pruned": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED),
        "validation-pruned": DecisionTreeClassifier(ccp_alpha=alpha, random_state=SEED),
    }
    print("Synthetic data: 600 rows; two numeric features; axis-aligned rule; 12% independent label flips.")
    print(f"Stratified split: {len(X_train)} train / {len(X_val)} validation / {len(X_test)} test; seed={SEED}.")
    print(f"Pruning: {candidates} training-path alpha candidates; selected alpha={alpha:.6f}; validation accuracy={validation_accuracy:.3f}.")
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: depth={model.get_depth()}, leaves={model.get_n_leaves()}, train accuracy={train_accuracy:.3f}, test accuracy={test_accuracy:.3f}")
    print("One synthetic split; interpretation requires author review.")


if __name__ == "__main__":
    main()
