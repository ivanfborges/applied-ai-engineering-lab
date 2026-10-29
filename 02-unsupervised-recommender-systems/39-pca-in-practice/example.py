"""Wine PCA diagnostics and a synthetic low-variance-signal counterexample."""

import platform

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def build_model(n_components=None):
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, svd_solver="full")))
    steps.append(("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)))
    return Pipeline(steps)


def compare_wine():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.25, stratify=wine.target, random_state=42,
    )
    print("Wine: real UCI data bundled in sklearn; 178 rows, 13 features, 3 cultivar classes")
    print(f"Stratified 75/25 split seed=42: train={len(X_train)}, test={len(X_test)}")
    print("Train/test class counts:", np.bincount(y_train), np.bincount(y_test))
    print("CV: 5 stratified shuffled folds seed=42; identical folds for all candidates")
    print("LR: C=1, lbfgs, max_iter=2000; PCA: full SVD, whiten=False")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_train, y_train))
    candidates = [("no_pca", None), ("pca_2", 2), ("pca_95", 0.95)]
    means = []
    print("candidate  CV_balanced_accuracy_mean  fold_SD(ddof=1)  fold_dimensions")
    for name, k in candidates:
        result = cross_validate(
            build_model(k), X_train, y_train, cv=folds,
            scoring="balanced_accuracy", return_estimator=True, error_score="raise",
        )
        scores = result["test_score"]
        dimensions = [m.named_steps["pca"].n_components_ if k is not None else X_train.shape[1]
                      for m in result["estimator"]]
        means.append(float(scores.mean()))
        print(f"{name:9s}  {scores.mean():.8f}                 {scores.std(ddof=1):.8f}       {dimensions}")
    # Ties favor earlier candidates; the unreduced baseline comes first.
    selected = int(np.argmax(means))
    name, k = candidates[selected]
    fitted = build_model(k).fit(X_train, y_train)
    final_score = balanced_accuracy_score(y_test, fitted.predict(X_test))
    print(f"CV-selected={name}; final_test_balanced_accuracy={final_score:.8f}")
    print("Only the CV-selected model receives a test-label score; no tuning on test labels.")

    scaler = StandardScaler().fit(X_train)
    train, test = scaler.transform(X_train), scaler.transform(X_test)
    full = PCA(svd_solver="full").fit(train)
    raw = PCA(svd_solver="full").fit(X_train)
    print("\nTraining-only representation diagnostics (test features used only for reconstruction)")
    print(f"Raw PC1 EVR={raw.explained_variance_ratio_[0]:.8f}; "
          f"standardized PC1 EVR={full.explained_variance_ratio_[0]:.8f}")
    print("Standardized cumulative EVR:", np.round(np.cumsum(full.explained_variance_ratio_), 6))
    print("PC1 direction weights on standardized inputs (axis sign arbitrary):")
    for j in np.argsort(np.abs(full.components_[0]))[::-1][:4]:
        print(f"  {wine.feature_names[j]}: {full.components_[0, j]:+.6f}")
    print("k   train_EVR    train_MSE     heldout_MSE (standardized feature units)")
    for k in (2, 0.95, 13):
        pca = PCA(n_components=k, svd_solver="full").fit(train)
        errors = [np.mean((part - pca.inverse_transform(pca.transform(part))) ** 2)
                  for part in (train, test)]
        print(f"{pca.n_components_:2d}  {pca.explained_variance_ratio_.sum():.8f}  "
              f"{errors[0]:.10f}  {errors[1]:.10f}")


def low_variance_signal():
    rng = np.random.default_rng(42)
    nuisance = rng.normal(0, 20, 800)
    signal = rng.normal(0, 0.1, 800)
    X = np.column_stack([nuisance, signal])
    y = (signal > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42,
    )
    baseline = build_model().fit(X_train, y_train)
    # Raw-unit PCA is deliberate: the nuisance has much larger variance.
    compressed = Pipeline([
        ("pca", PCA(n_components=1, svd_solver="full")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
    ]).fit(X_train, y_train)
    retained = compressed.named_steps["pca"].explained_variance_ratio_.sum()
    print("\nSynthetic counterexample: n=800; seed=42; independent Normal(0,20), Normal(0,0.1)")
    print("Arguments above are standard deviations; y=1 iff second feature>0.")
    print("Stratified 75/25 seed=42; fixed configurations, no model selection")
    print("Baseline scales both features. Compressed model fits PCA in RAW units, then scales scores.")
    print(f"Raw one-PC training EVR={retained:.8f}")
    for name, model in (("no_pca", baseline), ("raw_pca_1", compressed)):
        score = balanced_accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: test_balanced_accuracy={score:.8f}")


def main():
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; "
          f"SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}")
    with threadpool_limits(limits=1):
        compare_wine()
        low_variance_signal()
    print("Interpretation candidates require author review; no files written.")


if __name__ == "__main__":
    main()
