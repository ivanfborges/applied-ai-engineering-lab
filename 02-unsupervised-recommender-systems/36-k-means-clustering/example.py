"""Synthetic scaling and cluster-count diagnostics; prints tables, writes no files."""

from __future__ import annotations

import platform

import numpy as np
import sklearn
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from from_scratch import kmeans_numpy


SEED = 42


def make_data():
    """Three synthetic Gaussian groups with an artificial 50x second-axis scale."""
    X, generating_labels = make_blobs(
        n_samples=600,
        centers=[(-4, -1), (0, 4), (4, -1)],
        cluster_std=[0.8, 1.1, 0.7],
        random_state=SEED,
    )
    X[:, 1] *= 50
    return X, generating_labels


def build_model(k, *, standardized):
    clusterer = KMeans(
        n_clusters=k, init="k-means++", n_init=10,
        random_state=SEED, max_iter=300, tol=1e-4, algorithm="lloyd",
    )
    if standardized:
        return make_pipeline(StandardScaler(), clusterer)
    return make_pipeline(clusterer)


def diagnostics(X, labels, centers):
    """Independently recompute inertia; silhouette is undefined for one cluster."""
    residuals = X - centers[labels]
    inertia = float(np.sum(residuals ** 2))
    count = len(np.unique(labels))
    silhouette = float(silhouette_score(X, labels)) if 2 <= count < len(X) else None
    return inertia, silhouette


def run_experiment():
    X, generating_labels = make_data()
    fitted = {}
    for standardized in (False, True):
        space = "standardized" if standardized else "raw"
        print(f"\n{space} geometry: inertia and silhouette in this fit space only")
        print(" K     inertia   silhouette  cluster sizes (arbitrary ID order)")
        for k in range(1, 9):
            pipeline = build_model(k, standardized=standardized).fit(X)
            model = pipeline.named_steps["kmeans"]
            fit_X = pipeline.named_steps["standardscaler"].transform(X) if standardized else X
            inertia, silhouette = diagnostics(fit_X, model.labels_, model.cluster_centers_)
            np.testing.assert_allclose(inertia, model.inertia_, rtol=1e-10, atol=1e-8)
            score = "undefined" if silhouette is None else f"{silhouette:.6f}"
            print(f"{k:2d} {inertia:11.4f} {score:>12}  {np.bincount(model.labels_, minlength=k).tolist()}")
            if k == 3:
                fitted[space] = pipeline

    # Use one shared evaluation geometry when comparing the two partitions.
    scaled = fitted["standardized"].named_steps["standardscaler"].transform(X)
    print("\nFixed K=3 comparison; generating labels are used only for diagnostics")
    for space, pipeline in fitted.items():
        labels = pipeline.named_steps["kmeans"].labels_
        ari = adjusted_rand_score(generating_labels, labels)
        common_silhouette = silhouette_score(scaled, labels)
        print(f"{space}: ARI vs generator={ari:.6f}; silhouette in shared standardized space={common_silhouette:.6f}")
    print("Raw/scaled partition ARI:", f"{adjusted_rand_score(fitted['raw'].predict(X), fitted['standardized'].predict(X)):.6f}")

    initial, _ = kmeans_plusplus(scaled, n_clusters=3, random_state=SEED)
    scratch = kmeans_numpy(scaled, 3, init=initial, max_iter=300, tol=0)
    reference = KMeans(n_clusters=3, init=initial.copy(), n_init=1, max_iter=300, tol=0, algorithm="lloyd").fit(scaled)
    print("\nMatched-initialization check (one run each, standardized corpus)")
    print(f"NumPy inertia={scratch.inertia:.6f}; sklearn inertia={reference.inertia_:.6f}")
    print(f"Partition ARI={adjusted_rand_score(scratch.labels, reference.labels_):.6f}; NumPy iterations={scratch.n_iter}; converged={scratch.converged}")

    new_points = np.array([[-4, -50], [0, 200], [4, -50]])
    pipeline = fitted["standardized"]
    scaler = pipeline.named_steps["standardscaler"]
    print("\nNew synthetic points in raw units:", new_points.tolist())
    print("Frozen-pipeline assignments:", pipeline.predict(new_points).tolist())
    print("K=3 standardized-fit centers, inverse-transformed to raw units:\n", scaler.inverse_transform(pipeline.named_steps["kmeans"].cluster_centers_))


def main():
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; scikit-learn {sklearn.__version__}")
    print("Synthetic exploratory corpus; seed=42; n=600; no held-out performance claim")
    # Small dense fits use one numerical thread to avoid oversubscription overhead.
    with threadpool_limits(limits=1):
        run_experiment()


if __name__ == "__main__":
    main()
