"""Generate deterministic synthetic visual experiments for Day 24 KNN."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ASSETS = Path(__file__).resolve().parent / "assets"
COLORS = ("#2878a0", "#d66a35")
BACKGROUND = ListedColormap(("#dcebf2", "#f8e4d9"))
K_VALUES = (1, 3, 5, 9, 15, 25, 50)
DIMENSIONS = (2, 5, 10, 20, 50, 100, 250, 500, 1000)
NOISE_DIMENSIONS = (0, 2, 5, 10, 25, 50, 100, 250)


def nearest_indices(X, query, k, metric="euclidean"):
    """Return exact neighbors in stable distance order."""
    X = np.asarray(X, dtype=float)
    query = np.asarray(query, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or query.shape != (X.shape[1],):
        raise ValueError("Expected nonempty 2D reference data and one matching query")
    if not np.isfinite(X).all() or not np.isfinite(query).all():
        raise ValueError("Features and query must be finite")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("k must be an integer between 1 and the reference count")
    if metric == "euclidean":
        distances = np.linalg.norm(X - query, axis=1)
    elif metric == "manhattan":
        distances = np.abs(X - query).sum(axis=1)
    else:
        raise ValueError("metric must be euclidean or manhattan")
    return np.argsort(distances, kind="stable")[:k], distances


def scatter_classes(ax, X, y, size=32):
    for label in (0, 1):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=size, c=COLORS[label],
                   edgecolor="white", linewidth=0.4, alpha=0.9,
                   label=f"Class {label}")


def plot_knn_neighbors():
    X, y = make_moons(n_samples=80, noise=0.17, random_state=RANDOM_STATE)
    query = np.array([0.6, 0.2])
    selected, distances = nearest_indices(X, query, 5)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    scatter_classes(ax, X, y)
    for rank, index in enumerate(selected, start=1):
        ax.plot([query[0], X[index, 0]], [query[1], X[index, 1]],
                color="#263238", linewidth=1.3, alpha=0.8)
        ax.annotate(str(rank), X[index] + [0.035, 0.02], fontsize=9,
                    weight="bold", color="#17242b")
    ax.scatter(X[selected, 0], X[selected, 1], s=150, facecolors="none",
               edgecolors="#17242b", linewidth=1.7, label="Five nearest")
    ax.scatter(*query, s=125, marker="*", c="#17242b", label="Query", zorder=5)
    ax.set(title="A query borrows labels from nearby stored examples",
           xlabel="Feature 1", ylabel="Feature 2")
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "knn_neighbors.png", dpi=160)
    plt.close(fig)
    return selected, distances[selected]


def plot_distance_metrics():
    origin = np.array([0.0, 0.0])
    destination = np.array([1.5, 1.1])
    euclidean = np.linalg.norm(destination)
    manhattan = np.abs(destination).sum()
    theta = np.linspace(0, 2 * np.pi, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True, sharey=True)
    axes[0].plot(euclidean * np.cos(theta), euclidean * np.sin(theta),
                 "--", color=COLORS[0], alpha=0.7, label="Equal-distance circle")
    axes[0].plot([0, destination[0]], [0, destination[1]],
                 color=COLORS[0], linewidth=3, label="Straight path")
    diamond = np.array([[manhattan, 0], [0, manhattan],
                        [-manhattan, 0], [0, -manhattan], [manhattan, 0]])
    axes[1].plot(diamond[:, 0], diamond[:, 1], "--", color=COLORS[1],
                 alpha=0.7, label="Equal-distance diamond")
    axes[1].plot([0, destination[0], destination[0]],
                 [0, 0, destination[1]], color=COLORS[1],
                 linewidth=3, label="Axis-aligned path")
    for ax, name, value in zip(axes, ("Euclidean (L2)", "Manhattan (L1)"),
                               (euclidean, manhattan)):
        ax.scatter([0, destination[0]], [0, destination[1]],
                   color="#17242b", s=55, zorder=4)
        ax.set(xlim=(-2.9, 2.9), ylim=(-2.9, 2.9),
               title=f"{name}: distance = {value:.2f}",
               xlabel="Feature 1", ylabel="Feature 2")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "distance_metrics.png", dpi=160)
    plt.close(fig)
    return euclidean, manhattan


def scaling_neighbors():
    rng = np.random.default_rng(RANDOM_STATE)
    X = np.column_stack((rng.uniform(22, 68, 70), rng.uniform(30000, 150000, 70)))
    query = np.array([44.0, 81000.0])
    scaler = StandardScaler().fit(X)  # Reference observations only; query is excluded.
    raw_ids, _ = nearest_indices(X, query, 5)
    scaled_X = scaler.transform(X)
    scaled_query = scaler.transform(query.reshape(1, -1))[0]
    scaled_ids, _ = nearest_indices(scaled_X, scaled_query, 5)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    for ax, points, q, selected, title, xlabel, ylabel in (
        (axes[0], X, query, raw_ids, "Raw units", "Age (years)", "Income (USD)"),
        (axes[1], scaled_X, scaled_query, scaled_ids, "Training-fitted standardization",
         "Age (standard deviations)", "Income (standard deviations)"),
    ):
        ax.scatter(points[:, 0], points[:, 1], s=27, color="#a9b6bc", label="Reference rows")
        for index in selected:
            ax.plot([q[0], points[index, 0]], [q[1], points[index, 1]],
                    color=COLORS[1], linewidth=1.2)
        ax.scatter(points[selected, 0], points[selected, 1], s=90,
                   facecolors="none", edgecolors=COLORS[1], linewidth=1.8,
                   label="Five nearest")
        ax.scatter(*q, marker="*", s=180, c="#17242b", zorder=5, label="Query")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("The same query can have different neighbors after scaling", fontsize=13)
    fig.tight_layout()
    fig.savefig(ASSETS / "scaling_neighbors.png", dpi=160)
    plt.close(fig)
    return raw_ids, scaled_ids


def animate_k_boundaries():
    X, y = make_moons(n_samples=180, noise=0.24, random_state=RANDOM_STATE)
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 150))
    mesh = np.column_stack((xx.ravel(), yy.ravel()))
    regions = []
    for k in K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        regions.append(model.predict(mesh).reshape(xx.shape))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    def draw(frame):
        ax.clear()
        ax.contourf(xx, yy, regions[frame], levels=(-0.5, 0.5, 1.5),
                    cmap=BACKGROUND, alpha=0.9)
        scatter_classes(ax, X, y, size=24)
        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
               title=f"Decision regions: k = {K_VALUES[frame]}",
               xlabel="Feature 1", ylabel="Feature 2")
        ax.legend(loc="upper right", fontsize=9)

    animation = FuncAnimation(fig, draw, frames=len(K_VALUES),
                              interval=850, repeat=True)
    animation.save(ASSETS / "k_decision_boundary.gif", writer=PillowWriter(fps=1),
                   dpi=115)
    plt.close(fig)
    return K_VALUES


def plot_weighted_neighbors():
    X = np.array([[1.2, 0.0], [1.3, 0.2], [1.4, -0.1],
                  [0.15, 0.03], [0.18, -0.04]])
    y = np.array([0, 0, 0, 1, 1])
    query = np.array([0.0, 0.0])
    selected, distances = nearest_indices(X, query, 5)
    weights = 1.0 / distances[selected]
    weights /= weights.sum()
    predictions = []
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharex=True, sharey=True)
    for ax, mode in zip(axes, ("uniform", "distance")):
        model = KNeighborsClassifier(n_neighbors=5, weights=mode).fit(X, y)
        prediction = int(model.predict(query.reshape(1, -1))[0])
        predictions.append(prediction)
        influence = np.full(5, 0.2) if mode == "uniform" else weights
        for i, index in enumerate(selected):
            ax.plot([0, X[index, 0]], [0, X[index, 1]], color=COLORS[y[index]],
                    linewidth=1 + 9 * influence[i], alpha=0.65)
            ax.annotate(f"{influence[i]:.2f}", X[index] + [0.03, 0.025], fontsize=8)
        scatter_classes(ax, X, y, size=100)
        ax.scatter(*query, marker="*", s=210, color="#17242b", label="Query", zorder=5)
        ax.set(title=f"{mode.title()} vote: class {prediction}",
               xlabel="Feature 1", xlim=(-0.15, 1.7), ylim=(-0.38, 0.48))
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Feature 2")
    fig.suptitle("Line width and labels show each neighbor's normalized vote")
    fig.tight_layout()
    fig.savefig(ASSETS / "weighted_knn.png", dpi=160)
    plt.close(fig)
    return predictions, weights


def plot_neighborhood_volume():
    dims = np.arange(1, 101)
    fractions = (0.01, 0.05, 0.10)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for fraction in fractions:
        side = fraction ** (1.0 / dims)
        ax.plot(dims, side, linewidth=2, label=f"{fraction:.0%} of volume")
    ax.set(xlabel="Dimensions", ylabel="Required relative side length",
           ylim=(0, 1.02),
           title="An axis-aligned neighborhood expands in high dimension")
    ax.grid(alpha=0.25)
    ax.legend(title="Target fraction")
    fig.text(0.5, 0.01, "Unit-cube approximation: side length = fraction^(1/d).",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(ASSETS / "curse_dimensionality.png", dpi=160)
    plt.close(fig)
    return dims, fractions


def simulate_distance_concentration(n_reference=1000):
    if n_reference < 2:
        raise ValueError("n_reference must be at least 2")
    rng = np.random.default_rng(RANDOM_STATE)
    results = []
    distributions = {}
    for d in DIMENSIONS:
        query = rng.normal(size=d)
        reference = rng.normal(size=(n_reference, d))
        distances = np.linalg.norm(reference - query, axis=1)
        minimum, mean, maximum = distances.min(), distances.mean(), distances.max()
        spread = (maximum - minimum) / mean
        results.append((d, minimum, mean, maximum, distances.std(), spread))
        if d in (2, 50, 500):
            distributions[d] = distances / mean
    result = np.asarray(results)
    if not np.isfinite(result).all():
        raise ValueError("Distance simulation produced non-finite values")
    return result, distributions


def plot_distance_concentration():
    result, distributions = simulate_distance_concentration()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].plot(result[:, 0], result[:, 5], marker="o", linewidth=2)
    axes[0].set(xscale="log", xlabel="Dimensions (log scale)",
                ylabel="(maximum - minimum) / mean",
                title="Relative distance spread to one random query")
    axes[0].grid(alpha=0.25)
    bins = np.linspace(0, 2.5, 40)
    for d, values in distributions.items():
        axes[1].hist(values, bins=bins, density=True, histtype="step",
                     linewidth=2, label=f"d = {d}")
    axes[1].set(xlabel="Distance / mean distance", ylabel="Density",
                title="Normalized distance distributions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "distance_concentration.png", dpi=160)
    plt.close(fig)
    return result


def experiment_irrelevant_features():
    X_signal, y = make_moons(n_samples=500, noise=0.18, random_state=RANDOM_STATE)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    rng = np.random.default_rng(RANDOM_STATE + 1)
    noise = rng.normal(size=(len(y), max(NOISE_DIMENSIONS)))
    results = []
    for d in NOISE_DIMENSIONS:
        X = np.column_stack((X_signal, noise[:, :d]))
        pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9))
        pipeline.fit(X[train_idx], y[train_idx])
        accuracy = accuracy_score(y[test_idx], pipeline.predict(X[test_idx]))
        results.append((d, accuracy))
    result = np.asarray(results)
    if not np.isfinite(result).all():
        raise ValueError("Irrelevant-feature experiment produced non-finite values")
    return result


def plot_irrelevant_features():
    result = experiment_irrelevant_features()
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(result[:, 0], result[:, 1], marker="o", linewidth=2)
    ax.set(xlabel="Independent noise features appended",
           ylabel="Held-out accuracy", ylim=(0, 1.02),
           title="KNN with two signal features and added noise coordinates")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "irrelevant_features.png", dpi=160)
    plt.close(fig)
    return result


def plot_vector_retrieval():
    angles = np.deg2rad([8, 24, 51, 105, 155, 210, 260, 318])
    documents = np.column_stack((np.cos(angles), np.sin(angles)))
    query = np.array([1.0, 0.25])
    scores = documents @ query / (np.linalg.norm(documents, axis=1) * np.linalg.norm(query))
    selected = np.argsort(-scores, kind="stable")[:3]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.scatter(documents[:, 0], documents[:, 1], s=80, color="#a9b6bc")
    for i, point in enumerate(documents):
        ax.annotate(f"D{i}", point + [0.035, 0.025], fontsize=9)
    for index in selected:
        ax.plot([0, documents[index, 0]], [0, documents[index, 1]],
                color=COLORS[0], linewidth=1.8)
    ax.scatter(documents[selected, 0], documents[selected, 1], s=150,
               facecolors="none", edgecolors=COLORS[0], linewidth=2,
               label="Top 3 by cosine")
    ax.arrow(0, 0, query[0], query[1], head_width=0.055,
             color=COLORS[1], length_includes_head=True, linewidth=2)
    ax.set(xlim=(-1.3, 1.4), ylim=(-1.3, 1.35), aspect="equal",
           xlabel="Vector coordinate 1", ylabel="Vector coordinate 2",
           title="Mock document vectors and query")
    ax.legend(loc="lower left", fontsize=9)
    axes[1].axis("off")
    axes[1].set(xlim=(0, 1), ylim=(0, 1))
    for x, title, steps, color in (
        (0.05, "KNN classification", ["Query", "Nearby labeled rows",
                                     "Aggregate labels", "Prediction"], COLORS[1]),
        (0.54, "Vector retrieval", ["Query embedding", "Nearby document vectors",
                                    "Retrieve context", "Downstream model"], COLORS[0]),
    ):
        axes[1].text(x, 0.92, title, fontsize=11, weight="bold", color=color)
        for j, step in enumerate(steps):
            y_pos = 0.76 - 0.19 * j
            axes[1].text(x, y_pos, step, fontsize=9.5,
                         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color))
            if j < 3:
                axes[1].annotate("", xy=(x + 0.13, y_pos - 0.155),
                                 xytext=(x + 0.13, y_pos - 0.06),
                                 arrowprops=dict(arrowstyle="->", color=color))
    fig.suptitle("Both search neighborhoods; their downstream tasks differ")
    fig.tight_layout()
    fig.savefig(ASSETS / "vector_retrieval.png", dpi=160)
    plt.close(fig)
    return selected, scores[selected]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    selected, distances = plot_knn_neighbors()
    l2, l1 = plot_distance_metrics()
    raw_ids, scaled_ids = scaling_neighbors()
    animate_k_boundaries()
    predictions, weights = plot_weighted_neighbors()
    plot_neighborhood_volume()
    concentration = plot_distance_concentration()
    irrelevant = plot_irrelevant_features()
    retrieved, similarities = plot_vector_retrieval()
    print("Generated assets: knn_neighbors.png, distance_metrics.png, scaling_neighbors.png, "
          "k_decision_boundary.gif, weighted_knn.png, curse_dimensionality.png, "
          "distance_concentration.png, irrelevant_features.png, vector_retrieval.png")
    print(f"Nearest indices: {selected.tolist()}; distances: {np.round(distances, 3).tolist()}")
    print(f"Metric example: Euclidean={l2:.3f}; Manhattan={l1:.3f}")
    print(f"Scaling neighbors: raw={raw_ids.tolist()}; standardized={scaled_ids.tolist()}")
    print(f"Voting example: uniform class={predictions[0]}; distance class={predictions[1]}")
    print("Distance weights: " + str(np.round(weights, 3).tolist()))
    print("Distance concentration: d, min, mean, max, SD, relative spread")
    for row in concentration:
        print("  " + ", ".join(f"{value:.4f}" if j else str(int(value))
                             for j, value in enumerate(row)))
    print("Irrelevant dimensions: noise features, held-out accuracy")
    for d, accuracy in irrelevant:
        print(f"  {int(d)}, {accuracy:.3f}")
    print(f"Cosine retrieval indices={retrieved.tolist()}; similarities={np.round(similarities, 3).tolist()}")


if __name__ == "__main__":
    main()
