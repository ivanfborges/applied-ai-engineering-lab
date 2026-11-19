"""Synthetic normal-only training, independent calibration and frozen test scoring."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError as error:
    raise ImportError('Install the canonical extra: python -m pip install -e ".[dev,deep-learning]"') from error

from from_scratch import matrix, positive_integer


@dataclass
class SyntheticData:
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    kinds: np.ndarray
    basis: np.ndarray


def make_data(seed=42, n_train=1800, n_calibration=800, n_normal=360, n_each_anomaly=20):
    for name, value in (("n_train", n_train), ("n_calibration", n_calibration),
                        ("n_normal", n_normal), ("n_each_anomaly", n_each_anomaly)):
        positive_integer(value, name)
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))

    def normal(n):
        return rng.normal(size=(n, 2)) @ basis[:, :2].T + 0.15 * rng.normal(size=(n, 12))

    train, calibration = normal(n_train), normal(n_calibration)
    normal_test = normal(n_normal)
    off = normal(n_each_anomaly) + 4 * basis[:, 2]
    on = normal(n_each_anomaly) + 6 * basis[:, 0]
    test = np.vstack((normal_test, off, on))
    kinds = np.asarray(["normal"] * n_normal + ["off_subspace"] * n_each_anomaly
                       + ["on_subspace"] * n_each_anomaly)
    labels = (kinds != "normal").astype(int)
    order = rng.permutation(len(test))
    return SyntheticData(train, calibration, test[order], labels[order], kinds[order], basis)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=12, latent_dim=2):
        super().__init__()
        positive_integer(input_dim, "input_dim")
        positive_integer(latent_dim, "latent_dim")
        if latent_dim > input_dim:
            raise ValueError("latent_dim must not exceed input_dim in this bottleneck study")
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def fit_autoencoder(X, *, seed=42, epochs=120, latent_dim=2):
    X = matrix(X)
    positive_integer(epochs, "epochs")
    torch.manual_seed(seed)
    model = Autoencoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.tensor(X, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    def loss_value():
        model.eval()
        with torch.no_grad():
            return float(nn.functional.mse_loss(model(inputs), inputs))

    history = [loss_value()]
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        for indices in order.split(128):
            batch = inputs[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
        history.append(loss_value())
    model.eval()
    return model, history


def reconstruction_scores(X, reconstructed):
    X, reconstructed = matrix(X), matrix(reconstructed, "reconstructed")
    if X.shape != reconstructed.shape:
        raise ValueError("Reconstruction shape must match X exactly")
    return np.mean((X - reconstructed)**2, axis=1)


def ae_scores(model, X):
    X = matrix(X)
    if X.shape[1] != model.encoder[0].in_features:
        raise ValueError("Feature count differs from the fitted model")
    model.eval()
    with torch.no_grad():
        reconstructed = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return reconstruction_scores(X, reconstructed)


def score_vector(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty finite vector")
    return values


def calibrate_threshold(normal_scores, quantile=0.99):
    scores = score_vector(normal_scores)
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must be finite and strictly between 0 and 1")
    return float(np.quantile(scores, quantile, method="higher"))


def evaluate_scores(labels, scores, threshold):
    scores = score_vector(scores)
    labels = np.asarray(labels)
    if labels.shape != scores.shape or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Labels must match scores and contain both binary classes")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    alert = scores > threshold
    tp = int(np.sum(alert & (labels == 1)))
    fp = int(np.sum(alert & (labels == 0)))
    fn = int(np.sum(~alert & (labels == 1)))
    tn = int(np.sum(~alert & (labels == 0)))
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn), "false_positive_rate": fp / (fp + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp}


def run_study(seed=42, epochs=120):
    positive_integer(epochs, "epochs")
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        data = make_data(seed)
        scaler = StandardScaler().fit(data.train)
        train, calibration, test = [scaler.transform(x) for x in (data.train, data.calibration, data.test)]
        model, history = fit_autoencoder(train, seed=seed, epochs=epochs)
        pca = PCA(n_components=2, svd_solver="full").fit(train)
        forest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(train)
        scorers = {
            "autoencoder": lambda x: ae_scores(model, x),
            "pca": lambda x: reconstruction_scores(x, pca.inverse_transform(pca.transform(x))),
            # sklearn score_samples is larger for normal observations; reverse it.
            "isolation_forest": lambda x: -forest.score_samples(x),
        }
        results = {}
        for name, scorer in scorers.items():
            calibration_scores = scorer(calibration)
            threshold = calibrate_threshold(calibration_scores)
            scores = scorer(test)
            metrics = evaluate_scores(data.labels, scores, threshold)
            metrics.update({"threshold": threshold,
                            "calibration_alerts": int(np.sum(calibration_scores > threshold)),
                            "recall_by_kind": {kind: float(np.mean(scores[data.kinds == kind] > threshold))
                                               for kind in ("off_subspace", "on_subspace")}})
            results[name] = metrics
        return {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hypothesis": "Reconstruction can detect subspace violations while missing rare in-subspace points.",
            "synthetic": True,
            "config": {"seed": seed, "train": 1800, "calibration": 800, "test_normal": 360,
                       "test_each_anomaly": 20, "features": 12, "latent_dim": 2,
                       "normal_noise_sd": 0.15, "off_subspace_shift": 4, "on_subspace_shift": 6,
                       "architecture": "12-16-ReLU-2-16-ReLU-12", "epochs": epochs,
                       "batch_size": 128, "optimizer": "Adam", "learning_rate": 0.001,
                       "checkpoint_policy": "fixed final epoch; no validation selection",
                       "quantile": 0.99, "quantile_method": "higher", "decision": "score > threshold",
                       "forest_trees": 100, "pca_solver": "full", "device": "cpu", "torch_threads": 1},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scikit_learn": sklearn.__version__, "torch": torch.__version__},
            "ae_train_mse": {"initial": history[0], "final": history[-1]},
            "results": results,
            "interpretation_candidate": "Compare recall by anomaly mechanism; aggregate metrics can hide failures.",
            "interpretation_status": "author review required",
            "limitations": "One seed, linear synthetic normal structure, 10% test anomalies, fixed untuned models; no production evidence.",
        }
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--output", type=Path, help="Optional new JSON record; existing files are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new run filename")
    report = run_study(args.seed, args.epochs)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)
