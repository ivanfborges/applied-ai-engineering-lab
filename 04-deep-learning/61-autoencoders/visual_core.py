"""Deterministic CPU experiments for the visual lab, separate from example.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError as error:
    raise ImportError('From the repository root: python -m pip install -e ".[deep-learning]"') from error

from example import calibrate_threshold, evaluate_scores, reconstruction_scores
from from_scratch import matrix, positive_integer


@dataclass
class Splits:
    train: np.ndarray
    validation: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    kinds: np.ndarray


def tabular_data(seed: int = 42, quick: bool = True) -> Splits:
    """Independent rows; shifted legitimate normals and observationally ambiguous anomalies."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))

    def normal(n):
        return rng.normal(size=(n, 2)) @ basis[:, :2].T + 0.12 * rng.normal(size=(n, 12))

    train = normal(512 if quick else 1600)
    validation, calibration = normal(128 if quick else 400), normal(240 if quick else 800)
    n, a = (160, 24) if quick else (480, 60)
    groups = [normal(n), normal(a) + 4 * basis[:, 2], normal(a) + 6 * basis[:, 0],
              normal(a), normal(a) - 4 * basis[:, 1]]
    names = ["normal", "off_subspace", "on_subspace", "normal_like_anomaly", "unusual_normal"]
    kinds = np.concatenate([np.repeat(name, len(x)) for name, x in zip(names, groups)])
    labels = np.isin(kinds, names[1:4]).astype(int)
    order = rng.permutation(len(kinds))
    return Splits(train, validation, calibration, np.vstack(groups)[order], labels[order], kinds[order])


def geometry_data(seed: int = 42, nonlinear: bool = True, quick: bool = True) -> Splits:
    rng = np.random.default_rng(seed)

    def draw(n):
        uv = rng.uniform(-1.8, 1.8, (n, 2))
        u, v = uv.T
        height = 0.8 * (u**2 + 0.6 * v**2) if nonlinear else 0.7 * u - 0.4 * v
        return np.column_stack((u, v, height)) + rng.normal(0, 0.035, (n, 3))

    train = draw(512 if quick else 1600)
    validation, calibration, normal = draw(128), draw(240), draw(160)
    ood = draw(32)
    ood[:, 2] += 2.5
    test = np.vstack((normal, ood))
    return Splits(train, validation, calibration, test, np.r_[np.zeros(160, int), np.ones(32, int)],
                  np.asarray(["normal"] * 160 + ["off_manifold"] * 32))


def scale_splits(data: Splits) -> tuple[Splits, StandardScaler]:
    scaler = StandardScaler().fit(data.train)
    arrays = [scaler.transform(x) for x in (data.train, data.validation, data.calibration, data.test)]
    return Splits(*arrays, data.labels.copy(), data.kinds.copy()), scaler


class VisualAE(nn.Module):
    def __init__(self, dimension: int, latent: int = 2, hidden: int = 8, linear: bool = False):
        super().__init__()
        for name, value in (("dimension", dimension), ("latent", latent), ("hidden", hidden)):
            positive_integer(value, name)
        if linear:
            # Centered inputs and no biases match the linear PCA training objective.
            self.encoder = nn.Sequential(nn.Linear(dimension, latent, bias=False))
            self.decoder = nn.Sequential(nn.Linear(latent, dimension, bias=False))
        else:
            self.encoder = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, latent))
            self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, dimension))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def infer(model: VisualAE, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = matrix(X)
    if X.shape[1] != model.encoder[0].in_features:
        raise ValueError("Input feature count does not match the model")
    model.eval()
    with torch.no_grad():
        latent = model.encoder(torch.as_tensor(X, dtype=torch.float32))
        reconstruction = model.decoder(latent)
    return latent.numpy(), reconstruction.numpy()


def train_model(train: np.ndarray, validation: np.ndarray, *, latent: int = 2, hidden: int = 8,
                epochs: int = 40, seed: int = 42, linear: bool = False,
                corruption_sd: float = 0, probe: np.ndarray | None = None,
                checkpoint_count: int = 12) -> dict:
    train, validation = matrix(train), matrix(validation)
    positive_integer(epochs, "epochs")
    positive_integer(checkpoint_count, "checkpoint_count")
    if train.shape[1] != validation.shape[1] or not np.isfinite(corruption_sd) or corruption_sd < 0:
        raise ValueError("Matching feature counts and a finite nonnegative corruption_sd are required")
    torch.manual_seed(seed)
    model = VisualAE(train.shape[1], latent, hidden, linear)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    X, val = torch.tensor(train, dtype=torch.float32), torch.tensor(validation, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed + 100)
    checkpoints, history = [], []
    selected = set(np.linspace(0, epochs, min(checkpoint_count, epochs + 1), dtype=int))
    probe = train[:64] if probe is None else matrix(probe)

    def record(epoch):
        model.eval()
        with torch.no_grad():
            losses = [float(((model(t) - t)**2).mean()) for t in (X, val)]
        if not np.isfinite(losses).all():
            raise ValueError("Training produced nonfinite losses")
        history.append(losses)
        if epoch in selected:
            z, rec = infer(model, probe)
            checkpoints.append({"epoch": epoch, "latent": z.copy(), "reconstruction": rec.copy(),
                                "state": {k: v.detach().clone() for k, v in model.state_dict().items()}})

    record(0)
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(X), generator=generator).split(128):
            target = X[indices]
            corrupted = target + corruption_sd * torch.randn(target.shape, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(corrupted) - target)**2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite optimization loss")
            loss.backward()
            optimizer.step()
        record(epoch)
    model.eval()
    return {"model": model, "history": np.asarray(history), "checkpoints": checkpoints,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": {"seed": seed, "epochs": epochs, "latent": latent, "hidden": hidden,
                       "linear": linear, "corruption_sd": corruption_sd, "learning_rate": 0.01,
                       "batch_size": 128, "optimizer": "Adam", "selection": "fixed final epoch"}}


def metrics_at(labels, scores, threshold) -> dict:
    result = evaluate_scores(labels, scores, threshold)
    p, r = result["precision"], result["recall"]
    result["f1"] = 2 * p * r / (p + r) if p + r else 0.0
    result["confusion_matrix"] = [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]
    return result


def save_checkpoints(checkpoints: list[dict], path: Path) -> None:
    """Write only tensors/basic types, compatible with torch.load(weights_only=True)."""
    if not checkpoints:
        raise ValueError("Cannot save an empty checkpoint sequence")
    payload = [{"epoch": int(cp["epoch"]), "latent": torch.as_tensor(cp["latent"]),
                "reconstruction": torch.as_tensor(cp["reconstruction"]), "state": cp["state"]}
               for cp in checkpoints]
    with Path(path).open("xb") as handle:
        torch.save(payload, handle)


def evaluate_model(model: VisualAE, data: Splits) -> dict:
    z, reconstruction = infer(model, data.test)
    scores = reconstruction_scores(data.test, reconstruction)
    calibration = reconstruction_scores(data.calibration, infer(model, data.calibration)[1])
    thresholds = {str(q): calibrate_threshold(calibration, q) for q in (0.95, 0.975, 0.99)}
    metrics = {q: metrics_at(data.labels, scores, t) for q, t in thresholds.items()}
    normal = data.labels == 0
    return {"latent": z, "reconstruction": reconstruction, "scores": scores,
            "calibration_scores": calibration, "thresholds": thresholds, "metrics": metrics,
            "normal_test_mse": float(scores[normal].mean()),
            "alert_rate_by_kind": {kind: float(np.mean(scores[data.kinds == kind] > thresholds["0.99"]))
                                   for kind in np.unique(data.kinds)}}


class LabExperiments:
    """Small per-process cache avoids fitting the same demonstration repeatedly."""
    def __init__(self, quick: bool = True, seed: int = 42):
        self.quick, self.seed = quick, seed
        self.epochs = 45 if quick else 180
        self.frames = 12 if quick else 36
        self.raw = tabular_data(seed, quick)
        self.data, self.scaler = scale_splits(self.raw)
        self.cache = {}

    def base(self):
        if "base" not in self.cache:
            print("  Training base 12-8-2-8-12 AE; independent validation and calibration", flush=True)
            trained = train_model(self.data.train, self.data.validation, epochs=self.epochs,
                                  seed=self.seed, probe=self.data.test, checkpoint_count=self.frames)
            self.cache["base"] = {**trained, **evaluate_model(trained["model"], self.data)}
        return self.cache["base"]

    def geometry(self, nonlinear=True):
        key = f"geometry-{nonlinear}"
        if key not in self.cache:
            raw = geometry_data(self.seed, nonlinear, self.quick)
            data, scaler = scale_splits(raw)
            fit = train_model(data.train, data.validation, hidden=24, epochs=self.epochs * 2,
                              seed=self.seed, probe=data.test, checkpoint_count=self.frames)
            self.cache[key] = {**fit, **evaluate_model(fit["model"], data), "data": data,
                               "raw": raw, "scaler": scaler}
        return self.cache[key]

    def comparison(self):
        if "comparison" not in self.cache:
            results = []
            for nonlinear in (False, True):
                geometry = self.geometry(nonlinear)
                data = geometry["data"]
                pca = PCA(2, svd_solver="full").fit(data.train)
                pca_rec = pca.inverse_transform(pca.transform(data.test))
                linear = train_model(data.train, data.validation, linear=True,
                                     epochs=self.epochs * 2, seed=self.seed)
                linear_z, linear_rec = infer(linear["model"], data.test)
                records = [("PCA", pca.transform(data.test), pca_rec),
                           ("Linear AE", linear_z, linear_rec),
                           ("Nonlinear AE", geometry["latent"], geometry["reconstruction"])]
                for name, z, rec in records:
                    normal = data.labels == 0
                    results.append({"dataset": "curved" if nonlinear else "linear", "name": name,
                                    "data": data, "raw": geometry["raw"], "scaler": geometry["scaler"],
                                    "latent": z, "reconstruction": rec,
                                    "normal_mse": float(reconstruction_scores(data.test[normal], rec[normal]).mean()),
                                    "train_mse": float(reconstruction_scores(data.train,
                                        pca.inverse_transform(pca.transform(data.train)) if name == "PCA"
                                        else infer(linear["model"] if name == "Linear AE" else geometry["model"], data.train)[1]).mean())})
            self.cache["comparison"] = results
        return self.cache["comparison"]

    def bottlenecks(self):
        if "bottlenecks" not in self.cache:
            results = []
            for seed in range(self.seed, self.seed + (2 if self.quick else 3)):
                for latent in (1, 2, 4, 8):
                    print(f"  Bottleneck k={latent}, initialization seed={seed}", flush=True)
                    trained = train_model(self.data.train, self.data.validation, latent=latent,
                                          epochs=self.epochs, seed=seed)
                    results.append({**trained, **evaluate_model(trained["model"], self.data)})
            self.cache["bottlenecks"] = results
        return self.cache["bottlenecks"]

    def failures(self):
        if "failures" not in self.cache:
            # Factorial 2x2: the same added rows isolate the contamination intervention.
            rng = np.random.default_rng(self.seed + 90)
            _, _, vt = np.linalg.svd(self.data.train, full_matrices=False)
            index = rng.choice(len(self.data.train), max(1, len(self.data.train) // 10), replace=False)
            contaminated = self.data.train.copy()
            contaminated[index] += 5 * vt[2]
            results = []
            for hidden, latent in ((4, 1), (32, 8)):
                for dirty in (False, True):
                    name = f"{'High' if hidden == 32 else 'Low'} capacity / {'10% contaminated' if dirty else 'clean'}"
                    train = contaminated if dirty else self.data.train
                    trained = train_model(train, self.data.validation, hidden=hidden, latent=latent,
                                          epochs=self.epochs, seed=self.seed)
                    results.append({**trained, **evaluate_model(trained["model"], self.data), "name": name,
                                    "contaminated": dirty})
            self.cache["failures"] = results
        return self.cache["failures"]

    def denoising(self):
        if "denoising" not in self.cache:
            t = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            rng = np.random.default_rng(self.seed + 70)

            def clean(n):
                phase = rng.uniform(-np.pi, np.pi, (n, 1))
                return rng.uniform(0.6, 1.4, (n, 1)) * np.sin(t + phase)

            train, val, test = clean(512), clean(128), clean(64)
            scaler = StandardScaler().fit(train)
            train, val, test = [scaler.transform(x) for x in (train, val, test)]
            noisy = test + rng.normal(0, 0.5, test.shape)
            results = []
            for corruption in (0., 0.5):
                fit = train_model(train, val, hidden=16, epochs=self.epochs, seed=self.seed,
                                  corruption_sd=corruption)
                rec = infer(fit["model"], noisy)[1]
                results.append({**fit, "clean": test, "noisy": noisy, "reconstruction": rec,
                                "clean_target_mse": float(np.mean((test - rec)**2))})
            self.cache["denoising"] = results
        return self.cache["denoising"]

    def scaling(self):
        if "scaling" not in self.cache:
            scales = np.geomspace(1, 100, 12)
            train, val, test = [x * scales for x in (self.raw.train, self.raw.validation, self.raw.test)]
            scaler = StandardScaler().fit(train)
            results = []
            for standardized in (False, True):
                inputs = [scaler.transform(x) for x in (train, val, test)] if standardized else [train, val, test]
                fit = train_model(inputs[0], inputs[1], epochs=self.epochs, seed=self.seed)
                rec = infer(fit["model"], inputs[2])[1]
                physical_rec = scaler.inverse_transform(rec) if standardized else rec
                normal = self.data.labels == 0
                physical_errors = np.mean((test[normal] - physical_rec[normal])**2, axis=0)
                objective_errors = np.mean((inputs[2][normal] - rec[normal])**2, axis=0)
                results.append({**fit, "name": "Standardized" if standardized else "Unscaled",
                                "objective_errors": objective_errors, "physical_errors": physical_errors,
                                "scales": scales})
            self.cache["scaling"] = results
        return self.cache["scaling"]


def serializable(value):
    """Keep only measured arrays/configuration; omit live models and tensor checkpoints."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Splits):
        return {"train": value.train.tolist(), "validation": value.validation.tolist(),
                "calibration": value.calibration.tolist(), "test": value.test.tolist(),
                "labels": value.labels.tolist(), "kinds": value.kinds.tolist()}
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items() if k not in ("model", "state", "scaler")}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value
