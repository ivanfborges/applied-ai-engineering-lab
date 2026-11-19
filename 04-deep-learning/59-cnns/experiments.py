"""Matched CPU CNN/MLP experiments; serialized real checkpoints, no downloads."""

from datetime import datetime, timezone
import platform

import numpy as np
import torch
from torch import nn

from example import valid_seed
from from_scratch import integer
from visual_core import shift_image


def bars(n, seed, noise=0.12):
    """Balanced 32x32 bars. Narrow center support makes translations a new condition."""
    n = integer(n, "n", 2)
    valid_seed(seed)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % 2)
    images = rng.normal(0, noise, (n, 1, 32, 32)).astype(np.float32)
    for index, label in enumerate(labels):
        row, col = rng.integers(14, 18, 2)
        if label == 0:
            images[index, 0, row - 7:row + 8, col - 1:col + 2] += 1
        else:
            images[index, 0, row - 1:row + 2, col - 7:col + 8] += 1
    return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))


class VisionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, device="cpu", dtype=torch.float32)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1, device="cpu", dtype=torch.float32)
        self.head = nn.Linear(16, 2, device="cpu", dtype=torch.float32)

    def features(self, images):
        first = torch.relu(self.conv1(images))
        pooled = nn.functional.max_pool2d(first, 2)
        second = torch.relu(self.conv2(pooled))
        return first, second, nn.functional.max_pool2d(second, 2)

    def forward(self, images):
        return self.head(self.features(images)[-1].mean(dim=(-2, -1)))


class VisionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(32 * 32, 32, device="cpu", dtype=torch.float32)
        self.head = nn.Linear(32, 2, device="cpu", dtype=torch.float32)

    def forward(self, images):
        return self.head(torch.relu(self.hidden(images.flatten(1))))


def make_model(kind, seed):
    if kind not in ("cnn", "mlp"):
        raise ValueError("kind must be cnn or mlp")
    valid_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return VisionCNN() if kind == "cnn" else VisionMLP()


def serialize_state(model):
    return {name: values.detach().cpu().tolist() for name, values in model.state_dict().items()}


def load_model(kind, state):
    model = make_model(kind, 0)
    model.load_state_dict({name: torch.tensor(value, dtype=torch.float32) for name, value in state.items()})
    return model.eval()


@torch.no_grad()
def evaluate(model, images, labels):
    logits = model(images)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite evaluation logits")
    predicted = logits.argmax(1)
    matrix = torch.bincount(2 * labels + predicted, minlength=4).reshape(2, 2)
    return {"loss": nn.functional.cross_entropy(logits, labels).item(),
            "accuracy": (predicted == labels).double().mean().item(),
            "confusion_true_rows": matrix.tolist()}


def run_comparison(*, seeds=(59, 60), epochs=20, train_size=256, val_size=128,
                   test_size=128, progress=None):
    """Equal splits/order, Adam rate and updates; deliberately not parameter-matched.

    The final epoch is fixed in advance. Validation curves are descriptive;
    no early stopping or best-seed selection. Test sweeps never affect fitting.
    """
    epochs = integer(epochs, "epochs")
    for value, name in ((train_size, "train_size"), (val_size, "val_size"), (test_size, "test_size")):
        integer(value, name, 2)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    for seed in seeds:
        valid_seed(seed)
        if seed > (2**32 - 2010) // 10:
            raise ValueError("seed is too large for the derived experiment seeds")
    runs = []
    total = len(seeds) * 2 * epochs
    for seed_index, seed in enumerate(seeds):
        data_seed = 1000 + 10 * seed
        train_x, train_y = bars(train_size, data_seed)
        val_x, val_y = bars(val_size, data_seed + 1)
        test_x, test_y = bars(test_size, data_seed + 2)
        run = {"seed": seed, "dataset_seeds": [data_seed, data_seed + 1, data_seed + 2], "models": {}}
        for model_index, kind in enumerate(("cnn", "mlp")):
            model = make_model(kind, seed + 2000)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            order_rng = torch.Generator(device="cpu").manual_seed(seed + 2001)
            history, checkpoints = [], []
            for epoch in range(epochs + 1):
                if epoch:
                    model.train()
                    for batch in torch.randperm(train_size, generator=order_rng).split(32):
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.cross_entropy(model(train_x[batch]), train_y[batch])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training loss")
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise FloatingPointError("nonfinite parameter gradient")
                        optimizer.step()
                model.eval()
                train_metrics = evaluate(model, train_x, train_y)
                val_metrics = evaluate(model, val_x, val_y)
                history.append({"epoch": epoch, "train_loss": train_metrics["loss"],
                                "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]})
                if ((kind == "cnn" and epoch in {0, 1, 2, 5, 10, epochs})
                        or (kind == "mlp" and epoch == epochs)):
                    checkpoints.append({"epoch": epoch, "state": serialize_state(model)})
                if progress and epoch:
                    progress((seed_index * 2 * epochs + model_index * epochs + epoch) / total,
                             f"seed {seed} · {kind.upper()} · epoch {epoch}/{epochs}")
            shifts, noises = [], []
            for dx in range(-8, 9, 2):
                moved = torch.from_numpy(shift_image(test_x.numpy(), dx))
                metrics = evaluate(model, moved, test_y)
                with torch.no_grad():
                    probability = torch.softmax(model(moved[:1]), 1)[0, test_y[0]].item()
                shifts.append({"dx": dx, **metrics, "sample_true_class_probability": probability})
            perturbation = torch.from_numpy(np.random.default_rng(data_seed + 3).normal(size=test_x.shape).astype(np.float32))
            for extra_std in (0., 0.2, 0.5, 1.0):
                noises.append({"additional_noise_std": extra_std,
                               **evaluate(model, test_x + extra_std * perturbation, test_y)})
            with torch.no_grad():
                probabilities = torch.softmax(model(test_x), 1).tolist()
            run["models"][kind] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "history": history, "checkpoints": checkpoints,
                "test": evaluate(model, test_x, test_y), "shift_sweep": shifts,
                "noise_sweep": noises, "test_probabilities": probabilities,
            }
        runs.append(run)
    import matplotlib
    import plotly
    import streamlit
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Spatial weight sharing may give a compact CNN different translation sensitivity from a flattened MLP on the same bar data.",
        "configuration": {"seeds": list(seeds), "epochs": epochs, "train_size": train_size,
                          "val_size": val_size, "test_size": test_size, "batch_size": 32,
                          "optimizer": "Adam, lr=0.01; other PyTorch defaults", "base_noise_std": 0.12,
                          "selection": "fixed final epoch for both models; no hyperparameter or seed selection",
                          "data": "synthetic 32x32; bars 15x3 or 3x15; centers in 14..17; balanced labels",
                          "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                          "shift": "zero fill, horizontal -8..8 step 2; outer shifts may clip bars",
                          "noise": "same paired perturbations, extra std 0, 0.2, 0.5, 1; no clipping"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                     "matplotlib": matplotlib.__version__, "plotly": plotly.__version__, "streamlit": streamlit.__version__},
        "runs": runs,
        "interpretation_status": "author review required",
        "interpretation_candidate": "Compare paired clean and shifted results seed by seed; differing parameter counts and untuned shared hyperparameters prevent an architecture ranking.",
        "limitations": "Two synthetic seeds by default; narrow center support, easy clean labels; no parameter/compute matching, tuning, real data, calibration or deployment benchmark. Noise uses one paired perturbation per test image and shifts can crop signal.",
    }


def selected_example(report, seed_index=0, sample=0):
    run = report["runs"][seed_index]
    images, labels = bars(report["configuration"]["test_size"], run["dataset_seeds"][2])
    sample = integer(sample, "sample", 0)
    if sample >= len(images):
        raise ValueError("sample is outside the test split")
    return images[sample:sample + 1], int(labels[sample])


def translation_predictions(report, sample=None):
    """Actual per-image confidence sweep, selecting by label rather than model error."""
    run = report["runs"][0]
    images, labels = bars(report["configuration"]["test_size"], run["dataset_seeds"][2])
    if sample is None:
        sample = int(torch.nonzero(labels == 0)[0, 0])
    image, label = selected_example(report, sample=sample)
    results = {}
    for kind in ("cnn", "mlp"):
        model = load_model(kind, run["models"][kind]["checkpoints"][-1]["state"])
        values = []
        for dx in range(-8, 9, 2):
            moved = torch.from_numpy(shift_image(image.numpy(), dx))
            with torch.no_grad():
                probability = torch.softmax(model(moved), 1)[0, label].item()
            values.append({"dx": dx, "true_class_probability": probability})
        results[kind] = values
    return {"sample": sample, "label": label, "results": results,
            "selection": "first vertical test bar by label, before inspecting predictions" if sample == int(torch.nonzero(labels == 0)[0, 0]) else "user-selected test example"}
