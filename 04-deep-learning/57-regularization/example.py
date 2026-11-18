"""Six CPU MLP configurations on synthetic moons plus 18 irrelevant features.

JSON goes to stdout; no plots, downloads or saved model artifacts.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
except ImportError as error:
    raise SystemExit('PyTorch is required: python -m pip install -e ".[deep-learning]"') from error


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


@dataclass(frozen=True)
class Configuration:
    name: str
    dropout: float = 0.0
    weight_decay: float = 0.0
    batch_norm: bool = False
    early_stopping: bool = False

    def __post_init__(self):
        if not self.name or not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("name is required and dropout must be in [0, 1).")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        if not isinstance(self.batch_norm, bool) or not isinstance(self.early_stopping, bool):
            raise ValueError("batch_norm and early_stopping must be bools.")


CONFIGURATIONS = (
    Configuration("baseline"),
    Configuration("dropout", dropout=0.3),
    Configuration("weight_decay", weight_decay=0.01),
    Configuration("batch_norm", batch_norm=True),
    Configuration("early_stopping", early_stopping=True),
    Configuration("combined", dropout=0.3, weight_decay=0.01,
                  batch_norm=True, early_stopping=True),
)


def fingerprint(values):
    array = values.detach().cpu().contiguous().numpy() if torch.is_tensor(values) else values
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_data(n=1000, seed=57):
    integer(n, "n", 100)
    integer(seed, "seed", 0)
    if n % 10:
        raise ValueError("n must be a multiple of 10 for the 60/20/20 split.")
    signal, labels = make_moons(n_samples=n, noise=0.30, random_state=seed)
    noise = np.random.default_rng(seed + 1).normal(size=(n, 18))
    features = np.column_stack((signal, noise))
    dev, test = train_test_split(np.arange(n), test_size=0.2, stratify=labels, random_state=seed)
    train, validation = train_test_split(dev, test_size=0.25, stratify=labels[dev], random_state=seed)
    scaler = StandardScaler().fit(features[train])
    partitions = {}
    ids = {"train": train, "validation": validation, "test": test}
    for name, indices in ids.items():
        partitions[name] = (torch.tensor(scaler.transform(features[indices]), dtype=torch.float32),
                            torch.tensor(labels[indices], dtype=torch.long))
    return partitions, ids, scaler


def batch_indices(order, batch_size):
    """Retain every example; merge a final singleton so 2D BatchNorm can train."""
    integer(batch_size, "batch_size", 2)
    if order.ndim != 1 or len(order) < 2:
        raise ValueError("order must be a 1D tensor with at least two examples.")
    batches = list(order.split(batch_size))
    if len(batches[-1]) == 1:
        tail = batches.pop()
        batches[-1] = torch.cat((batches[-1], tail))
    return batches


class MLP(nn.Module):
    def __init__(self, config, seed, input_dim=20):
        super().__init__()
        integer(seed, "seed", 0)
        integer(input_dim, "input_dim")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.first = nn.Linear(input_dim, 64)
            self.second = nn.Linear(64, 64)
            self.output = nn.Linear(64, 2)
        self.norm1 = nn.BatchNorm1d(64) if config.batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(64) if config.batch_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.norm1(self.first(x))))
        x = self.dropout(torch.relu(self.norm2(self.second(x))))
        return self.output(x)


def linear_parameters(model):
    return torch.cat([p.detach().reshape(-1) for layer in (model.first, model.second, model.output)
                      for p in layer.parameters()])


def make_optimizer(model, decay, lr=0.001):
    if not np.isfinite(decay) or decay < 0 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("decay must be nonnegative and lr positive, both finite.")
    matrices, other = [], []
    for parameter in model.parameters():
        (matrices if parameter.ndim == 2 else other).append(parameter)
    return torch.optim.AdamW([{"params": matrices, "weight_decay": decay},
                              {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.999), eps=1e-8, foreach=False)


class EarlyStopping:
    """Absolute best checkpoint; min_delta controls patience resets separately."""
    def __init__(self, patience=15, min_delta=1e-4):
        integer(patience, "patience")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative.")
        self.patience, self.min_delta = patience, min_delta
        self.best_loss = self.progress_loss = float("inf")
        self.best_epoch, self.bad_checks, self.state = 0, 0, None

    def observe(self, loss, epoch, model):
        integer(epoch, "epoch")
        if not np.isfinite(loss):
            raise ValueError("Validation loss must be finite.")
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = float(loss), epoch
            self.state = copy.deepcopy(model.state_dict())
        if loss < self.progress_loss - self.min_delta:
            self.progress_loss, self.bad_checks = float(loss), 0
        else:
            self.bad_checks += 1
        return self.bad_checks >= self.patience

    def restore(self, model):
        if self.state is None:
            raise ValueError("No validation checkpoint has been observed.")
        model.load_state_dict(self.state)


@torch.no_grad()
def evaluate(model, features, labels):
    model.eval()
    logits = model(features)
    loss = nn.functional.cross_entropy(logits, labels).item()
    if not np.isfinite(loss):
        raise ValueError("Evaluation loss must be finite.")
    return {"loss": loss, "accuracy": (logits.argmax(1) == labels).double().mean().item()}


def train_configuration(config, train, validation, orders, *, seed, batch_size,
                        patience, min_delta):
    model = MLP(config, seed)
    initial_hash = fingerprint(linear_parameters(model))
    optimizer = make_optimizer(model, config.weight_decay)
    monitor = EarlyStopping(patience, min_delta)
    history, steps = [], 0
    started = perf_counter()
    for epoch, order in enumerate(orders, 1):
        model.train()
        for indices in batch_indices(order, batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train[0][indices]), train[1][indices])
            if not torch.isfinite(loss):
                raise ValueError("Training loss must be finite.")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("Gradients must be finite.")
            optimizer.step()
            steps += 1
        history.append({"epoch": epoch, "train": evaluate(model, *train),
                        "validation": evaluate(model, *validation)})
        stop = monitor.observe(history[-1]["validation"]["loss"], epoch, model)
        if config.early_stopping and stop:
            break
    selected_epoch = len(history)
    if config.early_stopping:
        monitor.restore(model)
        selected_epoch = monitor.best_epoch
    result = {
        "configuration": asdict(config), "initial_linear_sha256": initial_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history), "steps": steps,
        "stopped_early": config.early_stopping and len(history) < len(orders),
        "checkpoint_policy": "best_validation" if config.early_stopping else "last_epoch",
        "selected_epoch": selected_epoch,
        "best_observed_epoch": monitor.best_epoch, "best_observed_loss": monitor.best_loss,
        "selected_train": evaluate(model, *train), "selected_validation": evaluate(model, *validation),
        "elapsed_seconds": perf_counter() - started, "history": history,
    }
    return result, model


def run_comparison(*, epochs=120, batch_size=64, n=1000, seed=57, patience=15, min_delta=1e-4):
    integer(epochs, "epochs")
    integer(batch_size, "batch_size", 2)
    integer(patience, "patience")
    if not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative.")
    data, ids, scaler = make_data(n, seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    orders = [torch.randperm(len(ids["train"]), generator=generator) for _ in range(epochs)]
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    runs, winner_model, winner_index = [], None, None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        for config in CONFIGURATIONS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 4)
                result, model = train_configuration(
                    config, data["train"], data["validation"], orders, seed=seed + 3,
                    batch_size=batch_size, patience=patience, min_delta=min_delta)
            runs.append(result)
            # Test data is never passed to a training or candidate-selection function.
            if winner_model is None or result["selected_validation"]["loss"] < runs[winner_index]["selected_validation"]["loss"]:
                winner_model, winner_index = model, len(runs) - 1
        final_test = evaluate(winner_model, *data["test"])
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1},
        "hypothesis": "Activation noise, parameter shrinkage, normalization and validation "
                      "checkpoint selection can change fitting and held-out loss; benefits are not guaranteed.",
        "configuration": {"data": "synthetic make_moons(noise=0.30) plus 18 Gaussian noise features",
                          "n": n, "splits": {key: len(value) for key, value in ids.items()},
                          "class_counts": {key: torch.bincount(value[1], minlength=2).tolist()
                                           for key, value in data.items()},
                          "seed_data_split": seed, "seed_noise": seed + 1,
                          "seed_batches": seed + 2, "seed_model": seed + 3, "seed_dropout": seed + 4,
                          "split_sha256": {key: fingerprint(value) for key, value in ids.items()},
                          "batch_orders_sha256": fingerprint(torch.stack(orders)),
                          "scaler_training_mean": scaler.mean_.tolist(),
                          "architecture": "20-64-64-2; Linear -> optional BN -> ReLU -> Dropout twice",
                          "optimizer": "AdamW", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
                          "decay_policy": "2D linear weight matrices only; biases and BN affine excluded",
                          "batch_norm": {"eps": 1e-5, "momentum": 0.1, "track_running_stats": True},
                          "epochs_max": epochs, "batch_size": batch_size,
                          "singleton_policy": "merge final singleton, retain every example",
                          "patience": patience, "min_delta": min_delta,
                          "loss": "mean cross entropy excluding weight decay; both metrics in eval mode",
                          "checkpoint_policy": "last for non-ES; absolute best validation for ES",
                          "selection": "minimum selected-checkpoint validation loss; first configuration breaks ties",
                          "scheduler": None, "clipping": None},
        "runs": runs, "selected_configuration": runs[winner_index]["configuration"]["name"],
        "final_test": final_test, "test_evaluations": 1,
        "interpretation_candidate": "Compare only these untuned configurations; checkpoint policy "
                                    "and training effort differ, and BatchNorm adds affine parameters.",
        "interpretation_status": "author review required",
        "limitation": "One synthetic split/seed; not compute-matched, tuned, real-data or production evidence. "
                      "One selected-model test evaluation per invocation; repeated test-guided tuning invalidates its role. "
                      "Timing includes evaluation and checkpoint copies and is not a performance benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        report = run_comparison(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
