"""An explicit CPU training loop on a synthetic nonlinear binary task."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise SystemExit('Install from the repository root: python -m pip install -e ".[dev,deep-learning]"') from error


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_batch(features: torch.Tensor, targets: torch.Tensor) -> None:
    """Enforce the deliberately narrow CPU, float32, single-label contract."""
    if features.ndim != 2 or min(features.shape) < 1:
        raise ValueError("features must have nonempty shape (N, D)")
    if targets.shape != (len(features), 1):
        raise ValueError("targets must have shape (N, 1)")
    if features.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError("features and targets must use float32")
    if features.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("this educational loop supports CPU tensors only")
    if not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("features and targets must be finite")
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError("targets must be binary, 0 or 1")


def fit_standardizer(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    validate_batch(features, torch.zeros((len(features), 1), dtype=torch.float32))
    return features.mean(dim=0), features.std(dim=0, unbiased=False).clamp_min(1e-6)


def prepare_data(seed: int = 42) -> tuple[dict, dict, dict]:
    """Generate, split, then fit preprocessing using training observations only."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(1800, 2, generator=generator)
    noise = 0.3 * torch.randn(1800, generator=generator)
    signal = features[:, 0] * features[:, 1] + 0.4 * features[:, 0] - 0.2 * features[:, 1]
    targets = (signal + noise > 0).float().unsqueeze(1)
    order = torch.randperm(1800, generator=torch.Generator().manual_seed(seed + 1))
    indices = {"train": order[:1200], "validation": order[1200:1500], "test": order[1500:]}
    mean, scale = fit_standardizer(features[indices["train"]])
    partitions = {
        name: TensorDataset((features[idx] - mean) / scale, targets[idx])
        for name, idx in indices.items()
    }
    return partitions, {"mean": mean.tolist(), "scale": scale.tolist()}, indices


def make_loader(dataset: TensorDataset, batch_size: int, *, shuffle: bool = False,
                seed: int = 44) -> DataLoader:
    positive_integer(batch_size, "batch_size")
    if len(dataset) == 0:
        raise ValueError("dataset must not be empty")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )


def batch_loss(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> tuple:
    validate_batch(features, targets)
    logits = model(features)
    if logits.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError("model must produce finite logits with shape (N, 1)")
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if not torch.isfinite(loss):
        raise ValueError("loss is not finite")
    return logits, loss


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer) -> dict:
    model.train()
    total_loss, samples, steps = 0.0, 0, 0
    for features, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        _, loss = batch_loss(model, features, targets)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("gradient is not finite")
        optimizer.step()
        # The last batch may be smaller; weight its mean by actual sample count.
        total_loss += loss.item() * len(features)
        samples += len(features)
        steps += 1
    if samples == 0:
        raise ValueError("loader yielded no samples")
    return {"online_loss": total_loss / samples, "samples": samples, "steps": steps}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Leave the model in eval mode; neither parameters nor buffers are updated."""
    model.eval()
    total_loss, samples = 0.0, 0
    tp = fp = tn = fn = positives = 0
    for features, targets in loader:
        logits, loss = batch_loss(model, features, targets)
        predicted, actual = logits >= 0, targets == 1
        total_loss += loss.item() * len(features)
        samples += len(features)
        positives += int(actual.sum())
        tp += int((predicted & actual).sum())
        fp += int((predicted & ~actual).sum())
        tn += int((~predicted & ~actual).sum())
        fn += int((~predicted & actual).sum())
    if samples == 0:
        raise ValueError("loader yielded no samples")
    return {"loss": total_loss / samples, "accuracy": (tp + tn) / samples,
            "samples": samples, "positives": positives,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def fit(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
        *, epochs: int, learning_rate: float) -> dict:
    """Select an epoch by validation BCE, then restore that parameter snapshot."""
    positive_integer(epochs, "epochs")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, best_epoch, best_validation = math.inf, None, 0, None
    history = []
    for epoch in range(1, epochs + 1):
        training = train_one_epoch(model, train_loader, optimizer)
        validation = evaluate(model, validation_loader)
        history.append({"epoch": epoch, **training, "validation_loss": validation["loss"]})
        if validation["loss"] < best_loss:
            best_loss, best_epoch, best_validation = validation["loss"], epoch, validation
            # state_dict() alone holds references to tensors that later updates mutate.
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError("no finite validation checkpoint was selected")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation": best_validation, "history": history,
            "optimizer_steps": sum(row["steps"] for row in history)}


def run_study(*, seed: int = 42, epochs: int = 60, batch_size: int = 64,
              learning_rate: float = 1e-3) -> dict:
    positive_integer(epochs, "epochs")
    positive_integer(batch_size, "batch_size")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    partitions, preprocessing, indices = prepare_data(seed)
    validation_loader = make_loader(partitions["validation"], batch_size)
    candidates = {}
    selected_name, selected_model, selected_loss = None, None, math.inf
    for name in ("linear", "mlp"):
        # Isolate initialization from data and shuffle RNGs; restore caller RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 3)
            model = nn.Linear(2, 1) if name == "linear" else nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1),
            )
        training_loader = make_loader(partitions["train"], batch_size, shuffle=True, seed=seed + 2)
        result = fit(model, training_loader, validation_loader,
                     epochs=epochs, learning_rate=learning_rate)
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["restored_train"] = evaluate(model, make_loader(partitions["train"], batch_size))
        candidates[name] = result
        # Ties favor the earlier candidate, linear. No test metrics enter this decision.
        if result["validation"]["loss"] < selected_loss:
            selected_name, selected_loss = name, result["validation"]["loss"]
            selected_model = model
    test_metrics = evaluate(selected_model, make_loader(partitions["test"], batch_size))
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": "synthetic nonlinear binary classification; no external dataset",
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "configuration": {"seed": seed, "epochs": epochs, "batch_size": batch_size,
                          "learning_rate": learning_rate, "optimizer": "Adam (default betas, no decay)",
                          "device": "cpu", "dtype": "float32", "workers": 0,
                          "threads": torch.get_num_threads(), "logit_threshold": 0.0,
                          "partition_sizes": {k: len(v) for k, v in indices.items()}},
        "preprocessing": preprocessing, "candidates": candidates,
        "selected_model": selected_name, "selected_test": test_metrics,
        "hypothesis": "A nonlinear MLP can fit the interaction signal better than a linear boundary under this fixed budget.",
        "interpretation_candidate": "Compare restored validation losses before interpreting the selected test score; author review required.",
        "limitation": "One synthetic seed and fixed rate/budget; no architecture or optimizer ranking, production or cross-platform claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a fresh filename")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = run_study(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                       learning_rate=args.learning_rate)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
