"""Learn vertical/horizontal bars on CPU; write a dated, nonvisual study record."""

import argparse
from datetime import datetime, timezone
import json
from numbers import Real
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install the canonical extra: python -m pip install -e ".[dev,deep-learning]"') from exc

from from_scratch import integer


def valid_seed(seed):
    seed = integer(seed, "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed must be < 2**32")
    return seed


def make_data(n, seed, noise: float = 0.1):
    """Synthetic 16x16 grayscale bars: class 0 vertical, class 1 horizontal.

    Both classes have equal signal area/intensity and the same center support.
    Labels are balanced up to one sample and permuted before drawing centers.
    """
    n = integer(n, "n", 2)
    seed = valid_seed(seed)
    if isinstance(noise, (bool, np.bool_)) or not isinstance(noise, Real):
        raise ValueError("noise must be a finite nonnegative number")
    noise = float(noise)
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be a finite nonnegative number")
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.arange(n) % 2)
    with np.errstate(over="raise", invalid="raise"):
        images = rng.normal(0, noise, (n, 1, 16, 16)).astype(np.float32)
        for i, label in enumerate(labels):
            row, col = rng.integers(5, 11, size=2)
            if label == 0:
                images[i, 0, row - 4:row + 5, col - 1:col + 2] += 1
            else:
                images[i, 0, row - 1:row + 2, col - 4:col + 5] += 1
    return torch.from_numpy(images), torch.from_numpy(labels.astype(np.int64))


class SmallCNN(nn.Module):
    """Conv3 -> ReLU -> MaxPool2 -> spatial mean -> two raw logits."""

    def __init__(self, seed=61):
        super().__init__()
        seed = valid_seed(seed)
        # Seed initialization without consuming the caller's global CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, device="cpu", dtype=torch.float32)
            self.pool = nn.MaxPool2d(2)
            self.head = nn.Linear(8, 2, device="cpu", dtype=torch.float32)

    def feature_maps(self, images):
        if (not isinstance(images, torch.Tensor) or images.ndim != 4
                or images.shape[0] < 1 or tuple(images.shape[1:]) != (1, 16, 16)
                or images.device.type != "cpu" or images.dtype != torch.float32
                or not torch.isfinite(images).all()):
            raise ValueError("images must be finite nonempty CPU float32 tensors of shape (N,1,16,16)")
        activated = torch.relu(self.conv(images))
        return activated, self.pool(activated)

    def forward(self, images):
        _, pooled = self.feature_maps(images)
        return self.head(pooled.mean(dim=(-2, -1)))


def run_study(*, seed=59, train_size=256, test_size=128, epochs=30,
              batch_size=32, learning_rate=0.01, noise=0.1):
    seed = valid_seed(seed)
    if seed >= 2**32 - 3:
        raise ValueError("study seed must be < 2**32 - 3 (three derived seeds are reserved)")
    epochs = integer(epochs, "epochs")
    batch_size = integer(batch_size, "batch_size")
    if (isinstance(learning_rate, (bool, np.bool_)) or not isinstance(learning_rate, Real)
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be a finite positive number")
    train_x, train_y = make_data(train_size, seed, noise)
    test_x, test_y = make_data(test_size, seed + 1, noise)
    model = SmallCNN(seed + 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    with torch.no_grad():
        initial_loss = loss_fn(model(train_x), train_y).item()
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(train_size, generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("missing or non-finite parameter gradient")
            optimizer.step()
        with torch.no_grad():
            losses.append(loss_fn(model(train_x), train_y).item())
    model.eval()
    with torch.no_grad():
        train_logits = model(train_x)
        # Test labels do not influence fitting, stopping or configuration selection.
        test_logits = model(test_x)
        if not torch.isfinite(train_logits).all() or not torch.isfinite(test_logits).all():
            raise FloatingPointError("non-finite final logits")
        predicted = test_logits.argmax(dim=1)
        confusion = torch.bincount(2 * test_y + predicted, minlength=4).reshape(2, 2)
        activated, pooled = model.feature_maps(train_x[:1])
        majority_class = torch.bincount(train_y, minlength=2).argmax()
        results = {
            "initial_train_cross_entropy": initial_loss,
            "epoch_end_train_cross_entropy": losses,
            "final_train_accuracy": (train_logits.argmax(1) == train_y).float().mean().item(),
            "test_cross_entropy": loss_fn(test_logits, test_y).item(),
            "test_accuracy": (predicted == test_y).float().mean().item(),
            "test_confusion_true_rows_predicted_columns": confusion.tolist(),
            "train_majority_baseline_test_accuracy": (test_y == majority_class).float().mean().item(),
            "train_class_counts": torch.bincount(train_y, minlength=2).tolist(),
            "test_class_counts": torch.bincount(test_y, minlength=2).tolist(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "activation_shape": list(activated.shape), "pooled_shape": list(pooled.shape),
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Local learned filters and spatial aggregation can fit bar orientation and transfer to independent draws from the same synthetic generator.",
        "configuration": {
            "seed": seed, "train_seed": seed, "test_seed": seed + 1,
            "model_seed": seed + 2, "shuffle_seed": seed + 3,
            "train_size": train_size, "test_size": test_size, "epochs": epochs,
            "batch_size": batch_size, "learning_rate": learning_rate, "noise_std": noise,
            "optimizer": "Adam; other settings use PyTorch defaults",
            "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "architecture": "Conv2d(1,8,3,padding=1), ReLU, MaxPool2d(2), spatial mean, Linear(8,2)",
            "selection": "fixed configuration and epoch budget; no validation tuning or test selection",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "results": results,
        "interpretation_candidate": "Assess loss reduction and same-generator holdout accuracy against the majority baseline; these do not establish robustness or realistic image performance.",
        "interpretation_status": "author review required",
        "limitations": "Synthetic bars, one seed, one architecture, no real images, shift/rotation/noise sweep, calibration or runtime benchmark. Individual learned filters have no assigned semantics.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, help="New JSON path; existing records are never overwritten")
    parser.add_argument("--generate-visuals", action="store_true", help="Run the separate 32x32 CNN/MLP visual experiment and export collection")
    parser.add_argument("--visual-output-dir", type=Path, help="New/empty export directory; default is a dated ignored folder")
    parser.add_argument("--visual-epochs", type=int, default=20, help="Epoch budget for the separate visual experiment")
    parser.add_argument("--visual-report", type=Path, help="Reuse an existing visual experiment JSON without training again")
    args = parser.parse_args()
    if args.generate_visuals:
        from visualizations import generate_collection
        # The visual protocol has its own fixed 20-epoch default, distinct from
        # the original analytical study's 30 epochs and 16x16 images.
        measured = json.loads(args.visual_report.read_text(encoding="utf-8")) if args.visual_report else None
        output, _ = generate_collection(args.visual_output_dir, report=measured, seeds=(args.seed, args.seed + 1),
                                        epochs=args.visual_epochs, progress=lambda value, label: print(label, flush=True))
        print(f"Visual collection: {output}")
        return
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new path to preserve historical records")
    torch.set_num_threads(1)
    try:
        report = run_study(seed=args.seed, epochs=args.epochs)
    except ValueError as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(__file__).resolve().parent / "outputs" / f"cnn_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
