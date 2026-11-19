"""Compare CPU RNN/LSTM/GRU on synthetic delayed-bit recall; no external data."""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install the canonical extra: python -m pip install -e ".[dev,deep-learning]"') from exc

from from_scratch import finite_real, integer, valid_seed


CELLS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


def make_data(n, length, seed):
    """IID synthetic sequences: y is encoded ONLY at t=0 in feature 0.

    Feature 1 marks t=0; subsequent feature-0 values are N(0,1) distractors.
    Labels are independent Bernoulli(0.5), not forced to balance.
    """
    n, length = integer(n, "n", 2), integer(length, "length", 2)
    rng = np.random.default_rng(valid_seed(seed))
    y = rng.integers(0, 2, size=n).astype(np.float32)
    x = np.zeros((n, length, 2), dtype=np.float32)
    x[:, :, 0] = rng.normal(size=(n, length))
    x[:, 0, 0] = 2 * y - 1
    x[:, 0, 1] = 1
    return torch.from_numpy(x), torch.from_numpy(y)


class SequenceClassifier(nn.Module):
    """One unidirectional layer and one raw logit, with no padding/dropout."""

    def __init__(self, kind, hidden=16, seed=63):
        super().__init__()
        if kind not in CELLS:
            raise ValueError("kind must be rnn, lstm or gru")
        hidden, seed = integer(hidden, "hidden"), valid_seed(seed)
        self.kind = kind
        # Keep initialization reproducible without consuming the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.recurrent = CELLS[kind](2, hidden, batch_first=True, device="cpu", dtype=torch.float32)
            self.head = nn.Linear(hidden, 1, device="cpu", dtype=torch.float32)

    def encode(self, x, state=None):
        if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] < 1
                or x.shape[1] < 1 or x.shape[2] != 2 or x.device.type != "cpu"
                or x.dtype != torch.float32 or not torch.isfinite(x).all()):
            raise ValueError("x must be finite CPU float32 with nonempty shape (N,T,2)")
        return self.recurrent(x, state)

    def forward(self, x):
        _, state = self.encode(x)
        hidden = state[0] if self.kind == "lstm" else state
        return self.head(hidden[-1]).squeeze(-1)


def validate_labels(x, y):
    if (not isinstance(y, torch.Tensor) or y.shape != (x.shape[0],)
            or y.device.type != "cpu" or y.dtype != torch.float32
            or not torch.all((y == 0) | (y == 1))):
        raise ValueError("y must be binary CPU float32 with shape (N,)")


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    validate_labels(x, y)
    if not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite evaluation logits")
    return {"bce": nn.functional.binary_cross_entropy_with_logits(logits, y).item(),
            "accuracy": ((logits >= 0) == y.bool()).float().mean().item()}


def fit_model(kind, train, validation, *, hidden=16, seed=63, shuffle_seed=64,
              epochs=20, batch_size=64, learning_rate=0.005, clip_norm=1.0):
    """Select an epoch on validation BCE; test samples cannot enter this function."""
    epochs, batch_size = integer(epochs, "epochs"), integer(batch_size, "batch_size")
    learning_rate = finite_real(learning_rate, "learning_rate", positive=True)
    clip_norm = finite_real(clip_norm, "clip_norm", positive=True)
    generator = torch.Generator(device="cpu").manual_seed(valid_seed(shuffle_seed))
    model = SequenceClassifier(kind, hidden, seed)
    train_x, train_y = train
    val_x, val_y = validation
    initial = evaluate(model, train_x, train_y)
    evaluate(model, val_x, val_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_bce, best_state, best_epoch = float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        max_norm, clipped = 0.0, 0
        order = torch.randperm(len(train_y), generator=generator)
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True).item()
            max_norm = max(max_norm, norm)
            clipped += int(norm > clip_norm)
            optimizer.step()
        metrics = evaluate(model, val_x, val_y)
        history.append({"epoch": epoch, "train_bce": evaluate(model, train_x, train_y)["bce"],
                        "validation_bce": metrics["bce"], "validation_accuracy": metrics["accuracy"],
                        "max_preclip_gradient_norm": max_norm, "clipped_batches": clipped})
        if metrics["bce"] < best_bce:
            best_bce, best_epoch = metrics["bce"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, {"initial_train_bce": initial["bce"], "selected_epoch": best_epoch,
                   "selected_train": evaluate(model, train_x, train_y),
                   "selected_validation": evaluate(model, val_x, val_y), "history": history}


def run_study(*, seed=60, cells=("rnn", "lstm", "gru"), lengths=(10, 50),
              hidden=16, epochs=20, train_size=1024, validation_size=256,
              test_size=512, batch_size=64, learning_rate=0.005, clip_norm=1.0):
    seed = valid_seed(seed)
    if seed > 2**32 - 5:
        raise ValueError("seed must leave room for four derived seeds")
    cells, lengths = tuple(cells), tuple(lengths)
    if not cells or len(set(cells)) != len(cells) or any(cell not in CELLS for cell in cells):
        raise ValueError("cells must be nonempty, unique choices from rnn, lstm, gru")
    lengths = tuple(integer(length, "length", 2) for length in lengths)
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("lengths must be nonempty and unique")
    hidden, epochs = integer(hidden, "hidden"), integer(epochs, "epochs")
    train_size = integer(train_size, "train_size", 2)
    validation_size = integer(validation_size, "validation_size", 2)
    test_size = integer(test_size, "test_size", 2)
    batch_size = integer(batch_size, "batch_size")
    learning_rate = finite_real(learning_rate, "learning_rate", positive=True)
    clip_norm = finite_real(clip_norm, "clip_norm", positive=True)
    rows, controls = [], []
    for length in lengths:
        train = make_data(train_size, length, seed)
        validation = make_data(validation_size, length, seed + 1)
        # Reusing seeds across lengths pairs labels, not identical noise prefixes.
        test_x, test_y = make_data(test_size, length, seed + 2)
        prior = train[1].mean().item()
        prior = np.clip(prior, 1e-7, 1 - 1e-7)
        prior_bce = -(test_y.numpy() * np.log(prior) + (1 - test_y.numpy()) * np.log1p(-prior)).mean()
        controls.append({
            "length": length,
            "train_positive_rate": train[1].mean().item(),
            "validation_positive_rate": validation[1].mean().item(),
            "test_positive_rate": test_y.mean().item(),
            "train_majority_test_accuracy": ((prior >= 0.5) == test_y.bool()).float().mean().item(),
            "train_prior_test_bce": float(prior_bce),
            "first_step_sign_test_accuracy": ((test_x[:, 0, 0] > 0) == test_y.bool()).float().mean().item(),
            "last_step_sign_test_accuracy": ((test_x[:, -1, 0] > 0) == test_y.bool()).float().mean().item(),
        })
        for kind in cells:
            model, training = fit_model(kind, train, validation, hidden=hidden,
                                        seed=seed + 3, shuffle_seed=seed + 4, epochs=epochs,
                                        batch_size=batch_size, learning_rate=learning_rate, clip_norm=clip_norm)
            rows.append({"cell": kind, "length": length,
                         "parameters": sum(p.numel() for p in model.parameters()),
                         **training, "test": evaluate(model, test_x, test_y)})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Gated recurrence may preserve an early bit across distractors better than a tanh RNN; this fixed budget can also fail for every architecture.",
        "configuration": {
            "seed": seed, "train_seed": seed, "validation_seed": seed + 1, "test_seed": seed + 2,
            "model_seed": seed + 3, "shuffle_seed": seed + 4, "cells": list(cells), "lengths": list(lengths),
            "hidden": hidden, "epochs": epochs, "train_size": train_size, "validation_size": validation_size,
            "test_size": test_size, "batch_size": batch_size, "learning_rate": learning_rate,
            "clip_norm": clip_norm, "optimizer": "Adam; other settings use PyTorch defaults",
            "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "initialization": "PyTorch default; no custom forget bias",
            "selection": "Earliest minimum validation BCE among epoch ends, separately per cell/length; all predefined cells reported, no test-based selection",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "processor": platform.processor()},
        "controls": controls, "results": rows,
        "interpretation_candidate": "Compare held-out recall with train-majority and last-step controls; successful same-generator recall or failure under this budget does not rank architectures generally.",
        "interpretation_status": "author review required",
        "limitations": "Synthetic IID sequences, one seed per run, fixed hidden size rather than matched parameters/compute, no hyperparameter search, real time series, temporal-gradient measurement, latency or memory benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=["all", *CELLS], default="all")
    parser.add_argument("--lengths", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--seed", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--output", type=Path, help="New JSON path; existing records are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output exists; choose a new path to preserve the historical record")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        report = run_study(seed=args.seed, cells=tuple(CELLS) if args.cell == "all" else (args.cell,),
                           lengths=args.lengths, hidden=args.hidden, epochs=args.epochs)
    except ValueError as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path(__file__).resolve().parent / "outputs" / f"recall_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for row in report["results"]:
        print(f"{row['cell']:4} T={row['length']:2} params={row['parameters']:4} "
              f"epoch={row['selected_epoch']:2} val_BCE={row['selected_validation']['bce']:.6f} "
              f"test_BCE={row['test']['bce']:.6f} test_acc={row['test']['accuracy']:.4f}")
    print(f"Saved: {output}\nInterpretation: author review required")


if __name__ == "__main__":
    main()
