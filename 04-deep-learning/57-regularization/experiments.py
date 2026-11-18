"""Bounded real CPU training for genuine 2D boundaries; separate from the Day 57 analytical run."""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import platform
from time import perf_counter
from threading import RLock

import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from example import (CONFIGURATIONS, EarlyStopping, MLP, batch_indices, evaluate,
                     fingerprint, integer, linear_parameters, make_optimizer)


TRAINING_LOCK = RLock()


@contextmanager
def cpu_context():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def visual_data(n=400, seed=157, grid=41):
    integer(n, "n", 100)
    integer(seed, "seed", 0)
    integer(grid, "grid", 15)
    if n > 1000 or n % 10 or grid > 81:
        raise ValueError("n must be a multiple of 10 <= 1000; grid must be <= 81.")
    x, y = make_moons(n_samples=n, noise=0.30, random_state=seed)
    dev, test = train_test_split(np.arange(n), test_size=0.2, stratify=y, random_state=seed)
    train, validation = train_test_split(dev, test_size=0.25, stratify=y[dev], random_state=seed)
    scaler = StandardScaler().fit(x[train])
    axis_x = np.linspace(x[train, 0].min() - .5, x[train, 0].max() + .5, grid)
    axis_y = np.linspace(x[train, 1].min() - .5, x[train, 1].max() + .5, grid)
    gx, gy = np.meshgrid(axis_x, axis_y)
    grid_scaled = torch.tensor(scaler.transform(np.column_stack((gx.ravel(), gy.ravel()))), dtype=torch.float32)
    ids = {"train": train, "validation": validation, "test": test}
    splits = {key: (torch.tensor(scaler.transform(x[value]), dtype=torch.float32),
                    torch.tensor(y[value], dtype=torch.long)) for key, value in ids.items()}
    return {"x": x, "y": y, "ids": ids, "splits": splits, "grid_x": gx, "grid_y": gy,
            "grid_scaled": grid_scaled, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}


@torch.no_grad()
def snapshot(model, data, epoch, train, validation):
    model.eval()
    probability = torch.softmax(model(data["grid_scaled"]), dim=1)[:, 1]
    vector = linear_parameters(model).numpy().copy()
    return {"epoch": epoch, "probability": probability.numpy().reshape(data["grid_x"].shape),
            "weights": vector, "weight_norm": float(np.linalg.norm(vector)),
            "train": train, "validation": validation}


def train_visuals(epochs=120, n=400, seed=157, grid=41, patience=15, min_delta=1e-4,
                  evaluate_test=False, progress=None):
    integer(epochs, "epochs")
    integer(patience, "patience")
    if epochs > 200:
        raise ValueError("epochs must be <= 200 for this CPU teaching lab.")
    if not np.isfinite(min_delta) or not 0 <= min_delta <= 1:
        raise ValueError("min_delta must be finite and in [0, 1].")
    data = visual_data(n, seed, grid)
    runs, winner, winner_index = [], None, None
    # PyTorch CPU RNG/thread settings are process-global across Streamlit sessions.
    with TRAINING_LOCK, cpu_context():
        generator = torch.Generator().manual_seed(seed + 1)
        orders = [torch.randperm(len(data["ids"]["train"]), generator=generator) for _ in range(epochs)]
        for config_index, config in enumerate(CONFIGURATIONS):
            torch.manual_seed(seed + 3)
            model = MLP(config, seed + 2, input_dim=2)
            optimizer = make_optimizer(model, config.weight_decay, lr=0.003)
            initial_hash = fingerprint(linear_parameters(model))
            stopper = EarlyStopping(patience, min_delta)
            history, snapshots, steps = [], [], 0
            started = perf_counter()
            snapshots.append(snapshot(model, data, 0, evaluate(model, *data["splits"]["train"]),
                                      evaluate(model, *data["splits"]["validation"])))
            for epoch, order in enumerate(orders, 1):
                model.train()
                for indices in batch_indices(order, 64):
                    optimizer.zero_grad(set_to_none=True)
                    x, y = data["splits"]["train"]
                    loss = torch.nn.functional.cross_entropy(model(x[indices]), y[indices])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss.")
                    loss.backward()
                    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                        raise ValueError("Nonfinite gradient.")
                    optimizer.step()
                    steps += 1
                train = evaluate(model, *data["splits"]["train"])
                validation = evaluate(model, *data["splits"]["validation"])
                history.append({"epoch": epoch, "train": train, "validation": validation})
                snapshots.append(snapshot(model, data, epoch, train, validation))
                stop = stopper.observe(validation["loss"], epoch, model)
                if progress:
                    progress((config_index + epoch / epochs) / len(CONFIGURATIONS), config.name, epoch)
                if config.early_stopping and stop:
                    break
            selected_epoch = len(history)
            if config.early_stopping:
                stopper.restore(model)
                selected_epoch = stopper.best_epoch
            selected = snapshot(model, data, selected_epoch, evaluate(model, *data["splits"]["train"]),
                                evaluate(model, *data["splits"]["validation"]))
            run = {"name": config.name, "configuration": asdict(config), "history": history,
                   "snapshots": snapshots, "selected": selected,
                   "initial_linear_sha256": initial_hash, "epochs_run": len(history), "steps": steps,
                   "checkpoint_policy": "best validation" if config.early_stopping else "last epoch",
                   "duration_seconds": perf_counter() - started}
            runs.append(run)
            if winner is None or selected["validation"]["loss"] < runs[winner_index]["selected"]["validation"]["loss"]:
                winner, winner_index = model, len(runs) - 1
            if progress:
                progress((config_index + 1) / len(CONFIGURATIONS), config.name, len(history))
        final_test = evaluate(winner, *data["splits"]["test"]) if evaluate_test else None
    return {"executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": torch.__version__, "sklearn": sklearn.__version__, "threads": 1,
                            "device": "cpu", "dtype": "float32"},
            "hypothesis": "Regularization and checkpoint selection can change boundary evolution and validation loss; "
                          "no universal improvement or ordering is assumed.",
            "configuration": {"epochs_max": epochs, "n": n, "seed": seed, "grid": grid,
                              "data": "2D synthetic moons, noise=0.30; no extra noise features",
                              "splits": {key: len(value) for key, value in data["ids"].items()},
                              "seed_batches": seed + 1, "seed_model": seed + 2, "seed_dropout": seed + 3,
                              "architecture": "2-64-64-2 ReLU; optional BN then ReLU then dropout",
                              "optimizer": "AdamW", "lr": 0.003, "betas": [0.9, 0.999], "eps": 1e-8,
                              "batch_size": 64, "patience": patience, "min_delta": min_delta,
                              "dropout": 0.3, "weight_decay": 0.01, "decay_policy": "linear matrices only",
                              "batchnorm_eps": 1e-5, "batchnorm_momentum": 0.1,
                              "split_sha256": {key: fingerprint(value) for key, value in data["ids"].items()},
                              "batch_orders_sha256": fingerprint(torch.stack(orders)),
                              "scaler_mean": data["scaler_mean"], "scaler_scale": data["scaler_scale"],
                              "metric_mode": "eval; mean cross entropy excludes decay", "scheduler": None},
            "data": {key: data[key] for key in ("x", "y", "ids", "grid_x", "grid_y")},
            "runs": runs, "selected_configuration": runs[winner_index]["name"],
            "final_test": final_test, "test_evaluations": int(evaluate_test),
            "interpretation_candidate": "Inspect real boundaries and held-out losses under the declared "
                                        "checkpoint policies; the combination need not win.",
            "interpretation_status": "author review required",
            "limitation": "One synthetic 2D split; untuned settings; different checkpoint policies and updates. "
                          "Not comparable numerically with the historical 20-feature analytical study. "
                          "Snapshot norms include linear biases; duration includes grid evaluation and snapshots."}
