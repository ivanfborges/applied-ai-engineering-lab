"""Matched CPU optimizer configurations on synthetic noisy concentric rings.

Print JSON only. This single-seed validation exercise is not a tuned benchmark.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as error:
    raise SystemExit('PyTorch is required. Install the canonical extra: '
                     'python -m pip install -e ".[deep-learning]"') from error

from from_scratch import METHODS


LEARNING_RATES = {"sgd": 0.1, "momentum": 0.03, "rmsprop": 0.003,
                  "adam": 0.01, "adamw": 0.01}


def positive_integer(value, name: str, *, minimum=1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def make_data(n: int = 1000, seed: int = 56):
    """Balanced labels; Gaussian radial noise; one fixed random 80/20 split."""
    positive_integer(n, "n", minimum=20)
    positive_integer(seed, "seed", minimum=0)
    if n % 2:
        raise ValueError("n must be even for balanced synthetic classes.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(n, dtype=torch.long) % 2
    angles = 2 * torch.pi * torch.rand(n, generator=generator, dtype=torch.float32)
    radii = 0.8 + 0.7 * labels + 0.18 * torch.randn(n, generator=generator, dtype=torch.float32)
    features = torch.stack((radii * torch.cos(angles), radii * torch.sin(angles)), dim=1)
    indices = torch.randperm(n, generator=generator)
    cut = 4 * n // 5
    train, validation = indices[:cut], indices[cut:]
    return features[train], labels[train], features[validation], labels[validation]


def batch_orders(n: int, epochs: int, seed: int) -> list[torch.Tensor]:
    positive_integer(n, "n")
    positive_integer(epochs, "epochs")
    positive_integer(seed, "seed", minimum=0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randperm(n, generator=generator) for _ in range(epochs)]


def make_model(seed: int) -> nn.Module:
    positive_integer(seed, "seed", minimum=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16),
                             nn.Tanh(), nn.Linear(16, 2)).to(device="cpu", dtype=torch.float32)


def make_optimizer(parameters, method: str, lr: float, weight_decay: float):
    """Set decay explicitly, including AdamW's otherwise nonzero default."""
    shared = dict(lr=lr, weight_decay=weight_decay, foreach=False)
    if method in ("sgd", "momentum"):
        return torch.optim.SGD(parameters, momentum=0.9 if method == "momentum" else 0,
                               dampening=0, nesterov=False, **shared)
    if method == "rmsprop":
        return torch.optim.RMSprop(parameters, alpha=0.99, eps=1e-8,
                                   momentum=0, centered=False, **shared)
    if method in ("adam", "adamw"):
        optimizer_type = torch.optim.Adam if method == "adam" else torch.optim.AdamW
        return optimizer_type(parameters, betas=(0.9, 0.999), eps=1e-8,
                              amsgrad=False, **shared)
    raise ValueError(f"method must be one of {METHODS}.")


def parameter_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def fingerprint(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def evaluate(model, features, labels) -> dict:
    model.eval()
    logits = model(features)
    return {"loss": nn.functional.cross_entropy(logits, labels).item(),
            "accuracy": (logits.argmax(dim=1) == labels).float().mean().item()}


def run_comparison(*, epochs: int = 60, batch_size: int = 64,
                   n: int = 1000, seed: int = 56) -> dict:
    positive_integer(epochs, "epochs")
    positive_integer(batch_size, "batch_size")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    x_train, y_train, x_validation, y_validation = make_data(n, seed)
    orders = batch_orders(len(x_train), epochs, seed + 1)
    base = make_model(seed + 2)
    initial = parameter_vector(base)
    initial_validation = evaluate(base, x_validation, y_validation)
    configurations = [("no_decay", method, 0.0) for method in METHODS]
    configurations += [("decay_pair", method, 0.01) for method in ("adam", "adamw")]
    runs = []
    for group, method, decay in configurations:
        model = copy.deepcopy(base)
        lr = LEARNING_RATES[method]
        optimizer = make_optimizer(model.parameters(), method, lr, decay)
        initial_hash = fingerprint(parameter_vector(model))
        steps, max_gradient_norm, last_update_norm = 0, 0.0, 0.0
        history = []
        started = perf_counter()
        for epoch, order in enumerate(orders, start=1):
            model.train()
            for indices in order.split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x_train[indices]), y_train[indices])
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss for {method}, epoch {epoch}.")
                loss.backward()
                gradients = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
                if not torch.all(torch.isfinite(gradients)):
                    raise ValueError(f"Nonfinite gradient for {method}, epoch {epoch}.")
                max_gradient_norm = max(max_gradient_norm, torch.linalg.vector_norm(gradients).item())
                before = parameter_vector(model)
                optimizer.step()
                after = parameter_vector(model)
                if not torch.all(torch.isfinite(after)):
                    raise ValueError(f"Nonfinite parameters for {method}, epoch {epoch}.")
                last_update_norm = torch.linalg.vector_norm(after - before).item()
                steps += 1
            history.append({"epoch": epoch, "train": evaluate(model, x_train, y_train),
                            "validation": evaluate(model, x_validation, y_validation)})
        runs.append({"group": group, "method": method, "lr": lr, "weight_decay": decay,
                     "steps": steps, "initial_parameters_sha256": initial_hash,
                     "final_parameters_sha256": fingerprint(parameter_vector(model)),
                     "final_parameter_norm": torch.linalg.vector_norm(parameter_vector(model)).item(),
                     "max_gradient_norm": max_gradient_norm, "last_update_norm": last_update_norm,
                     "elapsed_seconds": perf_counter() - started,
                     "final_train": history[-1]["train"],
                     "final_validation": history[-1]["validation"], "history": history})
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__,
                        "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads()},
        "hypothesis": "Matched inputs expose configuration-dependent optimization; "
                      "nonzero coupled and decoupled decay may change fitted parameters.",
        "configuration": {"data": "synthetic noisy concentric rings; no downloads",
                          "n": n, "train_n": len(x_train), "validation_n": len(x_validation),
                          "seed_data_split": seed, "seed_batches": seed + 1, "seed_model": seed + 2,
                          "epochs": epochs, "batch_size": batch_size,
                          "architecture": "2-16-16-2 tanh MLP, raw two-class logits",
                          "parameter_count": initial.numel(), "initial_parameter_norm": initial.norm().item(),
                          "initial_parameters_sha256": fingerprint(initial),
                          "batch_orders_sha256": fingerprint(torch.stack(orders)),
                          "momentum": 0.9, "rmsprop_alpha": 0.99,
                          "adam_betas": [0.9, 0.999], "eps": 1e-8,
                          "decay_policy": "all parameters, including biases; educational choice",
                          "loss": "mean cross entropy; reported losses exclude decay penalties",
                          "scheduler": None, "clipping": None, "checkpoint_selection": None},
        "initial_validation": initial_validation, "runs": runs,
        "interpretation_candidate": "Differences apply to these fixed, untuned configurations only.",
        "interpretation_status": "author review required",
        "limitation": "One seed/split, hand-chosen rates, no tuning or final test set; "
                      "timings include diagnostics/evaluation and are not performance benchmarks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=56)
    args = parser.parse_args()
    try:
        result = run_comparison(epochs=args.epochs, batch_size=args.batch_size,
                                n=args.samples, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
