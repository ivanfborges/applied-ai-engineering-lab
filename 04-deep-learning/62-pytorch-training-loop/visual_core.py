"""Bounded CPU experiments and measured events for the Day 62 visual lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise ImportError('From the repository root: python -m pip install -e ".[dev,deep-learning]"') from error


@dataclass(frozen=True)
class LabConfig:
    seed: int = 62
    samples: int = 600
    epochs: int = 40
    batch_size: int = 32
    frames: int = 20
    grid_size: int = 35
    learning_rate: float = 0.02

    def validate(self) -> None:
        for name, lower, upper in (("seed", 0, 2**32 - 10), ("samples", 80, 1500),
                                   ("epochs", 2, 120), ("batch_size", 4, 256),
                                   ("frames", 4, 40), ("grid_size", 12, 45)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")


def make_data(config: LabConfig) -> dict:
    config.validate()
    rng = np.random.default_rng(config.seed)
    raw = rng.normal(size=(config.samples, 2)).astype("float32")
    signal = raw[:, 0] * raw[:, 1] + 0.4 * raw[:, 0] - 0.2 * raw[:, 1]
    targets = (signal + 0.3 * rng.normal(size=config.samples) > 0).astype("float32")[:, None]
    order = np.random.default_rng(config.seed + 1).permutation(config.samples)
    a, b = int(0.6 * config.samples), int(0.8 * config.samples)
    ids = {"train": order[:a], "validation": order[a:b], "test": order[b:]}
    mean = raw[ids["train"]].mean(0)
    scale = np.maximum(raw[ids["train"]].std(0), 1e-6)
    features = (raw - mean) / scale
    axis = np.linspace(-3, 3, config.grid_size, dtype="float32")
    gx, gy = np.meshgrid(axis, axis)
    return {"raw": raw, "x": features, "y": targets, "ids": ids,
            "mean": mean, "scale": scale, "axis": axis,
            "grid": np.column_stack([gx.ravel(), gy.ravel()]).astype("float32")}


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(x)))


def make_model(kind: str, seed: int) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == "mlp":
            return TinyMLP()
        if kind == "linear":
            return nn.Linear(2, 1)
    raise ValueError("kind must be mlp or linear")


def snapshot(model: nn.Module) -> dict:
    return {name: value.detach().numpy().copy() for name, value in model.state_dict().items()}


def restore(model: nn.Module, state: dict) -> None:
    model.load_state_dict({k: torch.as_tensor(v).clone() for k, v in state.items()})


def parameters_vector(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().numpy().ravel().copy() for p in model.parameters()])


@torch.inference_mode()
def measure(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    logits = model(torch.from_numpy(x))
    target = torch.from_numpy(y)
    return {"loss": float(nn.functional.binary_cross_entropy_with_logits(logits, target)),
            "accuracy": float(((logits >= 0) == (target == 1)).float().mean())}


@torch.inference_mode()
def predict_grid(model: nn.Module, data: dict) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(data["grid"]))).numpy().reshape(
        len(data["axis"]), len(data["axis"])).copy()


def layer_norms(named_values: dict) -> dict:
    return {layer: float(np.sqrt(sum(np.square(v).sum() for k, v in named_values.items()
                                    if k.rsplit(".", 1)[0] == layer)))
            for layer in sorted({k.rsplit(".", 1)[0] for k in named_values})}


def record_training(data: dict, config: LabConfig, kind: str = "mlp") -> dict:
    """Capture before/backward/after values belonging to the SAME actual update."""
    config.validate()
    model = make_model(kind, config.seed + 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_ids, val_ids = data["ids"]["train"], data["ids"]["validation"]
    dataset = TensorDataset(torch.from_numpy(data["x"][train_ids]),
                            torch.from_numpy(data["y"][train_ids]), torch.from_numpy(train_ids))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(config.seed + 2))
    total_steps = config.epochs * len(loader)
    capture = set(np.linspace(1, total_steps, min(config.frames, total_steps), dtype=int).tolist())
    history, checkpoints, events, grad_history, orders = [], [], [], [], []

    def checkpoint(epoch: int, step: int, online: float | None) -> None:
        train = measure(model, data["x"][train_ids], data["y"][train_ids])
        validation = measure(model, data["x"][val_ids], data["y"][val_ids])
        history.append({"epoch": epoch, "step": step, "online_loss": online,
                        "train_loss": train["loss"], "validation_loss": validation["loss"],
                        "train_accuracy": train["accuracy"], "validation_accuracy": validation["accuracy"]})
        checkpoints.append({"epoch": epoch, "step": step, "state": snapshot(model),
                            "grid_probability": predict_grid(model, data)})

    checkpoint(0, 0, None)
    step = 0
    for epoch in range(1, config.epochs + 1):
        total, count, epoch_order = 0., 0, []
        for batch_index, (x, y, ids) in enumerate(loader, 1):
            model.train()
            step += 1
            epoch_order.extend(ids.tolist())
            optimizer.zero_grad(set_to_none=True)
            before = snapshot(model)
            if kind == "mlp":
                z1 = model.hidden(x)
                hidden = torch.relu(z1)
                logits = model.output(hidden)
                z1.retain_grad()
                hidden.retain_grad()
            else:
                logits = model(x)
            logits.retain_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            gradients = {k: p.grad.detach().numpy().copy() for k, p in model.named_parameters()}
            if not all(np.isfinite(v).all() for v in gradients.values()) or not torch.isfinite(loss):
                raise ValueError("nonfinite loss or gradient")
            grad_norm = layer_norms(gradients)
            grad_history.append({"step": step, **grad_norm})
            event = None
            if kind == "mlp" and step in capture:
                event = {"epoch": epoch, "step": step, "batch_index": batch_index,
                         "batches_in_epoch": len(loader), "ids": ids.numpy().copy(),
                         "x": x.numpy().copy(), "y": y.numpy().copy(), "state_before": before,
                         "gradient": gradients, "gradient_norms": grad_norm,
                         "batch_loss_before": loss.item(), "logits_before": logits.detach().numpy().copy(),
                         "sample": {"x": x[0].numpy().copy(), "y": y[0].item(),
                                    "z1": z1[0].detach().numpy().copy(),
                                    "hidden": hidden[0].detach().numpy().copy(),
                                    "logit": logits[0].item(), "probability": torch.sigmoid(logits[0]).item(),
                                    "dlogit": logits.grad[0].item(),
                                    "dhidden": hidden.grad[0].numpy().copy(),
                                    "dz1": z1.grad[0].numpy().copy()}}
            # backward only fills gradients; this call is the parameter mutation.
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
            if event is not None:
                after = snapshot(model)
                if not all(np.isfinite(v).all() for v in after.values()):
                    raise ValueError("nonfinite parameters")
                delta = {k: after[k] - before[k] for k in before}
                event.update(state_after=after, delta=delta, update_norms=layer_norms(delta),
                             grid_probability=predict_grid(model, data),
                             train=measure(model, data["x"][train_ids], data["y"][train_ids]),
                             validation=measure(model, data["x"][val_ids], data["y"][val_ids]))
                with torch.inference_mode():
                    post_logits = model(x)
                    event["logits_after"] = post_logits.numpy().copy()
                    event["batch_loss_after"] = float(nn.functional.binary_cross_entropy_with_logits(post_logits, y))
                events.append(event)
        orders.append(np.asarray(epoch_order))
        checkpoint(epoch, step, total / count)
    best_epoch = min(range(1, len(history)), key=lambda i: history[i]["validation_loss"])
    return {"kind": kind, "history": history, "checkpoints": checkpoints,
            "best_epoch": best_epoch, "events": events, "gradient_history": grad_history,
            "orders": orders, "parameters": sum(p.numel() for p in model.parameters()),
            "total_steps": total_steps}


def logistic_objective(x: np.ndarray, y: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Fixed empirical BCE for arbitrary (..., 2) [weight, bias] coordinates."""
    x, y, parameters = (np.asarray(v, dtype=np.float64) for v in (x, y, parameters))
    if x.ndim != 1 or y.shape != x.shape or not len(x) or parameters.ndim < 1 or parameters.shape[-1] != 2:
        raise ValueError("expected nonempty x/y shape (N,) and parameters (..., 2)")
    if not all(np.isfinite(v).all() for v in (x, y, parameters)) or not np.isin(y, [0, 1]).all():
        raise ValueError("inputs must be finite and targets binary")
    z = parameters[..., 0, None] * x + parameters[..., 1, None]
    return np.logaddexp(0, np.where(y == 1, -z, z)).mean(axis=-1)


def record_logistic(*, seed: int = 62, steps: int = 64, grid_size: int = 41) -> dict:
    if not 4 <= steps <= 120 or not 12 <= grid_size <= 51:
        raise ValueError("steps must be 4..120 and grid_size 12..51")
    rng = np.random.default_rng(seed + 30)
    x = rng.normal(size=80)
    y = (1.2 * x - 0.3 + rng.normal(scale=0.8, size=80) > 0).astype(float)
    runs = {}
    for optimizer_name, rate in (("SGD", 0.05), ("SGD", 0.8), ("Adam", 0.03), ("Adam", 0.3)):
        for batch_size in (8, 80):
            parameters = nn.Parameter(torch.tensor([-1.5, 1.0], dtype=torch.float64))
            optimizer_class = torch.optim.SGD if optimizer_name == "SGD" else torch.optim.Adam
            optimizer = optimizer_class([parameters], lr=rate)
            tx, ty = torch.from_numpy(x), torch.from_numpy(y)
            loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True,
                                generator=torch.Generator().manual_seed(seed + 31), num_workers=0)
            iterator = iter(loader)
            path = [parameters.detach().numpy().copy()]
            grads, batch_ids = [], []
            for _ in range(steps):
                try:
                    xb, yb = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    xb, yb = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(parameters[0] * xb + parameters[1], yb)
                loss.backward()
                grads.append(parameters.grad.numpy().copy())
                batch_ids.append(xb.numpy().copy())
                optimizer.step()
                path.append(parameters.detach().numpy().copy())
            path = np.asarray(path)
            losses = logistic_objective(x, y, path)
            if not np.isfinite(path).all() or not np.isfinite(losses).all():
                raise ValueError("nonfinite optimizer trajectory")
            key = f"{optimizer_name} lr={rate:g} batch={batch_size}"
            runs[key] = {"optimizer": optimizer_name, "learning_rate": rate, "batch_size": batch_size,
                         "path": path, "loss": losses, "gradient": np.asarray(grads),
                         "batch_x": batch_ids, "uphill_steps": int((np.diff(losses) > 0).sum())}
    all_positions = np.concatenate([r["path"] for r in runs.values()])
    low, high = all_positions.min(0) - 0.8, all_positions.max(0) + 0.8
    weights = np.linspace(low[0], high[0], grid_size)
    biases = np.linspace(low[1], high[1], grid_size)
    w, b = np.meshgrid(weights, biases)
    surface = logistic_objective(x, y, np.stack([w, b], axis=-1))
    return {"x": x, "y": y, "runs": runs, "weights": weights, "biases": biases,
            "surface": surface, "primary": "SGD lr=0.8 batch=8"}


def dropout_probe(seed: int = 62) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 40)
        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
        x = torch.tensor([[1., -0.5]])
        result = {}
        for training, inference in ((True, False), (False, False), (True, True), (False, True)):
            torch.manual_seed(seed + 41)
            model.train(training)
            values, flags, activations = [], [], []
            for _ in range(12):
                with torch.inference_mode(inference):
                    hidden = model[:3](x)
                    logits = model[3](hidden)
                    values.append(torch.sigmoid(logits).item())
                    flags.append(logits.requires_grad)
                    activations.append(hidden.detach().numpy().ravel().copy())
            result[f"{'train' if training else 'eval'} + {'inference' if inference else 'grad'}"] = {
                "probability": np.asarray(values), "requires_grad": flags,
                "hidden": np.asarray(activations), "training": training, "inference": inference}
    return result


def run_lab(config: LabConfig) -> dict:
    config.validate()
    data = make_data(config)
    mlp = record_training(data, config, "mlp")
    linear = record_training(data, config, "linear")
    candidates = {"mlp": mlp, "linear": linear}
    selected = min(candidates, key=lambda k: candidates[k]["history"][candidates[k]["best_epoch"]]["validation_loss"])
    chosen = candidates[selected]
    model = make_model(selected, config.seed + 3)
    restore(model, chosen["checkpoints"][chosen["best_epoch"]]["state"])
    test_ids = data["ids"]["test"]
    test = measure(model, data["x"][test_ids], data["y"][test_ids])
    return {"configuration": asdict(config), "data": data, "mlp": mlp, "linear": linear,
            "selected": selected, "selected_test": test,
            "logistic": record_logistic(seed=config.seed), "dropout": dropout_probe(config.seed)}
