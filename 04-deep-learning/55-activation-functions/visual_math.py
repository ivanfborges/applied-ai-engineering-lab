"""Synthetic, testable mechanisms for the Day 55 visual lab. CPU only."""

from contextlib import contextmanager
import hashlib
from time import perf_counter

import numpy as np

from from_scratch import activation_and_derivative, softmax


def bounded_integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("probabilities must sum to one on the last axis")
    terms = np.zeros_like(p)
    np.log(p, out=terms, where=p > 0)
    return -np.sum(p * terms, axis=-1)


def gradient_chains(depth=20):
    """Local derivatives at controlled fixed preactivations; upstream adjoint=1."""
    bounded_integer(depth, "depth", 1, 60)
    cases = [("sigmoid at z=4; w=1", "sigmoid", 4., 1.),
             ("tanh at z=4; w=1", "tanh", 4., 1.),
             ("ReLU positive; w=1", "relu", 1., 1.),
             ("ReLU positive; w=0.5", "relu", 1., .5),
             ("ReLU positive; w=1.5", "relu", 1., 1.5)]
    layers = np.arange(depth + 1)
    return [{"label": label, "activation": name, "z": z, "weight": weight,
             "local_derivative": float(activation_and_derivative(z, name)[1]),
             "depth": layers,
             "magnitude": np.abs(weight * activation_and_derivative(z, name)[1]) ** layers}
            for label, name, z, weight in cases]


def dying_neuron(steps=80, learning_rate=0.5, alpha=0.1):
    """Matched full-batch GD on half-MSE, including the initial state."""
    bounded_integer(steps, "steps", 1, 500)
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be in (0, 1]")
    activation_and_derivative([0.], "leaky_relu", alpha)
    x = np.linspace(-1, 1, 41)
    y = .6 + .2 * x
    traces = {}
    for name in ("relu", "leaky_relu"):
        w, b = .2, -1.2
        history = []
        for step in range(steps + 1):
            z = w * x + b
            prediction, slope = activation_and_derivative(z, name, alpha)
            residual = prediction - y
            dw, db = np.mean(residual * slope * x), np.mean(residual * slope)
            history.append({"step": step, "w": w, "b": b, "z": z.copy(),
                            "output": prediction.copy(), "slope": slope.copy(),
                            "loss": float(np.mean(residual ** 2) / 2),
                            "gradient_norm": float(np.hypot(dw, db))})
            if step < steps:
                w, b = w - learning_rate * dw, b - learning_rate * db
        traces[name] = history
    return {"x": x, "target": y, "alpha": alpha, "learning_rate": learning_rate,
            "traces": traces}


def attention(temperature=1.0):
    """Three synthetic queries, four keys; normalize over keys (last axis)."""
    q = np.array([[1., 0.], [0., 1.], [1., -1.]])
    k = np.array([[1., 0.], [0., 1.], [-1., 0.], [.5, -.5]])
    scores = q @ k.T / np.sqrt(q.shape[1])
    return {"queries": q, "keys": k, "scores": scores,
            "weights": softmax(scores, axis=-1, temperature=temperature)}


def simplex_samples(seed=55, count=180):
    bounded_integer(count, "count", 3, 2000)
    logits = np.random.default_rng(seed).normal(0, 2, (count, 3))
    temperatures = np.geomspace(.15, 5, 60)
    trajectory = np.array([softmax([2., 1., -1.], temperature=t) for t in temperatures])
    return {"logits": logits, "probabilities": softmax(logits),
            "temperatures": temperatures, "trajectory": trajectory}


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch is required here. From the repository root run '
                           'python -m pip install -e ".[dev,deep-learning]". '
                           'See VISUAL_LAB.md for CPU-only installation.') from error
    return torch


@contextmanager
def torch_cpu():
    """Bound CPU threads and restore caller RNG/determinism/thread settings."""
    torch = require_torch()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            yield torch
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)


def autograd_checks():
    from from_scratch import ELEMENTWISE, cross_entropy_with_logits, softmax_jacobian
    with torch_cpu() as torch:
        functions = {"sigmoid": torch.sigmoid, "tanh": torch.tanh,
                     "relu": torch.nn.functional.relu,
                     "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, .01),
                     "gelu": lambda x: torch.nn.functional.gelu(x, approximate="none")}
        x_values = np.array([-6., -2., -.5, .5, 2., 6.])
        errors = {}
        for name in ELEMENTWISE:
            x = torch.tensor(x_values, dtype=torch.float64, requires_grad=True)
            output = functions[name](x)
            derivative = torch.autograd.grad(output.sum(), x)[0].detach().numpy()
            manual, slope = activation_and_derivative(x_values, name)
            np.testing.assert_allclose(output.detach().numpy(), manual, atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(derivative, slope, atol=1e-14, rtol=1e-10)
            errors[name] = float(np.max(np.abs(derivative - slope)))
        logits = torch.tensor([[2., 1., -1.], [-1., 0., 3.]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([0, 2])
        for t in (.5, 1., 2.):
            loss = torch.nn.functional.cross_entropy(logits / t, targets)
            derivative = torch.autograd.grad(loss, logits)[0].numpy()
            manual_loss, manual = cross_entropy_with_logits(
                logits.detach().numpy(), np.eye(3)[[0, 2]], t)
            np.testing.assert_allclose(derivative, manual, atol=1e-14, rtol=1e-12)
            np.testing.assert_allclose(loss.item(), manual_loss, atol=1e-14)
            j = torch.autograd.functional.jacobian(
                lambda z: torch.softmax(z / t, dim=-1), logits[0]).numpy()
            np.testing.assert_allclose(j, softmax_jacobian([2, 1, -1], t), atol=1e-14)
        errors["cross_entropy"] = float(np.max(np.abs(derivative - manual)))
        return {"dtype": "float64", "max_absolute_derivative_errors": errors,
                "torch_version": torch.__version__}


def train_moons(epochs=400, grid=61, seed=55):
    """Final-epoch observations; validation is never used for training/selection."""
    bounded_integer(epochs, "epochs", 1, 2000)
    bounded_integer(grid, "grid", 15, 121)
    bounded_integer(seed, "seed", 0, 2**32 - 1)
    try:
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError('Install the canonical dependencies: '
                           'python -m pip install -e ".[dev,deep-learning]"') from error
    x, y = make_moons(n_samples=400, noise=.18, random_state=seed)
    train_idx, valid_idx = train_test_split(np.arange(len(x)), test_size=.25,
                                           stratify=y, random_state=seed)
    scaler = StandardScaler().fit(x[train_idx])
    scaled = scaler.transform(x)
    gx, gy = np.meshgrid(np.linspace(scaled[:, 0].min() - .5, scaled[:, 0].max() + .5, grid),
                         np.linspace(scaled[:, 1].min() - .5, scaled[:, 1].max() + .5, grid))
    points = np.column_stack([gx.ravel(), gy.ravel()])
    results = {}
    with torch_cpu() as torch:
        nn = torch.nn
        activations = {"relu": lambda: nn.ReLU(), "leaky_relu": lambda: nn.LeakyReLU(.1),
                       "gelu": lambda: nn.GELU(approximate="none"), "identity": lambda: nn.Identity()}
        torch.manual_seed(seed)
        def create(activation):
            return nn.Sequential(nn.Linear(2, 16), activation(), nn.Linear(16, 16),
                                 activation(), nn.Linear(16, 1)).cpu()
        initial = {key: value.clone() for key, value in create(nn.Identity).state_dict().items()}
        initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        tx = torch.tensor(scaled[train_idx], dtype=torch.float32)
        ty = torch.tensor(y[train_idx, None], dtype=torch.float32)
        vx = torch.tensor(scaled[valid_idx], dtype=torch.float32)
        vy = torch.tensor(y[valid_idx, None], dtype=torch.float32)
        mesh = torch.tensor(points, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()
        checkpoints = set(np.linspace(0, epochs, 7, dtype=int))
        for name, activation in activations.items():
            model = create(activation)
            model.load_state_dict(initial)
            optimizer = torch.optim.Adam(model.parameters(), lr=.02)
            snapshots, losses = [], []
            model.eval()
            with torch.no_grad():
                initial_loss = criterion(model(tx), ty).item()
                snapshots.append({"epoch": 0, "probability": torch.sigmoid(model(mesh)).numpy().reshape(gx.shape)})
            started = perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(tx), ty)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if epoch in checkpoints:
                    model.eval()
                    with torch.no_grad():
                        snapshots.append({"epoch": epoch, "probability": torch.sigmoid(model(mesh)).numpy().reshape(gx.shape)})
            elapsed = perf_counter() - started
            model.eval()
            with torch.no_grad():
                valid_logits = model(vx)
                grid_logits = model(mesh).numpy().reshape(gx.shape)
                metrics = {"initial_train_loss": initial_loss,
                           "train_loss": criterion(model(tx), ty).item(),
                           "validation_loss": criterion(valid_logits, vy).item(),
                           "validation_accuracy": ((valid_logits >= 0) == (vy >= .5)).float().mean().item(),
                           "training_seconds": elapsed}
            results[name] = {"metrics": metrics, "logits": grid_logits,
                             "probability": torch.sigmoid(torch.from_numpy(grid_logits)).numpy(),
                             "snapshots": snapshots, "initial_parameter_sha256": initial_hash,
                             "pre_update_training_losses": losses}
    return {"configuration": {"seed": seed, "observations": 400, "training_rows": 300,
                              "validation_rows": 100, "noise": .18, "architecture": [2, 16, 16, 1],
                              "epochs": epochs, "grid": grid, "optimizer": "Adam", "learning_rate": .02,
                              "leaky_slope": .1, "dtype": "float32", "device": "cpu", "threads": 1,
                              "initialization": "identical PyTorch Linear defaults; no activation-specific gain",
                              "time_scope": "training loop including update-checkpoint grids; excludes setup/imports/initial evaluation"},
            "x": scaled, "y": y, "train_idx": train_idx, "valid_idx": valid_idx,
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "grid_x": gx, "grid_y": gy, "models": results}
