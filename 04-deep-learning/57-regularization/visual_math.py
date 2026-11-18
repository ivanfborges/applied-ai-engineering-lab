"""Deterministic educational simulations, separate from learned-model results."""

import numpy as np

from example import integer


def bounded(value, name, low, high):
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}].")
    return value


def dropout_pass(p=0.3, pass_index=0):
    bounded(p, "p", 0, 0.8)
    integer(pass_index, "pass_index", 0)
    weights = np.random.default_rng(157)
    matrices = [weights.normal(0, 0.7, size=shape) for shape in ((3, 6), (6, 5), (5, 2))]
    masks = np.random.default_rng(1000 + pass_index)
    x = np.array([1., 0.5, -0.4])
    raw, active, post = [x], [np.ones(3, dtype=bool)], [x]
    for matrix in matrices[:2]:
        h = np.maximum(post[-1] @ matrix + 0.4, 0)
        mask = masks.random(len(h)) >= p
        raw.append(h)
        active.append(mask)
        post.append(h * mask / (1 - p))
    logits = post[-1] @ matrices[-1]
    probability = np.exp(logits - logits.max())
    probability /= probability.sum()
    raw.append(probability)
    active.append(np.ones(2, dtype=bool))
    post.append(probability)
    return {"raw": raw, "active": active, "post": post, "weights": matrices, "p": p}


def dropout_moments(p=0.3, passes=2000):
    bounded(p, "p", 0, 0.8)
    integer(passes, "passes", 2)
    activation = np.array([0.5, 1., 2.])
    uniform = np.random.default_rng(157).random((passes, 3))
    outputs = activation * (uniform >= p) / (1 - p)
    return {"activation": activation, "samples": outputs, "mean": outputs.mean(0),
            "variance": outputs.var(0), "target_variance": p / (1 - p) * activation**2,
            "retained": float((uniform >= p).mean())}


MATRIX = np.array([[3., 1.], [1., 1.]])
CENTER = np.array([1.6, -1.])
METHODS = ("SGD", "SGD + L2", "Adam + L2", "AdamW")


def quadratic(points):
    w = np.asarray(points, dtype=float)
    d = w - CENTER
    return 0.5 * np.einsum("...i,ij,...j->...", d, MATRIX, d)


def regularized(points, decay):
    return quadratic(points) + decay / 2 * np.sum(np.asarray(points)**2, axis=-1)


def decay_paths(decay=0.3, steps=80):
    """NumPy SGD/Adam/AdamW on a shifted SPD quadratic, matching PyTorch defaults stated here."""
    bounded(decay, "decay", 0, 1)
    integer(steps, "steps")
    paths = {}
    for method in METHODS:
        w = np.array([-2., 2.])
        first, second = np.zeros(2), np.zeros(2)
        points = [w.copy()]
        for t in range(1, steps + 1):
            g = MATRIX @ (w - CENTER)
            if method in ("SGD + L2", "Adam + L2"):
                g = g + decay * w
            if method.startswith("SGD"):
                w = w - 0.12 * g
            else:
                first = 0.9 * first + 0.1 * g
                second = 0.999 * second + 0.001 * g**2
                w = w * (1 - 0.08 * decay) if method == "AdamW" else w
                w = w - 0.08 * (first / (1 - 0.9**t)) / (np.sqrt(second / (1 - 0.999**t)) + 1e-8)
            points.append(w.copy())
        points = np.asarray(points)
        paths[method] = {"points": points, "loss": quadratic(points),
                         "objective": regularized(points, decay), "norm": np.linalg.norm(points, axis=1)}
    minimum = np.linalg.solve(MATRIX + decay * np.eye(2), MATRIX @ CENTER)
    return {"paths": paths, "decay": decay, "minimum": minimum,
            "original_minimum": CENTER.copy(), "steps": steps}


def batchnorm_trace(batch_size=16, gamma=1., beta=0., eps=1e-5, momentum=0.1, batches=40):
    integer(batch_size, "batch_size", 2)
    integer(batches, "batches", 2)
    bounded(gamma, "gamma", -3, 3)
    bounded(beta, "beta", -5, 5)
    bounded(eps, "eps", 1e-8, 1)
    bounded(momentum, "momentum", 0, 1)
    # Fixed parent stream, different finite batches; no trained network or dataset shift.
    x = np.random.default_rng(157).normal(size=(batches, 128, 3))[:, :batch_size]
    if batch_size > 128:
        raise ValueError("batch_size must be <= 128.")
    x = x * np.array([0.2, 1., 2.5]) + np.array([-3., 1., 5.])
    means, variance = x.mean(1), x.var(1)
    running_mean, running_var = np.zeros(3), np.ones(3)
    running_means, running_vars, normalized, inference = [], [], [], []
    for batch, mean, var in zip(x, means, variance):
        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * var * batch_size / (batch_size - 1)
        normalized.append((batch - mean) / np.sqrt(var + eps))
        inference.append(gamma * (batch - running_mean) / np.sqrt(running_var + eps) + beta)
        running_means.append(running_mean.copy())
        running_vars.append(running_var.copy())
    standardized = np.asarray(normalized)
    return {"raw": x, "normalized": standardized, "affine": gamma * standardized + beta,
            "inference": np.asarray(inference), "batch_mean": means, "batch_var": variance,
            "running_mean": np.asarray(running_means), "running_var": np.asarray(running_vars),
            "batch_size": batch_size, "gamma": gamma, "beta": beta, "eps": eps,
            "momentum": momentum, "population_mean": np.array([-3., 1., 5.]),
            "population_var": np.array([0.04, 1., 6.25])}


def stopping_replay(history, patience=15, min_delta=1e-4, maximum=None):
    """Replay a stopping rule over an executed trajectory; unseen future is excluded."""
    integer(patience, "patience")
    bounded(min_delta, "min_delta", 0, 1)
    if not history:
        raise ValueError("history must not be empty.")
    maximum = len(history) if maximum is None else integer(maximum, "maximum")
    if maximum > len(history):
        raise ValueError("maximum exceeds recorded epochs.")
    best_loss = progress = float("inf")
    best_epoch, bad = 0, 0
    replay = []
    for row in history[:maximum]:
        loss = row["validation"]["loss"]
        if not np.isfinite(loss):
            raise ValueError("Validation loss must be finite.")
        if loss < best_loss:
            best_loss, best_epoch = loss, row["epoch"]
        if loss < progress - min_delta:
            progress, bad = loss, 0
        else:
            bad += 1
        replay.append({"epoch": row["epoch"], "best_epoch": best_epoch,
                       "best_loss": best_loss, "bad_checks": bad, "stop": bad >= patience})
        if bad >= patience:
            break
    return {"rows": replay, "stop_epoch": replay[-1]["epoch"], "best_epoch": best_epoch,
            "best_loss": best_loss, "patience": patience, "min_delta": min_delta}
