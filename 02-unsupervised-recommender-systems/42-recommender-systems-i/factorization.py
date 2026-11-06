"""Small masked explicit-rating factorization with inspectable batch gradients."""

from numbers import Real

import numpy as np


def fit_factorization(ratings, dimensions=3, epochs=240, learning_rate=0.08,
                      regularization=0.02, seed=42, snapshots=12):
    """Minimize mean squared observed error + L2 using full-batch gradient descent.

    The loss is mean(error**2) + lambda*(||P||²+||Q||²+||bu||²+||bi||²).
    All gradients come from the same pre-update state. Missing pairs contribute
    no residual. Empty user/item factors stay zero rather than random evidence.
    No held-out labels enter this interface; losses are training quantities.
    """
    ratings = np.asarray(ratings)
    if (ratings.ndim != 2 or ratings.size == 0 or ratings.dtype.kind not in "iuf"
            or np.isinf(ratings).any() or not np.isfinite(ratings).any()):
        raise ValueError("ratings must be a nonempty numeric matrix with finite observations or NaN")
    for value, name in ((dimensions, "dimensions"), (epochs, "epochs"), (snapshots, "snapshots")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in ((learning_rate, "learning_rate"), (regularization, "regularization")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if learning_rate == 0:
        raise ValueError("learning_rate must be positive")
    ratings = ratings.astype(float, copy=True)
    mask = np.isfinite(ratings)
    count = int(mask.sum())
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.2, (ratings.shape[0], dimensions))
    q = rng.normal(0, 0.2, (ratings.shape[1], dimensions))
    p[~mask.any(axis=1)] = 0
    q[~mask.any(axis=0)] = 0
    bu, bi = np.zeros(len(p)), np.zeros(len(q))
    mean = float(ratings[mask].mean())
    targets = np.where(mask, ratings, 0.)
    saved_epochs = set(np.linspace(0, epochs, min(snapshots, epochs + 1), dtype=int))
    history, states = [], []
    for epoch in range(epochs + 1):
        predictions = mean + bu[:, None] + bi[None, :] + p @ q.T
        residual = np.where(mask, predictions - targets, 0.)
        mse = float(np.sum(residual ** 2) / count)
        penalty = regularization * sum(float(np.sum(v ** 2)) for v in (p, q, bu, bi))
        if not np.isfinite(mse + penalty):
            raise ValueError("optimization diverged; reduce learning_rate")
        history.append({"epoch": epoch, "mse": mse, "objective": mse + penalty})
        if epoch in saved_epochs:
            states.append({"epoch": epoch, "p": p.copy(), "q": q.copy(),
                           "bu": bu.copy(), "bi": bi.copy(), "predictions": predictions.copy()})
        if epoch == epochs:
            break
        scaled = 2 * residual / count
        gp = scaled @ q + 2 * regularization * p
        gq = scaled.T @ p + 2 * regularization * q
        gbu = scaled.sum(axis=1) + 2 * regularization * bu
        gbi = scaled.sum(axis=0) + 2 * regularization * bi
        p -= learning_rate * gp
        q -= learning_rate * gq
        bu -= learning_rate * gbu
        bi -= learning_rate * gbi
    return dict(p=p, q=q, bu=bu, bi=bi, mean=mean, predictions=predictions,
                mask=mask.copy(), history=history, states=states,
                configuration=dict(dimensions=dimensions, epochs=epochs,
                                   learning_rate=learning_rate, regularization=regularization, seed=seed))
