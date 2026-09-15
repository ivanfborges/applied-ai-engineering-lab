"""Educational ordinary K-fold splitter, not a production replacement."""
from numbers import Integral

import numpy as np


def kfold_indices(n_samples, n_splits=5, *, shuffle=True, seed=42):
    """Yield balanced, disjoint train/validation index arrays."""
    for name, value in (("n_samples", n_samples), ("n_splits", n_splits)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= n_splits <= n_samples:
        raise ValueError("require 2 <= n_splits <= n_samples")
    if not isinstance(shuffle, (bool, np.bool_)):
        raise ValueError("shuffle must be boolean")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    indices = np.arange(n_samples)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, n_splits)
    for held_out in range(n_splits):
        train = np.concatenate([fold for i, fold in enumerate(folds) if i != held_out])
        yield train, folds[held_out].copy()


if __name__ == "__main__":
    for number, (train, valid) in enumerate(kfold_indices(11, 3), start=1):
        print(f"Fold {number}: train={train.tolist()}, validation={valid.tolist()}")
