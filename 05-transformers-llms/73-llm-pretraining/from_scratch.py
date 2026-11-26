"""Educational full-batch bigram maximum likelihood with explicit NumPy gradients."""

from datetime import datetime, timezone
import json
import platform

import numpy as np


def bigram_loss_gradient(weights, current, targets):
    """Mean NLL and dL/dW; repeated context rows must accumulate gradients."""
    weights = np.asarray(weights)
    current, targets = np.asarray(current), np.asarray(targets)
    if (weights.ndim != 2 or weights.shape[0] != weights.shape[1]
            or weights.shape[0] < 2 or weights.dtype.kind not in "fi"
            or not np.isfinite(weights).all()):
        raise ValueError("weights must be a finite real square matrix with size >= 2")
    for ids in (current, targets):
        if (ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= len(weights))):
            raise ValueError("token IDs must be a nonempty integer vector in vocabulary range")
    if current.shape != targets.shape:
        raise ValueError("current and targets must have equal shape")
    logits = weights.astype(np.float64)[current]
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    rows = np.arange(len(targets))
    loss = -log_probs[rows, targets].mean()
    residual = np.exp(log_probs)
    residual[rows, targets] -= 1
    gradient = np.zeros_like(weights, dtype=np.float64)
    np.add.at(gradient, current, residual / len(targets))
    return float(loss), gradient


def run_experiment():
    # Separate sequences prevent artificial transitions across document boundaries.
    sentences = ["<bos> ai learns patterns <eos>",
                 "<bos> ai learns language <eos>",
                 "<bos> models learn patterns <eos>"]
    vocabulary = sorted(set(" ".join(sentences).split()))
    stoi = {token: i for i, token in enumerate(vocabulary)}
    sequences = [[stoi[token] for token in text.split()] for text in sentences]
    current = np.array([a for seq in sequences for a in seq[:-1]])
    targets = np.array([b for seq in sequences for b in seq[1:]])
    weights = np.random.default_rng(42).normal(0, 0.01, (len(stoi), len(stoi)))
    initial, _ = bigram_loss_gradient(weights, current, targets)
    for _ in range(300):
        _, gradient = bigram_loss_gradient(weights, current, targets)
        weights -= gradient
    final, _ = bigram_loss_gradient(weights, current, targets)
    shifted = weights - weights.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Explicit mean-NLL gradients can fit observed synthetic bigram transitions.",
        "configuration": {"seed": 42, "steps": 300, "learning_rate": 1.0,
                          "sentences": sentences, "pairs": len(targets),
                          "parameters": weights.size, "dtype": "float64"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "initial_train_nll": initial, "final_train_nll": final,
        "next_token_after_learns": dict(zip(vocabulary, probabilities[stoi["learns"]].tolist())),
        "interpretation_candidate": "The table fits local transitions; author review required.",
        "limitation": "Training fit only, no held-out evaluation or context beyond one token.",
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, allow_nan=False))
