"""One-step constrained softmax on synthetic logits, not a grammar decoder."""

import json

import numpy as np


def constrained_softmax(logits, allowed) -> np.ndarray:
    """Normalize only permitted finite logits; reject ambiguous mask inputs."""
    scores = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(allowed)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("logits must be a nonempty one-dimensional array")
    if mask.dtype != np.bool_ or mask.shape != scores.shape:
        raise ValueError("allowed must be a boolean array with the same shape")
    if not np.isfinite(scores).all():
        raise ValueError("logits must all be finite")
    if not mask.any():
        raise ValueError("at least one token must be allowed")
    permitted = scores[mask]
    # Shift AFTER selecting permitted tokens: a forbidden maximum must not
    # underflow every allowed weight. Very negative differences may become -inf.
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(permitted - permitted.max())
    probabilities = np.zeros_like(scores)
    probabilities[mask] = weights / weights.sum()
    return probabilities


def main() -> None:
    tokens = ["100", "200", '"', "null", "true"]
    logits = np.array([1.2, 0.8, 2.0, -0.2, 0.1])
    allowed = np.array([True, True, False, False, False])
    original = constrained_softmax(logits, np.ones(5, dtype=bool))
    constrained = constrained_softmax(logits, allowed)
    print(json.dumps({
        "synthetic": True, "tokens": tokens, "logits": logits.tolist(),
        "allowed": allowed.tolist(), "original": original.tolist(),
        "admissible_mass": float(original[allowed].sum()),
        "constrained": constrained.tolist(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
