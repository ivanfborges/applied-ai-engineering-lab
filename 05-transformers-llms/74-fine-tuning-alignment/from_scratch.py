"""Educational NumPy DPO loss and gradient with respect to response log-probabilities."""

import json

import numpy as np


def dpo_loss_gradient(policy_logps, reference_logps, beta=0.5):
    """Mean pair loss; columns are chosen/rejected for the same prompt.

    The returned gradient differentiates independent policy log-probabilities,
    not model parameters. The reference is constant.
    """
    policy = np.asarray(policy_logps, dtype=np.float64)
    reference = np.asarray(reference_logps, dtype=np.float64)
    if policy.ndim != 2 or policy.shape[0] == 0 or policy.shape[1] != 2:
        raise ValueError("policy_logps must have nonempty shape (pairs, 2)")
    if reference.shape != policy.shape:
        raise ValueError("reference_logps must match policy_logps")
    if not np.isfinite(policy).all() or not np.isfinite(reference).all():
        raise ValueError("log-probabilities must be finite")
    if (policy > 0).any() or (reference > 0).any():
        raise ValueError("log-probabilities must be nonpositive")
    if isinstance(beta, bool) or not np.isscalar(beta) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    margin = policy[:, 0] - policy[:, 1] - reference[:, 0] + reference[:, 1]
    with np.errstate(over="ignore"):
        scaled = beta * margin
    if not np.isfinite(scaled).all():
        raise ValueError("scaled margins exceed the supported numeric range")
    loss = np.logaddexp(0.0, -scaled).mean()
    # exp(-softplus(z)) = sigmoid(-z), avoiding an overflowing exp(z).
    derivative = -beta * np.exp(-np.logaddexp(0.0, scaled)) / len(policy)
    return float(loss), np.column_stack((derivative, -derivative))


def main():
    policy = np.array([[-2.0, -4.0], [-3.0, -2.0]])
    reference = np.array([[-3.0, -4.0], [-2.0, -3.0]])
    loss, gradient = dpo_loss_gradient(policy, reference)
    print(json.dumps({"data": "synthetic fixed log-probabilities; no training",
                      "beta": 0.5, "loss": loss, "gradient": gradient.tolist()}, indent=2))


if __name__ == "__main__":
    main()
