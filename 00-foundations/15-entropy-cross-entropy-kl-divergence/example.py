"""Connect distribution mismatch, classification confidence, and token loss."""

from __future__ import annotations

import math

import numpy as np

from from_scratch import (
    categorical_nll_from_logits,
    cross_entropy,
    entropy,
    kl_divergence,
)


def distribution_demo() -> None:
    """Verify the entropy, cross-entropy, and KL decomposition."""
    target = np.array([0.7, 0.2, 0.1])
    prediction = np.array([0.6, 0.3, 0.1])
    target_entropy = entropy(target)
    model_cross_entropy = cross_entropy(target, prediction)
    forward_kl = kl_divergence(target, prediction)
    reverse_kl = kl_divergence(prediction, target)

    print("Distribution mismatch (code-defined probabilities)")
    print(f"  H(P):             {target_entropy:.6f} nats")
    print(f"  H(P, Q):          {model_cross_entropy:.6f} nats")
    print(f"  KL(P || Q):       {forward_kl:.6f} nats")
    print(f"  KL(Q || P):       {reverse_kl:.6f} nats")
    print(
        "  decomposition:    ",
        np.isclose(model_cross_entropy, target_entropy + forward_kl),
    )


def confidence_demo() -> None:
    """Show confidence changing log loss when both decisions are wrong."""
    target = np.array([1.0, 0.0])
    uncertain_wrong = np.array([0.49, 0.51])
    confident_wrong = np.array([0.001, 0.999])

    print("\nSame wrong class decision, different confidence")
    print(f"  uncertain wrong CE: {cross_entropy(target, uncertain_wrong):.6f} nats")
    print(f"  confident wrong CE: {cross_entropy(target, confident_wrong):.6f} nats")


def token_loss_demo() -> None:
    """Compute token-weighted LLM-style loss while excluding padding."""
    logits = np.array(
        [
            [
                [3.0, 1.0, 0.0, -1.0],
                [0.0, 2.0, 1.0, -1.0],
                [1.0, 0.0, 2.0, -2.0],
            ],
            [
                [0.5, 1.5, 0.0, -0.5],
                [2.0, 0.0, 1.0, -1.0],
                [9.0, -9.0, -9.0, -9.0],
            ],
        ]
    )
    targets = np.array([[0, 1, 2], [1, 0, 0]])
    valid_tokens = np.array([[True, True, True], [True, True, False]])
    token_losses = categorical_nll_from_logits(
        logits, targets, mask=valid_tokens, reduction="none"
    )
    mean_token_loss = categorical_nll_from_logits(
        logits, targets, mask=valid_tokens, reduction="mean"
    )

    print("\nSynthetic next-token batch (2 sequences, 5 valid tokens)")
    print(f"  per-position NLL:\n{np.array2string(token_losses, precision=6)}")
    print(f"  valid-token mean: {mean_token_loss:.6f} nats")
    print(f"  perplexity:       {math.exp(mean_token_loss):.6f}")
    print("  The final padded position is masked and contributes zero.")


def main() -> None:
    """Run all deterministic numerical demonstrations."""
    distribution_demo()
    confidence_demo()
    token_loss_demo()


if __name__ == "__main__":
    main()
