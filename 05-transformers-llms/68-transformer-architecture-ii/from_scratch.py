"""Educational float64 multi-head self-attention; fixed random weights, no training."""

import numpy as np


class MultiHeadAttention:
    """Batched self-attention with equal-width heads and no biases or dropout.

    Inputs: (batch, tokens, d_model). Returned maps: (batch, heads, tokens, tokens).
    Boolean masks use True for BLOCKED keys, matching torch.nn.MultiheadAttention.
    """

    def __init__(self, d_model: int = 32, num_heads: int = 4, seed: int = 42):
        for name, value in (("d_model", d_model), ("num_heads", num_heads)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        rng = np.random.default_rng(seed)
        self.wq, self.wk, self.wv, self.wo = (
            rng.normal(0, self.d_model ** -0.5, (self.d_model, self.d_model))
            for _ in range(4)
        )

    def forward(self, x, *, causal=False, blocked_mask=None, key_padding_mask=None, scaled=True, return_trace=False):
        """Return output and per-head maps; reject queries with no permitted key.

        blocked_mask is (tokens, tokens) or exactly (batch, heads, tokens, tokens).
        key_padding_mask is (batch, tokens); it masks keys, not padded queries.
        scaled=False is an experimental control, not standard attention.
        return_trace=True adds an intermediate-tensor dictionary as a third return.
        """
        for name, value in (("causal", causal), ("scaled", scaled), ("return_trace", return_trace)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[-1] != self.d_model or min(x.shape) == 0:
            raise ValueError("x must have nonempty shape (batch, tokens, d_model)")
        if x.dtype.kind not in "fiu" or not np.isfinite(x).all():
            raise ValueError("x must contain finite real numbers")
        x = x.astype(np.float64)
        batch, tokens, _ = x.shape
        shape = (batch, self.num_heads, tokens, tokens)
        blocked = np.zeros(shape, dtype=bool)
        if blocked_mask is not None:
            mask = np.asarray(blocked_mask)
            if mask.dtype != np.bool_ or mask.shape not in ((tokens, tokens), shape):
                raise ValueError("blocked_mask must be boolean with shape (T,T) or (B,H,T,T)")
            blocked |= mask
        if key_padding_mask is not None:
            padding = np.asarray(key_padding_mask)
            if padding.dtype != np.bool_ or padding.shape != (batch, tokens):
                raise ValueError("key_padding_mask must be boolean with shape (B,T)")
            blocked |= padding[:, None, None, :]
        if causal:
            blocked |= np.triu(np.ones((tokens, tokens), dtype=bool), k=1)
        if blocked.all(axis=-1).any():
            raise ValueError("every query in every head must have at least one permitted key")

        def split(projected):
            return projected.reshape(batch, tokens, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # Packed columns are learned projections of ALL input features, then exposed as heads.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                q, k, v = (split(x @ weight) for weight in (self.wq, self.wk, self.wv))
                raw_scores = q @ k.swapaxes(-1, -2)
                scaled_scores = raw_scores / np.sqrt(self.d_head) if scaled else raw_scores
                scores = np.where(blocked, -np.inf, scaled_scores)
                shifted = scores - scores.max(axis=-1, keepdims=True)
                numerators = np.exp(shifted)
                attention = numerators / numerators.sum(axis=-1, keepdims=True)
                heads = attention @ v
                combined = heads.transpose(0, 2, 1, 3).reshape(batch, tokens, self.d_model)
                output = combined @ self.wo
            except FloatingPointError as exc:
                raise ValueError("attention exceeded finite float64 arithmetic; reduce input/weight magnitude") from exc
        if return_trace:
            return output, attention, {
                "x": x, "q": q, "k": k, "v": v, "raw": raw_scores,
                "scaled": scaled_scores, "masked": scores, "blocked": blocked,
                "attention": attention, "heads": heads, "concat": combined,
                "output": output,
            }
        return output, attention


def main():
    x = np.random.default_rng(68).normal(size=(2, 6, 32))
    layer = MultiHeadAttention()
    output, attention = layer.forward(x, causal=True)
    print("Synthetic inputs; untrained fixed projections; float64")
    print(f"Input: {x.shape}; output: {output.shape}; attention: {attention.shape}")
    print(f"Maximum row-sum error: {np.max(abs(attention.sum(-1) - 1)):.3e}")
    print(f"Largest future weight: {np.triu(attention, k=1).max():.3e}")


if __name__ == "__main__":
    main()
