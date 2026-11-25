"""One fixed, untrained encoder-decoder block; NumPy, single head, no autograd."""

import numpy as np


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def matrix(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return array


def sinusoidal_positions(length, width):
    length, width = positive_int(length, "length"), positive_int(width, "width")
    angles = np.arange(length)[:, None] / 10000 ** (np.arange(0, width, 2) / width)
    positions = np.empty((length, width))
    positions[:, 0::2] = np.sin(angles)
    positions[:, 1::2] = np.cos(angles[:, :width // 2])
    return positions


def layer_norm(x, eps=1e-5):
    x = matrix(x, "x")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + eps)


class Attention:
    """Parameters are sampled once; True in a mask means blocked, as in nn.Transformer."""

    def __init__(self, width, rng):
        self.width = positive_int(width, "width")
        self.wq, self.wk, self.wv, self.wo = (
            rng.normal(0, width ** -0.5, (width, width)) for _ in range(4)
        )

    def __call__(self, query, memory, blocked=None):
        query, memory = matrix(query, "query"), matrix(memory, "memory")
        if query.shape[1] != self.width or memory.shape[1] != self.width:
            raise ValueError("query and memory widths must match the attention width")
        scores = (query @ self.wq) @ (memory @ self.wk).T / np.sqrt(self.width)
        if blocked is not None:
            blocked = np.asarray(blocked)
            if blocked.dtype != bool or blocked.shape != scores.shape:
                raise ValueError("blocked must be a boolean query-length by memory-length mask")
            if blocked.all(axis=-1).any():
                raise ValueError("each query must have at least one visible key")
            scores = np.where(blocked, -np.inf, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        return (weights @ (memory @ self.wv)) @ self.wo, weights


class EncoderDecoderBlock:
    """Post-norm composition; no dropout, biases or affine layer-norm parameters."""

    def __init__(self, width=8, hidden=32, seed=42):
        self.width = positive_int(width, "width")
        hidden = positive_int(hidden, "hidden")
        rng = np.random.default_rng(seed)
        self.encoder_attention = Attention(width, rng)
        self.decoder_attention = Attention(width, rng)
        self.cross_attention = Attention(width, rng)
        self.encoder_ffn = (
            rng.normal(0, width ** -0.5, (width, hidden)),
            rng.normal(0, hidden ** -0.5, (hidden, width)),
        )
        self.decoder_ffn = (
            rng.normal(0, width ** -0.5, (width, hidden)),
            rng.normal(0, hidden ** -0.5, (hidden, width)),
        )

    @staticmethod
    def feed_forward(x, weights):
        first, second = weights
        return np.maximum(0, x @ first) @ second

    def __call__(self, source, target, causal=True):
        source, target = matrix(source, "source"), matrix(target, "target")
        if source.shape[1] != self.width or target.shape[1] != self.width:
            raise ValueError("source and target widths must match the block width")
        if not isinstance(causal, bool):
            raise ValueError("causal must be boolean")
        x = source + sinusoidal_positions(len(source), self.width)
        y = target + sinusoidal_positions(len(target), self.width)
        attended, encoder_weights = self.encoder_attention(x, x)
        memory = layer_norm(x + attended)
        memory = layer_norm(memory + self.feed_forward(memory, self.encoder_ffn))
        blocked = np.triu(np.ones((len(y), len(y)), dtype=bool), k=1) if causal else None
        attended, decoder_weights = self.decoder_attention(y, y, blocked)
        y = layer_norm(y + attended)
        attended, cross_weights = self.cross_attention(y, memory)
        y = layer_norm(y + attended)
        y = layer_norm(y + self.feed_forward(y, self.decoder_ffn))
        return memory, y, {"encoder": encoder_weights, "decoder": decoder_weights, "cross": cross_weights}


def main():
    rng = np.random.default_rng(7)
    source, target = rng.normal(size=(5, 8)), rng.normal(size=(3, 8))
    block = EncoderDecoderBlock()
    memory, decoded, weights = block(source, target)
    changed = target.copy()
    changed[2] += rng.normal(size=8)
    _, other, _ = block(source, changed)
    print("Synthetic, untrained, fixed-parameter forward pass")
    print("memory", memory.shape, "decoder", decoded.shape)
    for name, attention in weights.items():
        print(name, attention.shape, "maximum row-sum error", float(abs(attention.sum(axis=1) - 1).max()))
    print("earlier-state change after changing future input", float(abs(decoded[:2] - other[:2]).max()))


if __name__ == "__main__":
    main()
