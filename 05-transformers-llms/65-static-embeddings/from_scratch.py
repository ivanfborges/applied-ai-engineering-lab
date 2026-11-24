"""Small, dense NumPy embedding learners for inspecting objectives and gradients."""

from dataclasses import dataclass
import json

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def positive_float(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def prepare_corpus(corpus):
    """Accept pretokenized sentences; never silently split a raw string."""
    if isinstance(corpus, (str, bytes)):
        raise ValueError("corpus must contain tokenized sentences")
    sentences = []
    for sentence in corpus:
        if isinstance(sentence, (str, bytes)):
            raise ValueError("each sentence must be a sequence of tokens")
        tokens = list(sentence)
        if not tokens or any(not isinstance(w, str) or not w or w.strip() != w for w in tokens):
            raise ValueError("sentences must contain nonempty, stripped string tokens")
        sentences.append(tokens)
    if not sentences:
        raise ValueError("corpus must not be empty")
    words = tuple(sorted({word for sentence in sentences for word in sentence}))
    vocab = {word: i for i, word in enumerate(words)}
    return [[vocab[w] for w in sentence] for sentence in sentences], words


def training_data(corpus, window=2):
    """Return CBOW means, directed Skip-gram pairs and inverse-distance counts."""
    positive_int(window, "window")
    sentences, words = prepare_corpus(corpus)
    n = len(words)
    contexts, centers, pairs = [], [], []
    counts = np.zeros((n, n), dtype=np.float64)
    for sentence in sentences:
        for t, center in enumerate(sentence):
            positions = [j for j in range(max(0, t - window), min(len(sentence), t + window + 1)) if j != t]
            if not positions:
                continue
            row = np.zeros(n)
            for j in positions:
                context = sentence[j]
                row[context] += 1.0 / len(positions)
                pairs.append((center, context))
                counts[center, context] += 1.0 / abs(j - t)
            contexts.append(row)
            centers.append(center)
    if not pairs:
        raise ValueError("corpus needs at least one within-sentence context pair")
    pairs = np.asarray(pairs, dtype=int)
    return words, np.asarray(contexts), np.asarray(centers), pairs, counts


def predictive_loss_grad(inputs, outputs, features, targets):
    """Mean full-softmax NLL and gradients of both embedding tables.

    Features contain context means (CBOW) or one-hot centers (Skip-gram).
    The caller supplies aligned, finite arrays from training_data.
    """
    hidden = features @ inputs
    logits = hidden @ outputs.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(targets)), targets].mean()
    delta = np.exp(log_probs)
    delta[np.arange(len(targets)), targets] -= 1
    delta /= len(targets)
    return float(loss), (features.T @ (delta @ outputs), delta.T @ hidden)


def glove_loss_grad(inputs, outputs, bias, context_bias, counts, x_max=10.0, alpha=0.75):
    """Mean weighted squared residual over positive counts only.

    This numerical kernel expects validated, aligned parameter/count arrays.
    Missing observations contribute neither log(0) nor a residual penalty.
    """
    observed = counts > 0
    weights = np.zeros_like(counts)
    weights[observed] = np.minimum(counts[observed] / x_max, 1.0) ** alpha
    residual = inputs @ outputs.T + bias[:, None] + context_bias[None, :]
    residual[observed] -= np.log(counts[observed])
    residual[~observed] = 0
    m = np.count_nonzero(observed)
    loss = np.sum(weights * residual**2) / m
    delta = 2 * weights * residual / m
    return float(loss), (delta @ outputs, delta.T @ inputs, delta.sum(axis=1), delta.sum(axis=0))


@dataclass
class EmbeddingFit:
    words: tuple[str, ...]
    vectors: np.ndarray
    losses: np.ndarray


def _optimize(parameters, objective, epochs, lr, recorder=None):
    """All gradients refer to the same pre-update parameters."""
    history = []
    for step in range(epochs + 1):
        loss, gradients = objective(*parameters)
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
            raise FloatingPointError("training diverged; reduce lr or inspect inputs")
        history.append(loss)
        if recorder is not None:
            recorder(step, parameters, loss)
        if step < epochs:
            for parameter, gradient in zip(parameters, gradients):
                parameter -= lr * gradient
    return np.asarray(history)


def train_predictive(corpus, mode="cbow", dim=8, window=2, epochs=400, lr=0.8, seed=65, recorder=None):
    """Educational batch Word2Vec architectures with full softmax, not SGNS."""
    if mode not in ("cbow", "skipgram"):
        raise ValueError("mode must be cbow or skipgram")
    for value, name in ((dim, "dim"), (epochs, "epochs")):
        positive_int(value, name)
    positive_float(lr, "lr")
    words, contexts, centers, pairs, _ = training_data(corpus, window)
    if mode == "cbow":
        features, targets = contexts, centers
    else:
        features, targets = np.eye(len(words))[pairs[:, 0]], pairs[:, 1]
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0, 0.1, (len(words), dim))
    outputs = rng.normal(0, 0.1, inputs.shape)
    losses = _optimize([inputs, outputs], lambda v, u: predictive_loss_grad(v, u, features, targets), epochs, lr, recorder)
    return EmbeddingFit(words, inputs.copy(), losses)


def train_glove(corpus, dim=8, window=2, epochs=400, lr=0.8, seed=65, x_max=10.0, alpha=0.75, recorder=None):
    """Educational batch gradient descent; the original GloVe uses AdaGrad."""
    for value, name in ((dim, "dim"), (epochs, "epochs")):
        positive_int(value, name)
    for value, name in ((lr, "lr"), (x_max, "x_max"), (alpha, "alpha")):
        positive_float(value, name)
    words, _, _, _, counts = training_data(corpus, window)
    rng = np.random.default_rng(seed)
    inputs = rng.normal(0, 0.1, (len(words), dim))
    outputs = rng.normal(0, 0.1, inputs.shape)
    bias, context_bias = np.zeros(len(words)), np.zeros(len(words))
    objective = lambda v, u, b, c: glove_loss_grad(v, u, b, c, counts, x_max, alpha)
    losses = _optimize([inputs, outputs, bias, context_bias], objective, epochs, lr, recorder)
    return EmbeddingFit(words, inputs + outputs, losses)


def unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or min(vectors.shape) < 1 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a nonempty finite matrix")
    # Scale first to avoid overflow/underflow in the Euclidean norm.
    scales = np.max(np.abs(vectors), axis=1, keepdims=True)
    if np.any(scales == 0):
        raise ValueError("cosine is undefined for zero vectors")
    scaled = vectors / scales
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("cosine requires equal-length one-dimensional vectors")
    normalized = unit_rows(np.stack([a, b]))
    return float(np.clip(normalized[0] @ normalized[1], -1, 1))


def nearest(words, vectors, query, k=3, exclude=()):
    """Cosine ranking in original dimensions; exact ties use lexical order."""
    positive_int(k, "k")
    normalized = unit_rows(vectors)
    if len(words) != len(normalized) or len(set(words)) != len(words):
        raise ValueError("words must be unique and aligned with vectors")
    query = np.asarray(query, dtype=float)
    if query.shape != (normalized.shape[1],):
        raise ValueError("query dimension must match vectors")
    scores = np.clip(normalized @ unit_rows(query[None, :])[0], -1, 1)
    excluded = set(exclude)
    order = sorted((i for i, word in enumerate(words) if word not in excluded), key=lambda i: (-scores[i], words[i]))
    return [(words[i], float(scores[i])) for i in order[:k]]


def word_vector(fit, word):
    if word not in fit.words:
        raise ValueError(f"out-of-vocabulary word: {word}")
    return fit.vectors[fit.words.index(word)]


def analogy(fit, a, b, c, k=3):
    """Raw-vector offset b - a + c followed by cosine (sources excluded)."""
    query = word_vector(fit, b) - word_vector(fit, a) + word_vector(fit, c)
    return nearest(fit.words, fit.vectors, query, k, exclude=(a, b, c))


if __name__ == "__main__":
    fit = train_glove([["ice", "water", "cold"], ["steam", "water", "hot"]])
    print(json.dumps({"data": "synthetic six-token corpus", "numpy": np.__version__,
                      "configuration": {"dim": 8, "window": 2, "epochs": 400, "lr": 0.8,
                                        "seed": 65, "x_max": 10.0, "alpha": 0.75},
                      "initial_loss": fit.losses[0], "final_loss": fit.losses[-1],
                      "interpretation": "author review required"}, indent=2, allow_nan=False))
