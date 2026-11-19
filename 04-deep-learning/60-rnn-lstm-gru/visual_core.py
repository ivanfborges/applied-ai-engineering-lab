"""Small, testable recurrent mechanisms; all controls are educational interventions."""

import numpy as np
import torch

from from_scratch import SimpleLSTMCell, finite_real, integer, sigmoid


def rnn_trace(sequence, strength=1.0):
    x = np.asarray(sequence, dtype=float)
    if x.ndim != 1 or not 1 <= len(x) <= 100 or not np.isfinite(x).all():
        raise ValueError("sequence must contain 1..100 finite scalars")
    strength = finite_real(strength, "strength")
    wx = np.array([0.8, -0.5])
    wh = strength * np.array([[0.8, 0.25], [-0.1, 0.7]])
    states = [np.zeros(2)]
    for value in x:
        states.append(np.tanh(wx * value + wh @ states[-1]))
    return np.asarray(states)


def scalar_gradients(a, transitions=50):
    a = finite_real(a, "a")
    transitions = integer(transitions, "transitions", 0)
    if abs(a) > 2 or transitions > 200:
        raise ValueError("bounded illustration requires |a| <= 2 and transitions <= 200")
    return np.abs(a) ** np.arange(transitions + 1)


def lstm_control(c_prev, forget, write, candidate, expose):
    values = [finite_real(v, n) for v, n in zip(
        (c_prev, forget, write, candidate, expose), ("c_prev", "forget", "write", "candidate", "expose"))]
    c_prev, forget, write, candidate, expose = values
    if not all(0 <= v <= 1 for v in (forget, write, expose)) or abs(candidate) > 1:
        raise ValueError("gates must lie in [0,1] and candidate in [-1,1]")
    retained, written = forget * c_prev, write * candidate
    cell = retained + written
    return dict(retained=retained, written=written, cell=cell, hidden=expose * np.tanh(cell))


def gru_control(h_prev, x, update, reset, convention="pytorch"):
    h_prev, x, update, reset = [finite_real(v, n) for v, n in zip(
        (h_prev, x, update, reset), ("h_prev", "x", "update", "reset"))]
    if not 0 <= update <= 1 or not 0 <= reset <= 1 or convention not in ("pytorch", "before"):
        raise ValueError("gates must lie in [0,1]; convention must be pytorch or before")
    # Scalar weights make the recurrent-bias placement difference visible too.
    recurrent = reset * (0.9 * h_prev + 0.3) if convention == "pytorch" else 0.9 * (reset * h_prev) + 0.3
    candidate = np.tanh(0.8 * x + 0.1 + recurrent)
    retained, written = update * h_prev, (1 - update) * candidate
    return dict(candidate=candidate, retained=retained, written=written, hidden=retained + written)


def computed_lstm(sequence=(1, 0, 0.2, -0.4, 0, 0, 0, 0)):
    cell = SimpleLSTMCell(1, 1, seed=160, forget_bias=1)
    h, c = np.zeros(1), np.zeros(1)
    rows = []
    for value in sequence:
        previous_h, previous_c = h.copy(), c.copy()
        h, c, gates = cell.step([value], h, c)
        rows.append(dict(x=float(value), h_prev=float(previous_h[0]), c_prev=float(previous_c[0]),
                         **{key: float(v[0]) for key, v in gates.items()},
                         **lstm_control(previous_c[0], gates["f"][0], gates["i"][0], gates["g"][0], gates["o"][0])))
    return rows


def computed_gru(sequence=(1, 0, 0.2, -0.4, 0, 0, 0, 0)):
    h = 0.0
    rows = []
    for value in sequence:
        z, r = float(sigmoid(0.6 * value + 0.4 * h + 0.5)), float(sigmoid(-0.4 * value + 0.3 * h))
        result = gru_control(h, value, z, r)
        rows.append(dict(x=float(value), h_prev=h, z=z, r=r, **result))
        h = float(result["hidden"])
    return rows


def input_gradients(model, x, y):
    """Mean per-example L2 norm of d BCE_i / d x_it, not gradient of mean BCE.

    Independent examples allow one summed loss to recover each individual
    loss gradient without cancellation or an extra 1/N scaling factor.
    """
    model.eval()
    values = x.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(values), y, reduction="sum")
    gradients, = torch.autograd.grad(loss, values)
    norms = gradients.norm(dim=-1).mean(dim=0)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("non-finite input gradients")
    return norms.detach().numpy()


def pca_states(states):
    """Fit one PCA to all provided sequence/timestep states, including h0.

    Axes are local to one architecture/checkpoint. Sign normalization is only
    for reproducible presentation; it does not align different architectures.
    """
    values = np.asarray(states, dtype=float)
    if values.ndim != 3 or min(values.shape) < 1 or values.shape[2] < 3 or not np.isfinite(values).all():
        raise ValueError("states must be finite (sequences,timesteps,hidden>=3)")
    flat = values.reshape(-1, values.shape[-1])
    centered = flat - flat.mean(0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    if len(singular) < 3:
        raise ValueError("PCA requires at least three observations")
    components = vectors[:3].copy()
    for vector in components:
        vector *= 1 if vector[np.argmax(abs(vector))] >= 0 else -1
    variance = singular**2
    ratios = variance[:3] / variance.sum() if variance.sum() > 0 else np.zeros(3)
    return (centered @ components.T).reshape(*values.shape[:2], 3), ratios


def causal_attention(length=6, query=None):
    """Deterministic, untrained Q/K/V example. Future positions have zero weight."""
    length = integer(length, "length", 2)
    if length > 50:
        raise ValueError("length must be <= 50")
    query = length - 1 if query is None else integer(query, "query", 0)
    if query >= length:
        raise ValueError("query must be within sequence")
    t = np.arange(length, dtype=float)
    embeddings = np.column_stack([np.sin(t), np.cos(t), t / length])
    scores = embeddings @ embeddings.T / np.sqrt(3)
    mask = np.arange(length)[None, :] <= np.arange(length)[:, None]
    scores = np.where(mask, scores, -np.inf)
    weights = np.exp(scores - scores.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    return weights, weights[query] @ embeddings, embeddings
