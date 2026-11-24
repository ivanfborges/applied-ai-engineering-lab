"""Deterministic numerical fixtures shared by the Day 66 visual laboratory."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from from_scratch import scaled_dot_product_attention, softmax


COLORS = {"query": "#2563eb", "key": "#d97706", "value": "#15803d",
          "output": "#7c3aed", "mask": "#94a3b8"}
SEED = 66


@dataclass
class AttentionTrace:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    raw: np.ndarray
    scaled: np.ndarray
    allowed: np.ndarray
    masked: np.ndarray
    weights: np.ndarray
    output: np.ndarray


def trace(query, key, value, allowed=None) -> AttentionTrace:
    """Expose intermediates while reusing the analytical implementation's policy."""
    output, weights = scaled_dot_product_attention(query, key, value, allowed)
    q, k, v = [np.asarray(a, dtype=np.float64) for a in (query, key, value)]
    raw = q @ k.T
    scaled = raw / np.sqrt(q.shape[1])
    mask = np.ones(raw.shape, bool) if allowed is None else np.broadcast_to(allowed, raw.shape)
    return AttentionTrace(q, k, v, raw, scaled, mask, np.where(mask, scaled, -np.inf), weights, output)


def projection_fixture(seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(size=(4, 3)), 1)
    matrices = [np.round(rng.normal(size=(3, d)) / np.sqrt(3), 1) for d in (3, 3, 2)]
    return x, matrices, trace(*(x @ w for w in matrices))


def pipeline_fixture():
    q = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    k = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    v = np.array([[2., 0.], [0., 2.], [1., 1.]])
    return trace(q, k, v)


def geometry_state(yaw=30., pitch=20., query_norm=1., key_gain=1., unequal=False):
    if not all(np.isfinite([yaw, pitch, query_norm, key_gain])) or query_norm <= 0 or key_gain <= 0:
        raise ValueError("finite directions and positive vector magnitudes are required")
    a, b = np.deg2rad([yaw, pitch])
    q = query_norm * np.array([np.cos(b)*np.cos(a), np.cos(b)*np.sin(a), np.sin(b)])
    keys = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.] / np.sqrt(3)])
    gains = np.array([key_gain, 1.6 if unequal else 1., .55 if unequal else 1., 1.])
    keys *= gains[:, None]
    raw = q @ keys.T
    cosine = raw / (query_norm * np.linalg.norm(keys, axis=1))
    return dict(query=q, keys=keys, raw=raw, scaled=raw/np.sqrt(3),
                cosine=cosine, angles=np.rad2deg(np.arccos(np.clip(cosine, -1, 1))),
                weights=softmax([raw/np.sqrt(3)])[0])


def softmax_state(logits=(2., 1., 0., -1.), dimension=4, temperature=1.):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError("logits must be a nonempty finite vector")
    if not np.isfinite(temperature) or temperature <= 0 or dimension <= 0:
        raise ValueError("temperature and dimension must be positive")
    scaled = logits / np.sqrt(dimension) / temperature
    stabilized = scaled - scaled.max()
    exponentials = np.exp(stabilized)
    weights = softmax([scaled])[0]
    jacobian = np.diag(weights) - np.outer(weights, weights)
    return dict(raw=logits, scaled=scaled, stabilized=stabilized, exponentials=exponentials,
                weights=weights, entropy=entropy(weights[None]), maximum=float(weights.max()),
                jacobian_norm=float(np.linalg.norm(jacobian)))


def entropy(weights):
    weights = np.asarray(weights)
    terms = np.zeros_like(weights)
    positive = weights > 0
    terms[positive] = weights[positive] * np.log(weights[positive])
    return float(-terms.sum(axis=-1).mean())


def value_fixture():
    return np.array([[-1.5, -.8, 0.], [1.5, -.8, .2], [0., 1.6, .3], [0., 0., 2.]])


def weighted_values(weights, values=None):
    values = value_fixture() if values is None else np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(values),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("nonnegative finite weights with positive total are required")
    weights = weights / weights.sum()
    return weights, weights[:, None] * values, weights @ values


def scaling_experiment(quick=False, seed=SEED):
    rng = np.random.default_rng(seed)
    queries, keys = (128 if quick else 512), 64
    results = []
    for dimension in (4, 16, 64, 256, 1024):
        q, k = rng.normal(size=(queries, dimension)), rng.normal(size=(keys, dimension))
        raw = q @ k.T
        scaled = raw / np.sqrt(dimension)
        metrics = {}
        for name, scores in (("raw", raw), ("scaled", scaled)):
            p = softmax(scores)
            squared = p*p
            # Sum diagonal/off-diagonal squares without subtracting nearly equal
            # O(1) terms when a distribution is saturated. Prefix/suffix sums
            # exclude the diagonal without cancellation or batched Jacobians.
            before = np.pad(np.cumsum(squared, axis=1)[:, :-1], ((0, 0), (1, 0)))
            after = np.pad(np.cumsum(squared[:, ::-1], axis=1)[:, :-1][:, ::-1], ((0, 0), (0, 1)))
            norm = np.sqrt(((p*(1-p))**2 + squared*(before+after)).sum(axis=1))
            metrics[name] = dict(variance=float(scores.var()), theory_variance=float(dimension if name == "raw" else 1),
                                 entropy=entropy(p), max_probability=float(p.max(axis=1).mean()),
                                 jacobian_norm=float(norm.mean()), diagonal_mean=float((p*(1-p)).mean()))
        results.append(dict(dimension=dimension, queries=queries, keys=keys, raw_scores=raw,
                            scaled_scores=scaled, metrics=metrics))
    return results


def mask_fixture():
    rng = np.random.default_rng(SEED)
    q, k, v = rng.normal(size=(6, 3)), rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    return {"Full": trace(q, k, v), "Causal": trace(q, k, v, np.tri(6, dtype=bool)),
            "Padding": trace(q, k, v, np.array([True]*4 + [False]*2))}


def leakage_fixture():
    original = mask_fixture()
    q, k, v = original["Full"].query.copy(), original["Full"].key.copy(), original["Full"].value.copy()
    k[-1], v[-1] = [9., -7., 6.], [12., -10.]
    changed = {"Full": trace(q, k, v), "Causal": trace(q, k, v, np.tri(6, dtype=bool))}
    return original, changed


def context_fixture(financial=False):
    words = ("the customer deposited money at the bank" if financial else "the fisherman sat on the bank near river").split()
    x = np.zeros((len(words), 2))
    bank = words.index("bank")
    x[bank] = [.3, .3]
    semantic = {"fisherman": [1., 0.], "river": [2., 0.], "customer": [0., 1.],
                "money": [0., 2.], "deposited": [0., 1.5]}
    for i, word in enumerate(words):
        if word in semantic:
            x[i] = semantic[word]
    # Deliberately engineered context features, not learned lexical embeddings.
    q = np.ones_like(x)
    return dict(words=words, x=x, bank=bank, attention=trace(q, 2*x, x))


def cross_fixture():
    rng = np.random.default_rng(SEED)
    source = rng.normal(size=(5, 3))
    target = rng.normal(size=(3, 3))
    wq, wk, wv = [rng.normal(size=(3, 2))*.6 for _ in range(3)]
    return dict(source=source, target=target, wq=wq, wk=wk, wv=wv,
                self_attention=trace(source@wq, source@wk, source@wv),
                cross_attention=trace(target@wq, source@wk, source@wv))


def multihead_fixture():
    x = np.array([[1., 0., .2], [0., 1., .2], [-1., 0., .2], [0., -1., .2]])
    matrices = [np.array([[2., 0.], [0., .3], [0., .5]]),
                np.array([[.3, 0.], [0., 2.], [.5, 0.]]),
                np.array([[1., 1.], [1., -1.], [.2, .2]])]
    heads = []
    projections = []
    for i, wq in enumerate(matrices):
        wk, wv = np.roll(wq, i, axis=0), np.roll(wq, -i, axis=0)
        projections.append((wq, wk, wv))
        heads.append(trace(x@wq, x@wk, x@wv))
    joined = np.concatenate([h.output for h in heads], axis=1)
    wo = np.random.default_rng(SEED).normal(size=(6, 3)) / np.sqrt(6)
    return dict(x=x, projections=projections, heads=heads, concatenated=joined, wo=wo, output=joined@wo)


def memory_estimates(lengths=(128, 256, 512, 1024, 2048, 4096, 8192, 16384)):
    n = np.asarray(lengths, dtype=np.int64)
    if n.ndim != 1 or not len(n) or (n <= 0).any():
        raise ValueError("positive sequence lengths are required")
    return dict(lengths=n, interactions=n*n, fp32_mib=n*n*4/2**20, fp16_mib=n*n*2/2**20)


def complexity_experiment(quick=False, measure=True):
    """Bounded timings of this NumPy reference, not a kernel comparison."""
    lengths = (128, 256) if quick else (128, 256, 512, 1024, 2048)
    dimension, repeats, warmups = 32, (3 if quick else 5), 1
    rng = np.random.default_rng(SEED)
    rows = []
    for n in lengths:
        arrays = [rng.normal(size=(n, dimension)) for _ in range(3)]
        timings = []
        if measure:
            for _ in range(warmups):
                scaled_dot_product_attention(*arrays)
            for _ in range(repeats):
                start = perf_counter()
                scaled_dot_product_attention(*arrays)
                timings.append(perf_counter()-start)
        rows.append(dict(length=n, dimension=dimension, warmups=warmups, repetitions=repeats,
                         median_seconds=float(np.median(timings)) if timings else None,
                         samples_seconds=timings))
    return rows


def rag_fixture():
    query = np.array([1., .2])
    docs = np.array([[1., 0.], [0., 1.], [.8, .3]])
    cosine = docs@query / (np.linalg.norm(docs, axis=1)*np.linalg.norm(query))
    selected = np.argsort(-cosine, kind="stable")[:2]
    tokens = np.vstack([query, docs[selected]])
    return dict(query=query, docs=docs, cosine=cosine, selected=selected,
                attention=trace(tokens, tokens, tokens))


def cache_fixture():
    """One frozen synthetic layer; incremental queries checked against full prefixes."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(6, 3))
    wq, wk, wv = [rng.normal(size=(3, 2))*.5 for _ in range(3)]
    k_cache, v_cache = x[:3]@wk, x[:3]@wv
    states = []
    for index in range(3, 6):
        q = x[index:index+1]@wq
        k_cache = np.vstack([k_cache, x[index:index+1]@wk])
        v_cache = np.vstack([v_cache, x[index:index+1]@wv])
        cached = trace(q, k_cache.copy(), v_cache.copy())
        full = trace(x[:index+1]@wq, x[:index+1]@wk, x[:index+1]@wv, np.tri(index+1, dtype=bool))
        states.append(dict(index=index, query=q, cache=cached, full_output=full.output[-1],
                           error=float(np.max(abs(cached.output[0]-full.output[-1]))),
                           cached_projection_rows=1, recomputed_prefix_rows=index+1))
    return dict(x=x, wq=wq, wk=wk, wv=wv, states=states)
