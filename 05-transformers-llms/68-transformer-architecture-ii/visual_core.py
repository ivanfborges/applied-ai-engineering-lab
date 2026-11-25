"""Numerical fixtures and measured controls for the Day 68 visual laboratory."""

import numpy as np

from from_scratch import MultiHeadAttention
from example import entropy, torch_reference

HEAD_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#6B4C9A", "#555555"]
TOKENS = ["The", "curious", "engineer", "checks", "every", "head"]


def default_case(seed=168):
    """A visual-phase fixture distinct from the seed-68 analytical record."""
    x = np.random.default_rng(seed).normal(size=(1, 6, 16))
    layer = MultiHeadAttention(16, 4, seed)
    traces = {name: layer.forward(x, causal=causal, return_trace=True)[2]
              for name, causal in (("bidirectional", False), ("causal", True))}
    return layer, traces


def pca_view(x, dimensions=3):
    """Centered SVD with deterministic signs; report retained centered variance."""
    x = np.asarray(x, dtype=float)
    centered = x - x.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(dimensions, len(singular))
    components = vt[:count].copy()
    for row in components:
        row *= 1 if row[np.argmax(abs(row))] >= 0 else -1
    projected = centered @ components.T
    total = float(singular @ singular)
    retained = float(singular[:count] @ singular[:count] / total) if total else 1.0
    return np.pad(projected, ((0, 0), (0, dimensions - count))), retained


def pair_geometry(x):
    """Exact original-space pair distances and angles, not distances in PCA plots."""
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosines = np.divide(x @ x.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
    angles[denominator == 0] = np.nan
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    return distances, angles


def principal_angles(a, b):
    """Angles between column spans; no assumption of orthogonality across heads."""
    def basis(matrix):
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = np.sum(s > max(matrix.shape) * np.finfo(float).eps * s[0]) if s[0] else 0
        return u[:, :rank]
    qa, qb = basis(a), basis(b)
    return np.degrees(np.arccos(np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1, 1)))


def subspace_cases(seed=168):
    rng = np.random.default_rng(seed + 1)
    low = rng.normal(size=(6, 3))
    layer, traces = default_case(seed)
    high = traces["bidirectional"]["x"][0]
    cases = {}
    for name, x in (("direct", low), ("pca", high)):
        original, retained = (x, 1.0) if name == "direct" else pca_view(x)
        distances, angles = pair_geometry(x)
        entries = []
        for head in range(4):
            for kind, attribute in (("Q", "wq"), ("K", "wk"), ("V", "wv")):
                weight = rng.normal(size=(3, 3)) / np.sqrt(3) if name == "direct" else getattr(layer, attribute)[:, head * 4:(head + 1) * 4]
                projected = x @ weight
                visible, fraction = (projected, 1.0) if name == "direct" else pca_view(projected)
                pd, pa = pair_geometry(projected)
                entries.append({"head": head, "kind": kind, "weight": weight, "projected": projected,
                                "view": visible, "retained": fraction, "distances": pd, "angles": pa})
        overlap = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                overlap[i, j] = principal_angles(entries[i * 3]["weight"], entries[j * 3]["weight"]).min()
        cases[name] = {"x": x, "original": original, "retained": retained,
                       "distances": distances, "angles": angles, "entries": entries,
                       "minimum_principal_angles": overlap}
    return cases


def scaling_samples(seed=168, samples=4000):
    """Nested paired IID N(0,1) coordinates; no fitted or normalized components."""
    rng = np.random.default_rng(seed + 2)
    q = rng.normal(size=(samples, 128))
    k = rng.normal(size=(samples, 128))
    dots = np.cumsum(q * k, axis=1)
    distribution_q = rng.normal(size=128)
    distribution_k = rng.normal(size=(6, 128))
    row_logits = np.cumsum(distribution_k * distribution_q, axis=1)
    return [{"dimension": d, "raw": dots[:, d - 1], "scaled": dots[:, d - 1] / np.sqrt(d),
             "logits": row_logits[:, d - 1], "raw_variance": float(dots[:, d - 1].var()),
             "scaled_variance": float((dots[:, d - 1] / np.sqrt(d)).var())}
            for d in (2, 4, 8, 16, 32, 64, 128)]


def constructed_routing():
    """One shared scalar mixture cannot independently select two payload channels.

    Source one stores entity code=1 and age=0; source two stores entity=0, age=10.
    A shared attention coefficient a yields (a,10*(1-a)); target (1,10) is outside
    this line segment. Two separately routed channels approximate it. This is
    restricted to this value encoding, before arbitrary learned output changes.
    """
    payloads = np.array([[1., 0.], [0., 10.]])
    logits = np.array([[6., -6.], [-6., 6.]])
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    routed = np.array([weights[0] @ payloads[:, 0], weights[1] @ payloads[:, 1]])
    alphas = np.linspace(0, 1, 101)
    return {"payloads": payloads, "logits": logits, "weights": weights,
            "routed": routed, "target": np.array([1., 10.]),
            "shared_curve": np.column_stack((alphas, 10 * (1 - alphas))),
            "features": [[1, 0, 0, 0], [0, 1, 10, 1]],
            "feature_names": ["entity indicator", "source position", "numeric attribute", "context flag"]}


def pair_cosine(rows):
    rows = np.asarray(rows).reshape(len(rows), -1)
    norms = np.linalg.norm(rows, axis=1)
    denominator = norms[:, None] * norms[None, :]
    return np.divide(rows @ rows.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def diversity_case(seed=168):
    layer, original = default_case(seed)
    layer.wq[:, 4:8] = layer.wq[:, :4]
    layer.wk[:, 4:8] = layer.wk[:, :4]
    trace = layer.forward(original["bidirectional"]["x"], return_trace=True)[2]
    contributions = np.stack([trace["heads"][0, i] @ layer.wo[i * 4:(i + 1) * 4] for i in range(4)])
    ablation = np.linalg.norm(contributions.reshape(4, -1), axis=1)
    return {"trace": trace, "map_similarity": pair_cosine(trace["attention"][0]),
            "output_similarity": pair_cosine(trace["heads"][0]),
            "entropy": [entropy(trace["attention"][:, i:i + 1]) for i in range(4)],
            "contributions": contributions, "ablation_l2": ablation,
            "identical_map_error": float(np.max(abs(trace["attention"][:, 0] - trace["attention"][:, 1]))),
            "distinct_value_output_difference": float(np.linalg.norm(trace["heads"][:, 0] - trace["heads"][:, 1]))}


def architecture_counts(query_heads, kv_heads, head_dimension):
    for value in (query_heads, kv_heads, head_dimension):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("head counts and dimensions must be positive integers")
    if query_heads % kv_heads:
        raise ValueError("KV heads must divide query heads")
    width = query_heads * head_dimension
    return {"d_model": width, "parameters": 2 * width ** 2 + 2 * width * kv_heads * head_dimension,
            "assignment": np.arange(query_heads) // (query_heads // kv_heads)}


def kv_bytes(batch, layers, length, kv_heads, head_dimension, bits=16):
    for value in (batch, layers, length, kv_heads, head_dimension):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError("cache dimensions must be positive integers")
    if bits not in (4, 8, 16, 32):
        raise ValueError("supported storage widths are 4, 8, 16 and 32 bits")
    return int(2 * batch * layers * length * kv_heads * head_dimension * bits // 8)


def complexity_counts(length, width, heads, batch=1, bits=16):
    architecture_counts(heads, heads, width // heads)
    if width % heads or length <= 0 or batch <= 0 or bits not in (4, 8, 16, 32):
        raise ValueError("invalid width, length, batch or storage precision")
    return {"projection_macs": 4 * batch * length * width ** 2,
            "attention_macs": 2 * batch * length ** 2 * width,
            "map_bytes": batch * heads * length ** 2 * bits / 8,
            "decode_attention_macs": 2 * batch * length * width}


def online_attention(q, k, v, block_size=2):
    """Educational tiled online softmax, not a GPU FlashAttention implementation."""
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    maximum = np.full((len(q), 1), -np.inf)
    denominator = np.zeros((len(q), 1))
    accumulator = np.zeros((len(q), v.shape[-1]))
    stages = []
    for start in range(0, len(k), block_size):
        scores = q @ k[start:start + block_size].T / np.sqrt(q.shape[-1])
        updated_maximum = np.maximum(maximum, scores.max(-1, keepdims=True))
        previous_scale = np.exp(maximum - updated_maximum)
        weights = np.exp(scores - updated_maximum)
        denominator = previous_scale * denominator + weights.sum(-1, keepdims=True)
        accumulator = previous_scale * accumulator + weights @ v[start:start + block_size]
        maximum = updated_maximum
        stages.append({"start": start, "end": min(start + block_size, len(k)), "scores": scores,
                       "maximum": maximum.copy(), "denominator": denominator.copy(),
                       "partial_output": accumulator / denominator})
    return accumulator / denominator, stages


def numerical_report(seed=168, samples=4000):
    layer, traces = default_case(seed)
    measurements = []
    for mode, trace in traces.items():
        output, maps = torch_reference(layer, trace["x"], causal=mode == "causal")
        measurements.append({"mode": mode,
                             "output_max_abs_error": float(np.max(abs(output - trace["output"]))),
                             "map_max_abs_error": float(np.max(abs(maps - trace["attention"]))),
                             "row_sum_error": float(np.max(abs(maps.sum(-1) - 1)))})
        np.testing.assert_allclose(output, trace["output"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(maps, trace["attention"], atol=1e-12, rtol=1e-12)
    changed = traces["causal"]["x"].copy()
    changed[:, 4] += np.random.default_rng(seed + 3).normal(size=16) * 5
    causal, _ = layer.forward(changed, causal=True)
    full, _ = layer.forward(changed)
    prefix = float(np.max(abs(causal[:, :4] - traces["causal"]["output"][:, :4])))
    np.testing.assert_allclose(causal[:, :4], traces["causal"]["output"][:, :4], atol=1e-12)
    diversity = diversity_case(seed)
    trace = traces["bidirectional"]
    tiled, stages = online_attention(trace["q"][0, 0], trace["k"][0, 0], trace["v"][0, 0])
    tiled_error = float(np.max(abs(tiled - trace["heads"][0, 0])))
    np.testing.assert_allclose(tiled, trace["heads"][0, 0], atol=1e-12)
    return {"parity": measurements, "causal_prefix_max_abs_change": prefix,
            "bidirectional_prefix_max_abs_change": float(np.max(abs(full[:, :4] - traces["bidirectional"]["output"][:, :4]))),
            "identical_map_error": diversity["identical_map_error"],
            "distinct_value_output_l2": diversity["distinct_value_output_difference"],
            "online_softmax_max_abs_error": tiled_error, "online_blocks": len(stages),
            "scaling": [{key: row[key] for key in ("dimension", "raw_variance", "scaled_variance")}
                        for row in scaling_samples(seed, samples)],
            "constructed_routing": constructed_routing()["routed"],
            "cache_gib": {str(h): kv_bytes(1, 32, 8192, h, 128) / 2 ** 30 for h in (32, 8, 1)}}
