"""Numerical fixtures for the visual lab; synthetic, untrained and CPU bounded."""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.decomposition import PCA

from from_scratch import absolute_position_lookup, apply_rope, causal_alibi_bias, sinusoidal_encoding


@dataclass(frozen=True)
class Config:
    seq_len: int = 48
    dim: int = 16
    base: float = 10000.0
    frames: int = 24
    fps: int = 6
    seed: int = 269
    dpi: int = 100

    def validate(self):
        for name in ("seq_len", "dim", "frames", "fps", "seed", "dpi"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if not 8 <= self.seq_len <= 512:
            raise ValueError("seq_len must be 8–512; long-context views use sampled positions.")
        if not 4 <= self.dim <= 128 or self.dim % 2:
            raise ValueError("dim must be even and between 4 and 128.")
        if not np.isfinite(self.base) or not 2 <= self.base <= 1e8:
            raise ValueError("base must be finite and between 2 and 1e8.")
        if not 4 <= self.frames <= 80 or not 1 <= self.fps <= 12:
            raise ValueError("Require 4–80 frames and 1–12 fps.")
        if self.seed < 0 or not 60 <= self.dpi <= 160:
            raise ValueError("Require nonnegative seed and dpi 60–160.")
        return self


METHODS = ("none", "sinusoidal", "absolute", "relative", "rope", "alibi")
TOKENS = ["The", "dog", "chased", "the", "cat", "."]
PERMUTATION = np.array([0, 4, 2, 3, 1, 5])


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("Softmax needs a nonempty final axis.")
    if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits).any(axis=-1).all():
        raise ValueError("Every row needs a finite logit; NaN/+inf are unsupported.")
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    return weights / weights.sum(-1, keepdims=True)


def fixture(config, length=6):
    rng = np.random.default_rng(config.seed)
    weights = rng.normal(size=(3, config.dim, config.dim)) / np.sqrt(config.dim)
    x = rng.normal(size=(length, config.dim))
    if length == 6:
        x[3] = x[0]  # Repeated "the" shares content; occurrence IDs remain distinct.
    table = rng.normal(scale=.5, size=(max(128, length), config.dim))
    return x, weights, table


def attention(x, weights, *, method="none", positions=None, table=None, base=10000., causal=False, slope=.1):
    """One head with identical projections across mechanisms; bias after scaling."""
    x, weights = np.asarray(x, dtype=float), np.asarray(weights, dtype=float)
    if x.ndim != 2 or weights.shape != (3, x.shape[1], x.shape[1]):
        raise ValueError("Require x (tokens, dim) and weights (3, dim, dim).")
    if method not in METHODS or not np.isfinite(x).all() or not np.isfinite(weights).all():
        raise ValueError("Require a known method and finite inputs.")
    n, dim = x.shape
    positions = np.arange(n) if positions is None else np.asarray(positions)
    if positions.shape != (n,):
        raise ValueError("One position is required per input token.")
    pe = sinusoidal_encoding(positions, dim, base)
    if method == "sinusoidal":
        x = x + pe
    elif method == "absolute":
        if table is None:
            raise ValueError("absolute requires a supplied synthetic table.")
        x = x + absolute_position_lookup(table, positions)
    q, k, v = (x @ weight for weight in weights)
    if method == "rope":
        q, k = apply_rope(q, positions, base), apply_rope(k, positions, base)
    raw = q @ k.T / np.sqrt(dim)
    distance = positions[:, None] - positions[None, :]
    bias = np.zeros_like(raw)
    if method == "relative":
        bias = -.08 * np.abs(distance)  # Explicit deterministic scalar bias, not Shaw's full method.
    elif method == "alibi":
        if not causal:
            raise ValueError("This ALiBi example requires causal=True.")
        bias = causal_alibi_bias(positions, positions, [slope])[0]
    logits = raw + bias
    if causal:
        logits = np.where(distance >= 0, logits, -np.inf)
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(np.maximum(probs, np.finfo(float).tiny)), axis=-1)
    return {"input": x, "q": q, "k": k, "v": v, "raw_logits": raw,
            "logits": logits, "probabilities": probs, "output": probs @ v,
            "entropy": entropy, "expected_distance": np.sum(probs * np.abs(distance), axis=-1)}


def permutation_data(config):
    x, weights, table = fixture(config)
    p, inverse = PERMUTATION, np.argsort(PERMUTATION)
    pe = sinusoidal_encoding(np.arange(6), config.dim, config.base)
    results = {}
    for method in ("none", "sinusoidal"):
        original = attention(x, weights, method=method, base=config.base)
        permuted = attention(x[p], weights, method=method, base=config.base)
        aligned = permuted["output"][inverse]
        results[method] = {"original": original, "permuted": permuted, "aligned_output": aligned,
                           "error": float(np.max(np.abs(aligned - original["output"])))}
    joint = attention((x + pe)[p], weights)["output"][inverse]
    results["joint_error"] = float(np.max(np.abs(joint - results["sinusoidal"]["original"]["output"])))
    results.update(x=x, weights=weights, table=table, permutation=p, inverse=inverse,
                   tokens=TOKENS, permuted_tokens=[TOKENS[i] for i in p])
    return results


def projected_geometry(config, dim=None):
    dim = config.dim if dim is None else dim
    length = max(128, config.seq_len)
    positions = np.arange(length)
    pe = sinusoidal_encoding(positions, dim, config.base)
    pca = PCA(n_components=3, svd_solver="full").fit(pe)
    coords = pca.transform(pe)
    exact_dims = [0, 1, 2]
    norms = np.linalg.norm(pe, axis=-1)
    previous_cos = np.ones(length)
    previous_cos[1:] = np.sum(pe[1:] * pe[:-1], axis=-1) / (norms[1:] * norms[:-1])
    previous_distance = np.zeros(length)
    previous_distance[1:] = np.linalg.norm(pe[1:] - pe[:-1], axis=-1)
    return {"positions": positions, "coordinates": coords, "exact": pe[:, exact_dims],
            "exact_dimensions": exact_dims, "norm": norms, "previous_cosine": previous_cos,
            "previous_distance": previous_distance, "components": pca.components_, "mean": pca.mean_,
            "explained_variance_ratio": pca.explained_variance_ratio_, "fit_positions": length}


def addition_data(config):
    x, _, _ = fixture(config)
    pe = sinusoidal_encoding(np.arange(6), config.dim, config.base)
    combined = x + pe
    # PCA determines a shared basis; vectors are projected without affine centering.
    pca = PCA(n_components=3, svd_solver="full").fit(np.concatenate([x, pe, combined]))
    basis = pca.components_.T
    return {"content": x, "position": pe, "combined": combined, "basis": basis,
            "projected_content": x @ basis, "projected_position": pe @ basis,
            "projected_combined": combined @ basis,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "projection_rule": "Vectors use x @ basis, without subtracting the fitted mean; addition stays linear."}


def rope_planes(query_position=2., key_position=5., frequency=.5, query_angle=.3, key_angle=1., query_norm=1., key_norm=.8):
    values = np.array([query_position, key_position, frequency, query_angle, key_angle, query_norm, key_norm], dtype=float)
    if not np.isfinite(values).all() or min(query_norm, key_norm) < 0 or frequency <= 0:
        raise ValueError("Require finite controls, positive frequency and nonnegative norms.")
    q = query_norm * np.array([np.cos(query_angle), np.sin(query_angle)])
    k = key_norm * np.array([np.cos(key_angle), np.sin(key_angle)])
    def rotate(vector, angle):
        return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ vector
    qr, kr = rotate(q, query_position * frequency), rotate(k, key_position * frequency)
    return {"q": q, "k": k, "rotated_q": qr, "rotated_k": kr,
            "relative_angle": (key_position - query_position) * frequency,
            "dot_before": float(q @ k), "dot_after": float(qr @ kr),
            "relative_dot": float(q @ rotate(k, (key_position - query_position) * frequency))}


def multifrequency_data(config, position=5):
    rng = np.random.default_rng(config.seed + 1)
    q, k = rng.normal(size=(2, 1, config.dim))
    qr, kr = apply_rope(q, [2], config.base), apply_rope(k, [position], config.base)
    contributions = (qr * kr).reshape(-1, 2).sum(-1)
    positions = np.arange(max(128, config.seq_len))
    trajectory = apply_rope(np.repeat(q, len(positions), axis=0), positions, config.base)
    pca = PCA(n_components=3, svd_solver="full").fit(trajectory)
    return {"q": q[0], "k": k[0], "query_position": 2, "key_position": position,
            "rotated_q": qr[0], "rotated_k": kr[0], "contributions": contributions,
            "dot": float(qr[0] @ kr[0]), "coordinates": pca.transform(trajectory),
            "explained_variance_ratio": pca.explained_variance_ratio_}


def alibi_data(length=32, query=24, slope=.1, strength=0., relevant=2):
    if not 2 <= length <= 128 or not 0 <= query < length or not 0 <= relevant < length:
        raise ValueError("Require length 2–128 and valid query/relevant indices.")
    if not np.isfinite(strength) or not 0 <= strength <= 30:
        raise ValueError("Content strength must be between 0 and 30 logit units.")
    content = np.zeros(length)
    content[relevant] = strength
    bias = causal_alibi_bias([query], np.arange(length), [slope])[0, 0]
    logits = content + bias
    probs = softmax(logits)
    return {"content": content, "bias": bias, "logits": logits, "probabilities": probs,
            "expected_distance": float(probs @ np.abs(query - np.arange(length)))}


def precision_data(config):
    rng = np.random.default_rng(config.seed + 2)
    q, k = rng.normal(size=(2, config.dim))
    origins = [0, 128, 4096, 1000000, 100000000, 1000000000]
    rows = []
    reference = float(q @ apply_rope(k[None], [5], config.base)[0])
    for origin in origins:
        row = {"position": origin, "displacement": 5}
        for dtype, name in ((np.float32, "float32"), (np.float64, "float64")):
            frequency = np.exp(-np.arange(0, config.dim, 2, dtype=dtype) * np.log(dtype(config.base)) / dtype(config.dim))
            def rotate(vector, position):
                angle = dtype(position) * frequency
                even, odd = vector.astype(dtype)[0::2], vector.astype(dtype)[1::2]
                result = np.empty(config.dim, dtype=dtype)
                result[0::2] = even * np.cos(angle) - odd * np.sin(angle)
                result[1::2] = even * np.sin(angle) + odd * np.cos(angle)
                return result
            dot = float(rotate(q, origin) @ rotate(k, origin + 5))
            row[name + "_dot"] = dot
            row[name + "_error"] = abs(dot - reference)
            row[name + "_represented_offset"] = float(dtype(origin + 5) - dtype(origin))
        rows.append(row)
    return {"rows": rows, "reference_relative_dot_float64": reference,
            "construction": "phase, trig and dot all use the named dtype; float32 also rounds input vectors"}


def numerical_report(config):
    perm = permutation_data(config)
    pe = sinusoidal_encoding(np.arange(config.seq_len), config.dim, config.base)
    frequency = config.base ** (-np.arange(0, config.dim, 2) / config.dim)
    formula = np.cos(5 * frequency).sum()
    dot = pe[0] @ pe[5]
    rope = rope_planes()
    shifted = rope_planes(102, 105)
    add = addition_data(config)
    x, weights, table = fixture(config, min(config.seq_len, 128))
    methods = {method: attention(x, weights, method=method, table=table, base=config.base, causal=True)
               for method in METHODS}
    return {
        "permutation_no_position_error": perm["none"]["error"],
        "permutation_fixed_position_difference": perm["sinusoidal"]["error"],
        "permutation_joint_coordinates_error": perm["joint_error"],
        "sinusoidal_norm_error": float(np.max(np.abs(np.linalg.norm(pe, axis=-1) - np.sqrt(config.dim / 2)))),
        "sinusoidal_relative_dot_error": float(abs(dot - formula)),
        "rope_2d_common_shift_error": abs(rope["dot_after"] - shifted["dot_after"]),
        "linear_projection_addition_error": float(np.max(np.abs(add["projected_combined"] - add["projected_content"] - add["projected_position"]))),
        "method_probability_row_errors": {name: float(np.max(np.abs(value["probabilities"].sum(-1) - 1))) for name, value in methods.items()},
        "pca_explained_variance_ratio": projected_geometry(config)["explained_variance_ratio"],
        "precision": precision_data(config),
        "alibi_flat_content": [{"slope": slope, "expected_distance": alibi_data(32, 24, slope)["expected_distance"]}
                               for slope in (.02, .1, .5, 1.)],
        "configuration": asdict(config),
    }


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
