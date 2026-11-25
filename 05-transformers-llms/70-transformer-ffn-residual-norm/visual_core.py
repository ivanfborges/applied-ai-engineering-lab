"""Deterministic numerical mechanisms for the Day 70 visual laboratory."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from example import TransformerBlock, matched_stacks, numpy_parity
from from_scratch import feed_forward, gelu, layer_norm, layer_norm_vjp


@dataclass(frozen=True)
class Config:
    seed: int = 270
    device: str = "cpu"
    depth: int = 24
    d_model: int = 64
    expansion: float = 4.0
    quick: bool = False

    @property
    def d_ff(self) -> int:
        return max(1, round(self.d_model * self.expansion))

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(sorted(set([6, 12, self.depth] + ([] if self.quick else [24, 48]))))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(self.seed + i for i in range(2 if self.quick else 3))

    @property
    def tokens(self) -> int:
        return 8 if self.quick else 16

    def validate(self) -> None:
        for name in ("seed", "depth", "d_model"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**32 - 1000:
            raise ValueError("seed must be in [0, 2**32 - 1000)")
        if not 1 <= self.depth <= 96 or not 4 <= self.d_model <= 256 or self.d_model % 4:
            raise ValueError("depth must be 1..96; d_model must be a multiple of 4 in 4..256")
        if not math.isfinite(self.expansion) or not 0.25 <= self.expansion <= 8:
            raise ValueError("expansion must be finite in 0.25..8")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")


def activation(x, name: str):
    x = np.asarray(x, dtype=np.float64)
    if name == "identity":
        return x.copy()
    if name == "relu":
        return np.maximum(x, 0)
    if name == "gelu":
        return gelu(x)
    if name == "silu":
        # Split the sigmoid to avoid overflowing exp(-x) for large negative x.
        sigmoid = np.empty_like(x)
        positive = x >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-x[positive]))
        e = np.exp(x[~positive])
        sigmoid[~positive] = e / (1 + e)
        return x * sigmoid
    raise ValueError(f"Unknown activation: {name}")


def activation_derivative(x, name: str):
    t = torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=True)
    functions = {"identity": lambda z: z, "relu": nn.functional.relu,
                 "gelu": lambda z: nn.functional.gelu(z, approximate="tanh"), "silu": nn.functional.silu}
    if name not in functions:
        raise ValueError(f"Unknown activation: {name}")
    return torch.autograd.grad(functions[name](t).sum(), t)[0].numpy()


def norm_steps(x, gamma=None, beta=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.ones(x.shape[-1]) if gamma is None else np.asarray(gamma)
    beta = np.zeros(x.shape[-1]) if beta is None else np.asarray(beta)
    output = layer_norm(x, gamma, beta, eps)
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    variance = np.mean(centered**2, axis=-1, keepdims=True)
    std = np.sqrt(variance + eps)
    normalized = centered / std
    return {"input": x, "mean": mean, "centered": centered, "variance": variance,
            "std": std, "normalized": normalized, "scaled": normalized * gamma,
            "gamma": gamma, "beta": beta, "eps": eps, "output": output}


def rms_norm(x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.size == 0 or not np.isfinite(x).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm requires a nonempty finite tensor and positive epsilon")
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def batch_norm(x, training=True, running_mean=None, running_var=None, eps=1e-5):
    """B,T,D layout; equivalent to BatchNorm1d on B,D,T with affine disabled."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("BatchNorm fixture must be finite B,T,D")
    if training:
        if x.shape[0] * x.shape[1] < 2:
            raise ValueError("training BatchNorm needs at least two values per channel")
        mean, variance = x.mean(axis=(0, 1)), x.var(axis=(0, 1))
    else:
        mean, variance = np.asarray(running_mean), np.asarray(running_var)
        if mean.shape != (x.shape[-1],) or variance.shape != mean.shape or np.any(variance < 0):
            raise ValueError("running moments must have shape (D,) and nonnegative variance")
    return (x - mean) / np.sqrt(variance + eps)


def ffn_fixture(seed=270):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, 8))
    w1, w2 = rng.normal(size=(8, 32)) / np.sqrt(8), rng.normal(size=(32, 8)) / np.sqrt(32)
    b1, b2 = rng.normal(scale=.1, size=32), rng.normal(scale=.1, size=8)
    expanded = x @ w1 + b1
    hidden = gelu(expanded)
    return {"x": x, "w1": w1, "w2": w2, "b1": b1, "b2": b2,
            "expanded": expanded, "hidden": hidden, "output": feed_forward(x, w1, b1, w2, b2)}


def geometry_fixture(seed=270, samples=96):
    """Use selected first three hidden coordinates, never independently fitted PCAs."""
    rng = np.random.default_rng(seed + 1)
    labels = np.arange(samples) % 3
    centers = np.array([[-1.5, 0, 0], [1, 1.5, .5], [0, -.5, 1.8]])
    x = centers[labels] + rng.normal(scale=.3, size=(samples, 3))
    w1, w2 = rng.normal(size=(3, 12)) / np.sqrt(3), rng.normal(size=(12, 3)) / np.sqrt(12)
    b1, b2 = rng.normal(scale=.1, size=12), rng.normal(scale=.1, size=3)
    expanded = x @ w1 + b1
    variants = {name: [x, expanded[:, :3], activation(expanded, name)[:, :3],
                       activation(expanded, name) @ w2 + b2] for name in ("identity", "relu", "gelu")}
    affine_error = np.max(np.abs(variants["identity"][-1] - (x @ (w1 @ w2) + b1 @ w2 + b2)))
    return {"x": x, "labels": labels, "ids": np.arange(samples), "w1": w1, "w2": w2,
            "variants": variants, "affine_error": float(affine_error),
            "coordinate_policy": "original 3D input/output; selected hidden coordinates 0,1,2; arrows between stages cross feature spaces, not physical motion"}


def residual_fixture(seed=270, layers=8, alpha=1.):
    rng = np.random.default_rng(seed + 2)
    weights = rng.normal(scale=.18, size=(layers, 3, 3))
    biases = rng.normal(scale=.08, size=(layers, 3))
    x = np.array([.8, -.4, .6])
    trajectories = {}
    for residual in (False, True):
        states = [x.copy()]
        for w, b in zip(weights, biases):
            update = np.tanh(states[-1] @ w + b)
            states.append(states[-1] + alpha * update if residual else update)
        trajectories["residual" if residual else "replacement"] = np.array(states)
    return {"x": x, "weights": weights, "biases": biases,
            "first_update": np.tanh(x @ weights[0] + biases[0]), **trajectories}


def scalar_gradient(a: float, depth: int, alpha: float = 1., residual=False):
    """Compute magnitude in log space and retain exact zeros and float64 limits."""
    if not np.isfinite([a, alpha]).all() or not isinstance(depth, int) or depth < 0:
        raise ValueError("finite a/alpha and nonnegative integer depth required")
    base = abs(1 + alpha * a) if residual else abs(a)
    if depth == 0:
        return {"value": 1., "log10": 0., "status": "finite"}
    if base == 0:
        return {"value": 0., "log10": None, "status": "exact zero"}
    log10 = depth * math.log10(base)
    if log10 > math.log10(np.finfo(float).max):
        return {"value": None, "log10": log10, "status": "float64 overflow"}
    if log10 < math.log10(np.nextafter(0., 1.)):
        return {"value": 0., "log10": log10, "status": "float64 underflow"}
    return {"value": float(base**depth), "log10": log10, "status": "finite"}


def parameter_counts(d=64, h=256, layers=1, gated=False, bias=True):
    for value in (d, h, layers):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("dimensions and layers must be positive integers")
    attention = 4 * d * d + (4 * d if bias else 0)
    ffn = (3 if gated else 2) * d * h + ((2 * h + d if gated else h + d) if bias else 0)
    norms = 4 * d
    return {"attention": attention * layers, "ffn": ffn * layers,
            "normalization": norms * layers, "total": (attention + ffn + norms) * layers}


def operation_counts(d=64, h=256, tokens=16, layers=1, gated=False, cached=False):
    """MACs per sequence or one new token; excludes softmax/activation/norm/bias."""
    parameter_counts(d, h, layers, gated)
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("tokens must be a positive integer")
    positions = 1 if cached else tokens
    attention = 4 * positions * d * d + 2 * (tokens if cached else tokens**2) * d
    ffn = (3 if gated else 2) * positions * d * h
    return {"attention_macs": attention * layers, "ffn_macs": ffn * layers,
            "total_macs": (attention + ffn) * layers}


def swiglu_fixture(seed=270, d=8, dense_h=32, bias=True):
    rng = np.random.default_rng(seed + 3)
    # Choose the closest positive width including the stated bias convention.
    target = parameter_counts(d, dense_h, bias=bias)["ffn"]
    h = min(range(1, dense_h + 1), key=lambda k: abs(parameter_counts(d, k, gated=True, bias=bias)["ffn"] - target))
    x = rng.normal(size=(4, d))
    gate_w, up_w, down_w = [rng.normal(size=s) / np.sqrt(s[0]) for s in [(d, h), (d, h), (h, d)]]
    gate_b, up_b, down_b = [rng.normal(scale=.1, size=n) if bias else np.zeros(n) for n in [h, h, d]]
    gate, features = x @ gate_w + gate_b, x @ up_w + up_b
    activated = activation(gate, "silu")
    multiplied = activated * features
    return {"x": x, "gate_projection": gate, "features": features, "activated_gate": activated,
            "gated_features": multiplied, "output": multiplied @ down_w + down_b,
            "dense_width": dense_h, "gated_width": h, "bias": bias,
            "dense_count": target, "gated_count": parameter_counts(d, h, gated=True, bias=bias)["ffn"]}


def _rms(tensor):
    return tensor.detach().square().mean().sqrt().item()


def block_trace(block: TransformerBlock, x):
    """Observe actual module operations, retaining the full autograd graph."""
    n1 = block.norm1(x) if block.order == "pre" else x
    attention = block.attention(n1, n1, n1, need_weights=False)[0]
    attention_sum = x + attention if block.residual else attention
    r1 = attention_sum if block.order == "pre" else block.norm1(attention_sum)
    n2 = block.norm2(r1) if block.order == "pre" else r1
    expansion = block.ffn[0](n2)
    hidden = block.ffn[1](expansion)
    projection = block.ffn[2](hidden)
    final_sum = r1 + projection if block.residual else projection
    output = final_sum if block.order == "pre" else block.norm2(final_sum)
    return {"input": x, "norm1": n1, "attention": attention, "residual1": r1,
            "norm2": n2, "expansion": expansion, "gelu": hidden,
            "projection": projection, "residual2": final_sum, "output": output}


def backward_decomposition(seed=270):
    blocks = matched_stacks(seed, 1, 8, 2, 32)["pre_residual"]
    block = blocks[0]
    generator = torch.Generator().manual_seed(seed + 50)
    x = torch.randn(1, 4, 8, generator=generator, dtype=torch.float64, requires_grad=True)
    target = torch.randn(1, 4, 8, generator=generator, dtype=torch.float64)
    trace = block_trace(block, x)
    loss = nn.functional.mse_loss(trace["output"], target)
    upstream = torch.autograd.grad(loss, trace["output"], retain_graph=True)[0]
    ffn_path = torch.autograd.grad(trace["projection"], trace["residual1"], upstream, retain_graph=True)[0]
    middle = upstream + ffn_path
    attention_path = torch.autograd.grad(trace["attention"], x, middle, retain_graph=True)[0]
    input_gradient = torch.autograd.grad(loss, x, retain_graph=True)[0]
    intermediates = {key: torch.autograd.grad(loss, value, retain_graph=True)[0].detach().numpy()
                     for key, value in trace.items()}
    error = (input_gradient - middle - attention_path).abs().max().item()
    return {"trace": {k: v.detach().numpy() for k, v in trace.items()},
            "gradients": intermediates, "loss": loss.item(),
            "contributions": {"output upstream / FFN identity": upstream.detach().numpy(),
                              "FFN learned path": ffn_path.detach().numpy(),
                              "middle / attention identity": middle.detach().numpy(),
                              "attention learned path": attention_path.detach().numpy(),
                              "input total": input_gradient.detach().numpy()},
            "decomposition_max_abs_error": error}


def dense_diagnostics(config: Config):
    """Matched tanh dense networks; no normalization, dropout or optimizer."""
    config.validate()
    results = []
    for seed in config.seeds:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 4)
            layers = nn.ModuleList([nn.Linear(config.d_model, config.d_model).double() for _ in range(max(config.depths))])
        generator = torch.Generator().manual_seed(seed + 100)
        data = torch.randn(2, config.tokens, config.d_model, generator=generator, dtype=torch.float64).to(config.device)
        target = torch.randn(data.shape, generator=generator, dtype=torch.float64).to(config.device)
        for depth in config.depths:
            for residual in (False, True):
                model = copy.deepcopy(layers[:depth]).to(config.device)
                state = data.detach().clone().requires_grad_(True)
                states = [state]
                for layer in model:
                    update = torch.tanh(layer(state))
                    state = state + update if residual else update
                    state.retain_grad()
                    states.append(state)
                loss = nn.functional.mse_loss(state, target)
                loss.backward()
                denominator = states[-1].grad.norm().item()
                for i, value in enumerate(states):
                    gradient = value.grad.detach()
                    finite = bool(torch.isfinite(gradient).all() and torch.isfinite(value).all())
                    results.append({"experiment": "dense-tanh", "seed": seed, "depth": depth,
                                    "condition": "residual" if residual else "plain", "layer": i,
                                    "activation_rms": _rms(value), "gradient_l2": gradient.norm().item(),
                                    "gradient_rms": _rms(gradient),
                                    "relative_gradient": gradient.norm().item() / denominator if denominator else None,
                                    "loss": loss.item(), "finite": finite,
                                    "gradient_samples": gradient.flatten().cpu().numpy()[::max(1, gradient.numel() // 64)]})
                if not all(r["finite"] for r in results):
                    raise FloatingPointError("nonfinite empirical dense diagnostic; see configuration")
    return results


def transformer_diagnostics(config: Config):
    """Real attention stacks and measured per-sublayer residual updates."""
    config.validate()
    rows = []
    for seed in config.seeds:
        stacks = matched_stacks(seed, max(config.depths), config.d_model, 4, config.d_ff)
        reference = stacks["pre_residual"].state_dict()
        if not all(torch.equal(value, stacks["post_residual"].state_dict()[key]) for key, value in reference.items()):
            raise AssertionError("Pre/Post parameters differ")
        generator = torch.Generator().manual_seed(seed + 200)
        data = torch.randn(2, config.tokens, config.d_model, generator=generator, dtype=torch.float64).to(config.device)
        target = torch.randn(data.shape, generator=generator, dtype=torch.float64).to(config.device)
        for depth in config.depths:
            for order, final_norm in (("pre", False), ("post", False), ("pre", True)):
                blocks = copy.deepcopy(stacks[order + "_residual"][:depth]).to(config.device)
                state = data.detach().clone().requires_grad_(True)
                states, traces = [state], []
                for block in blocks:
                    trace = block_trace(block, state)
                    state = trace["output"]
                    state.retain_grad()
                    states.append(state)
                    traces.append(trace)
                readout = nn.functional.layer_norm(state, (config.d_model,), eps=1e-5) if final_norm else state
                loss = nn.functional.mse_loss(readout, target)
                loss.backward()
                denominator = states[-1].grad.norm().item()
                for i, value in enumerate(states):
                    gradient = value.grad.detach()
                    row = {"experiment": "transformer", "seed": seed, "depth": depth,
                           "condition": order + ("+final-LN" if final_norm else ""), "layer": i,
                           "activation_rms": _rms(value), "gradient_l2": gradient.norm().item(),
                           "gradient_rms": _rms(gradient),
                           "relative_gradient": gradient.norm().item() / denominator if denominator else None,
                           "loss": loss.item(), "parameters_matched": True,
                           "finite": bool(torch.isfinite(value).all() and torch.isfinite(gradient).all())}
                    if i:
                        trace, previous = traces[i - 1], states[i - 1]
                        attention_rms, ffn_rms = _rms(trace["attention"]), _rms(trace["projection"])
                        stream1, stream2 = _rms(previous), _rms(trace["residual1"])
                        row.update(attention_update_rms=attention_rms, ffn_update_rms=ffn_rms,
                                   attention_update_ratio=attention_rms / stream1 if stream1 else None,
                                   ffn_update_ratio=ffn_rms / stream2 if stream2 else None,
                                   consecutive_cosine=nn.functional.cosine_similarity(previous.detach().flatten(), value.detach().flatten(), dim=0).item(),
                                   parameter_gradient_l2=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in blocks[i - 1].parameters())).item()))
                    rows.append(row)
                if any(not r["finite"] for r in rows):
                    raise FloatingPointError("nonfinite Transformer diagnostic")
    return rows


def normalization_interventions(seed=270):
    rng = np.random.default_rng(seed + 5)
    base = rng.normal(size=(2, 4, 3)) + np.array([1., -1., 2.])
    changed = base.copy()
    changed[1, 2] += np.array([5., -3., 1.])
    added = np.concatenate([base, rng.normal(loc=8, size=(1, 4, 3))], axis=0)
    # Update running moments using PyTorch: train variance is biased, running variance unbiased.
    bn = nn.BatchNorm1d(3, affine=False, momentum=.1).double()
    bn(torch.tensor(base).transpose(1, 2))
    running_mean, running_var = bn.running_mean.numpy().copy(), bn.running_var.numpy().copy()
    variants = {"base": base, "scale x2": base * 2, "offset +3": base + 3,
                "modified token": changed, "added sample": added, "single sample": base[:1]}
    results = {}
    for name, x in variants.items():
        results[name] = {"input": x,
                         "LayerNorm": layer_norm(x, np.ones(3), np.zeros(3)),
                         "RMSNorm": rms_norm(x), "BatchNorm train": batch_norm(x),
                         "BatchNorm eval": batch_norm(x, False, running_mean, running_var)}
    coupling = {method: float(np.max(np.abs(results["base"][method][0] - results["modified token"][method][0])))
                for method in results["base"] if method != "input"}
    return {"variants": results, "running_mean": running_mean, "running_var": running_var,
            "unmodified_sample_max_change": coupling, "layout": "B,T,D -> B,D,T for BatchNorm1d; reduce B,T per D channel"}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def numerical_checks(seed=270):
    fixture = ffn_fixture(seed)
    altered = fixture["x"].copy()
    altered[0] += 10
    changed = feed_forward(altered, fixture["w1"], fixture["b1"], fixture["w2"], fixture["b2"])
    independence = np.max(np.abs(changed[1:] - fixture["output"][1:]))
    coupling = normalization_interventions(seed)["unmodified_sample_max_change"]
    backward = backward_decomposition(seed)
    parity = numpy_parity(seed)
    if max(parity.values()) > 1e-10 or independence != 0 or backward["decomposition_max_abs_error"] > 1e-10:
        raise AssertionError("Numerical parity, position independence or VJP reconstruction failed")
    if coupling["LayerNorm"] != 0 or coupling["RMSNorm"] != 0 or coupling["BatchNorm eval"] != 0 or coupling["BatchNorm train"] <= 0:
        raise AssertionError("Normalization axes or running-statistic independence failed")
    return {"numpy_pytorch": parity, "ffn_other_tokens_change": float(independence),
            "linear_composition_error": geometry_fixture(seed)["affine_error"],
            "normalization_coupling": coupling,
            "backward_decomposition_error": backward["decomposition_max_abs_error"],
            "intentional_scalar_extremes": {
                "zero": scalar_gradient(0., 48), "cancellation": scalar_gradient(-1., 48, residual=True),
                "overflow": scalar_gradient(100., 1000), "underflow": scalar_gradient(.001, 1000)},
            "interpretation_status": "author review required"}
