"""Matched CPU Transformer diagnostics at initialization; zero optimizer steps."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn

from from_scratch import feed_forward, layer_norm, layer_norm_vjp


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class TransformerBlock(nn.Module):
    """Bidirectional attention + GELU FFN, two residual sublayers, no dropout."""

    def __init__(self, d_model=64, n_heads=4, d_ff=256, order="pre", residual=True):
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads), ("d_ff", d_ff)):
            _positive_int(value, name)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if order not in ("pre", "post") or not isinstance(residual, bool):
            raise ValueError("order must be pre/post and residual must be boolean")
        self.d_model, self.order, self.residual = d_model, order, residual
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(approximate="tanh"), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model, eps=1e-5), nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.d_model or 0 in x.shape:
            raise ValueError("x must have nonempty shape (batch, tokens, d_model)")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("x must be finite floating-point values")
        q = self.norm1(x) if self.order == "pre" else x
        update = self.attention(q, q, q, need_weights=False)[0]
        x = x + update if self.residual else update
        if self.order == "post":
            x = self.norm1(x)
        z = self.norm2(x) if self.order == "pre" else x
        update = self.ffn(z)
        x = x + update if self.residual else update
        return self.norm2(x) if self.order == "post" else x


def matched_stacks(seed=70, depth=24, d_model=64, n_heads=4, d_ff=256):
    """Copy parameters once; depth layers are independently initialized, not tied."""
    _positive_int(depth, "depth")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        base = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff).double() for _ in range(depth)
        ])
    stacks = {}
    for order, residual in (("pre", True), ("post", True), ("pre", False), ("post", False)):
        blocks = copy.deepcopy(base)
        for block in blocks:
            block.order, block.residual = order, residual
        stacks[f"{order}_{'residual' if residual else 'no_residual'}"] = blocks
    return stacks


def profile_stack(blocks, inputs, target, final_norm=False):
    """Measure block-boundary states/gradients of mean MSE with a random target."""
    if not isinstance(final_norm, bool):
        raise ValueError("final_norm must be boolean")
    if inputs.shape != target.shape or not torch.isfinite(target).all():
        raise ValueError("target must be finite and have exactly the input shape")
    blocks.zero_grad(set_to_none=True)
    state = inputs.detach().clone().requires_grad_(True)
    states = [state]
    for block in blocks:
        state = block(state)
        state.retain_grad()
        states.append(state)
    # A shared optional readout control; raw block-boundary states stay observable.
    readout = nn.functional.layer_norm(state, (state.shape[-1],), eps=1e-5) if final_norm else state
    loss = nn.functional.mse_loss(readout, target, reduction="mean")
    loss.backward()
    parameters = list(blocks.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("missing or nonfinite parameter gradient")
    if any(s.grad is None or not torch.isfinite(s.grad).all() for s in states):
        raise FloatingPointError("missing or nonfinite state gradient")
    rms = [s.detach().square().mean().sqrt().item() for s in states]
    grads = [s.grad.norm().item() for s in states]
    ffn_grads = [b.ffn[0].weight.grad.norm().item() for b in blocks]
    result = {
        "loss": loss.item(),
        "state_rms": rms,
        "state_gradient_l2": grads,
        "ffn_expansion_weight_gradient_l2": ffn_grads,
        "input_to_output_gradient_ratio": grads[0] / grads[-1] if grads[-1] > 0 else None,
    }
    if not np.isfinite([loss.item(), *rms, *grads, *ffn_grads]).all():
        raise FloatingPointError("nonfinite diagnostic")
    return result


def numpy_parity(seed=70):
    """Compare independent NumPy formulas with copied PyTorch affine parameters."""
    rng = np.random.default_rng(seed)
    x, upstream = rng.normal(size=(2, 3, 8)), rng.normal(size=(2, 3, 8))
    gamma, beta = rng.normal(size=8), rng.normal(size=8)
    w1, w2 = rng.normal(size=(8, 32)), rng.normal(size=(32, 8))
    b1, b2 = rng.normal(size=32), rng.normal(size=8)
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    out = nn.functional.layer_norm(tx, (8,), torch.tensor(gamma), torch.tensor(beta), 1e-5)
    vjp = torch.autograd.grad(out, tx, torch.tensor(upstream))[0]
    ffn = nn.functional.gelu(tx @ torch.tensor(w1) + torch.tensor(b1), approximate="tanh") @ torch.tensor(w2) + torch.tensor(b2)
    return {
        "layer_norm_max_abs_error": float(np.max(np.abs(layer_norm(x, gamma, beta) - out.detach().numpy()))),
        "layer_norm_vjp_max_abs_error": float(np.max(np.abs(layer_norm_vjp(x, gamma, upstream) - vjp.detach().numpy()))),
        "ffn_max_abs_error": float(np.max(np.abs(feed_forward(x, w1, b1, w2, b2) - ffn.detach().numpy()))),
    }


def run_diagnostics(seeds=(70, 71, 72), depth=24, d_model=64, n_heads=4, d_ff=256, final_norm=False):
    if not seeds:
        raise ValueError("provide at least one seed")
    runs = []
    for seed in seeds:
        stacks = matched_stacks(seed, depth, d_model, n_heads, d_ff)
        generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
        inputs = torch.randn(2, 8, d_model, generator=generator, dtype=torch.float64)
        target = torch.randn(2, 8, d_model, generator=generator, dtype=torch.float64)
        reference = stacks["pre_residual"].state_dict()
        matched = all(
            torch.equal(value, stack.state_dict()[name])
            for stack in stacks.values() for name, value in reference.items()
        )
        if not matched:
            raise AssertionError("initial parameters are not identical across conditions")
        runs.append({
            "seed": seed, "parameters_matched": matched,
            "parameter_count_per_stack": sum(p.numel() for p in stacks["pre_residual"].parameters()),
            "conditions": {name: profile_stack(stack, inputs, target, final_norm) for name, stack in stacks.items()},
        })
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "configuration": {
            "data": "synthetic independent standard-normal input and target tensors",
            "device": "cpu", "dtype": "float64", "threads": torch.get_num_threads(),
            "depth": depth, "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
            "batch": 2, "tokens": 8, "eps": 1e-5, "final_norm": final_norm,
            "attention": "bidirectional, no mask", "dropout": 0.0,
            "position_encoding": False, "optimizer_steps": 0,
            "loss": "mean MSE to independent random target", "gelu": "tanh approximation",
            "initialization": "PyTorch module defaults, copied across conditions",
        },
        "hypothesis": "Residual identity paths and norm placement change initialization-time signal propagation.",
        "numpy_parity": numpy_parity(), "runs": runs,
        "interpretation_status": "author review required",
        "limitation": "No training, natural-language data, quality ranking, latency benchmark or production evidence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[70, 71, 72])
    parser.add_argument("--final-norm", action="store_true", help="apply the same final non-affine LayerNorm to every condition")
    parser.add_argument("--output", type=Path, help="optional new JSON record; existing files are never overwritten")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output exists; choose a new dated filename")
    torch.set_num_threads(1)
    report = run_diagnostics(args.seeds, args.depth, final_norm=args.final_norm)
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
