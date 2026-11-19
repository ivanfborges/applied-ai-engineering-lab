"""CPU forward/backward diagnostics at initialization, without optimizer steps."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install the canonical extra: python -m pip install -e ".[dev,deep-learning]"') from exc

from from_scratch import positive_int


STRATEGIES = ("small", "xavier", "he", "he_layernorm", "large")


class DeepReLU(nn.Module):
    """Constant-width, bias-free blocks; optional LayerNorm before ReLU."""

    def __init__(self, width=128, depth=20, strategy="he", seed=58):
        super().__init__()
        width = positive_int(width, "width")
        depth = positive_int(depth, "depth")
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 - 2:
            raise ValueError("seed must be an integer in [0, 2**32 - 2)")
        self.width = width
        # Restore the caller's RNG; a private generator pairs initializer draws.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.linears = nn.ModuleList([
                nn.Linear(width, width, bias=False, dtype=torch.float64) for _ in range(depth)
            ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(width, eps=1e-5, dtype=torch.float64)
            if strategy == "he_layernorm" else nn.Identity() for _ in range(depth)
        ])
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        for linear in self.linears:
            if strategy in ("small", "large"):
                nn.init.normal_(linear.weight, std=0.06 if strategy == "small" else 0.18, generator=generator)
            elif strategy == "xavier":
                nn.init.xavier_normal_(linear.weight, gain=1.0, generator=generator)
            else:
                nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu", generator=generator)

    def forward(self, inputs):
        if (inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] != self.width
                or inputs.device.type != "cpu" or inputs.dtype != torch.float64
                or not torch.isfinite(inputs).all()):
            raise ValueError("inputs must be a finite nonempty CPU float64 batch with matching width")
        activations = [inputs]
        for layer, (linear, norm) in enumerate(zip(self.linears, self.norms), 1):
            h = torch.relu(norm(linear(activations[-1])))
            if not torch.isfinite(h).all():
                raise FloatingPointError(f"Non-finite activation at layer {layer}")
            activations.append(h)
        return activations


def rms(tensor):
    value = tensor.detach().square().mean().sqrt()
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite diagnostic RMS; reduce depth or scale")
    return float(value)


def diagnose(model, inputs, probe):
    """Measure two vector-Jacobian products; never update model parameters."""
    inputs = inputs.detach().clone().requires_grad_(True)
    activations = model(inputs)
    output = activations[-1]
    if (probe.shape != output.shape or probe.dtype != output.dtype
            or probe.device != output.device or not torch.isfinite(probe).all()
            or rms(probe) == 0):
        raise ValueError("probe must be finite, nonzero and match output shape, dtype and device")
    probe = probe.detach() / rms(probe)
    loss = output.square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite squared-output loss")
    weights = [linear.weight for linear in model.linears]
    gradients = torch.autograd.grad(loss, activations + weights, retain_graph=True)
    probe_gradients = torch.autograd.grad(output, activations, grad_outputs=probe)
    rows = []
    for index, (h, grad_loss, grad_probe) in enumerate(zip(activations, gradients, probe_gradients)):
        h = h.detach()
        rows.append({
            "layer": index,
            "second_moment": rms(h)**2,
            "mean": float(h.mean()),
            "variance": float(h.var(unbiased=False)),
            "zero_fraction": float((h == 0).double().mean()),
            "loss_activation_gradient_rms": rms(grad_loss),
            "probe_activation_gradient_rms": rms(grad_probe),
        })
    return {
        "squared_output_loss": float(loss.detach()),
        "layers": rows,
        "loss_weight_gradient_rms": [rms(g) for g in gradients[len(activations):]],
    }


def run_study(*, width=128, depth=20, batch=128, seed=58):
    batch = positive_int(batch, "batch")
    # Construct once before drawing data so invalid configuration fails explicitly.
    first_model = DeepReLU(width, depth, STRATEGIES[0], seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    inputs = torch.randn(batch, width, dtype=torch.float64, generator=generator)
    probe = torch.randn(batch, width, dtype=torch.float64, generator=generator)
    results = {}
    for index, strategy in enumerate(STRATEGIES):
        model = first_model if index == 0 else DeepReLU(width, depth, strategy, seed)
        results[strategy] = diagnose(model, inputs, probe)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": seed, "weight_seed": seed + 1, "width": width, "depth": depth, "batch": batch,
            "dtype": "float64", "device": "cpu", "threads": torch.get_num_threads(),
            "data": "synthetic standard Gaussian; no labels or external dataset",
            "loss": "mean(a_depth**2)", "probe": "shared Gaussian output adjoint normalized to RMS=1",
            "small_std": 0.06, "large_std": 0.18, "xavier_gain": 1.0,
            "he_mode": "fan_in", "layernorm": "before ReLU; eps=1e-5; gamma=1; beta=0",
            "optimizer_steps": 0,
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "platform": platform.platform()},
        "hypothesis": "Scale changes compound across ReLU layers; LayerNorm controls forward scale but changes the Jacobian.",
        "interpretation_status": "author review required",
        "limitations": ["one seed, finite width, synthetic Gaussian inputs", "no supervised training or generalization evidence",
                        "one probe is not a Jacobian spectrum", "loss gradient also depends on output scale"],
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--output", type=Path, help="Optional JSON path; use ignored outputs/ for generated records")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report = run_study(width=args.width, depth=args.depth, batch=args.batch, seed=args.seed)
    except (ValueError, FloatingPointError) as exc:
        parser.error(str(exc))
    print("Synthetic CPU float64 diagnostic; no optimizer steps. Interpretation: author review required.")
    print("strategy        q_depth/q_input   loss        dL/da_input RMS   probe input RMS   probe output RMS")
    for strategy, result in report["results"].items():
        first, last = result["layers"][0], result["layers"][-1]
        print(f"{strategy:15s} {last['second_moment']/first['second_moment']:14.6e} "
              f"{result['squared_output_loss']:11.4e} {first['loss_activation_gradient_rms']:17.6e} "
              f"{first['probe_activation_gradient_rms']:17.6e} {last['probe_activation_gradient_rms']:17.6e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Keep dated records; an explicit path must not silently replace an old run.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
