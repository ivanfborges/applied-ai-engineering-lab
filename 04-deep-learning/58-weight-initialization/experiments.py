"""Deterministic CPU experiments for the separate Day 58 visual lab."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install python -m pip install -e ".[dev,deep-learning]"') from exc


SCALES = {"small": 0.25, "xavier": 1 / math.sqrt(2), "he": 1., "large": 1.6}
COLORS = {"small": "#c08615", "xavier": "#2878b5", "he": "#16846b",
          "large": "#c74440", "he_layer": "#8458aa"}


@dataclass(frozen=True)
class Config:
    depth: int = 30
    width: int = 64
    batch: int = 64
    seed: int = 58
    activation: str = "relu"
    normalization: str = "none"

    def __post_init__(self):
        for name, low, high in (("depth", 1, 50), ("width", 2, 128), ("batch", 2, 128), ("seed", 0, 2**31-10)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.activation not in ("relu", "tanh") or self.normalization not in ("none", "layer", "batch"):
            raise ValueError("Use relu/tanh and none/layer/batch normalization")


class SignalNet(nn.Module):
    """Bias-zero hidden layers, common Gaussian samples; optional linear head."""

    def __init__(self, config, strategy="he", *, multiplier=None, input_size=None, supervised=False):
        super().__init__()
        if strategy not in SCALES:
            raise ValueError(f"Unknown strategy: {strategy}")
        scale = SCALES[strategy] if multiplier is None else multiplier
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("multiplier must be positive and finite")
        self.config = config
        sizes = [input_size or config.width] + [config.width] * config.depth
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.linears = nn.ModuleList([
                nn.Linear(a, b, dtype=torch.float64) for a, b in zip(sizes[:-1], sizes[1:])
            ])
            self.head = nn.Linear(config.width, 1, dtype=torch.float64) if supervised else None
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.width, dtype=torch.float64) if config.normalization == "layer" else
            nn.BatchNorm1d(config.width, dtype=torch.float64) if config.normalization == "batch" else
            nn.Identity() for _ in self.linears
        ])
        generator = torch.Generator().manual_seed(config.seed + 1)
        with torch.no_grad():
            for layer in self.linears:
                # For rectangular first layers, Xavier uses both actual fans.
                std = math.sqrt(2 / layer.in_features) * scale
                if strategy == "xavier" and multiplier is None:
                    std = math.sqrt(2 / (layer.in_features + layer.out_features))
                layer.weight.copy_(torch.randn(layer.weight.shape, dtype=torch.float64, generator=generator) * std)
                layer.bias.zero_()
            if self.head is not None:
                self.head.weight.copy_(torch.randn(self.head.weight.shape, dtype=torch.float64,
                                                  generator=generator) * math.sqrt(2 / (config.width + 1)))
                self.head.bias.zero_()

    def trace(self, inputs):
        values = [inputs]
        for linear, norm in zip(self.linears, self.norms):
            z = norm(linear(values[-1]))
            values.append(torch.relu(z) if self.config.activation == "relu" else torch.tanh(z))
        return values

    def forward(self, inputs):
        h = self.trace(inputs)[-1]
        return self.head(h) if self.head is not None else h


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def rms(tensor):
    """Scaled RMS avoids a premature square overflow; non-finite input stays explicit."""
    tensor = tensor.detach()
    maximum = tensor.abs().max()
    if not torch.isfinite(maximum):
        return None
    if maximum == 0:
        return 0.
    return number(maximum * (tensor / maximum).square().mean().sqrt())


def status(value):
    if value is None:
        return "non-finite"
    if value == 0:
        return "zero (exact or numerical underflow)"
    if 0 < abs(value) < np.finfo(np.float64).tiny:
        return "subnormal"
    return "finite"


def diagnostic(config, strategy="he", *, multiplier=None, sample_limit=1024):
    net = SignalNet(config, strategy, multiplier=multiplier)
    generator = torch.Generator().manual_seed(config.seed)
    x = torch.randn(config.batch, config.width, dtype=torch.float64, generator=generator).requires_grad_(True)
    probe = torch.randn(x.shape, dtype=torch.float64, generator=generator)
    probe /= probe.square().mean().sqrt()
    activations = net.trace(x)
    loss = activations[-1].square().mean()
    weights = [layer.weight for layer in net.linears]
    loss_grads = torch.autograd.grad(loss, activations + weights, retain_graph=True)
    probe_grads = torch.autograd.grad(activations[-1], activations, grad_outputs=probe)
    rows = []
    samples = {}
    selected = {1, max(1, config.depth // 2), config.depth}
    for index, (h, gl, gp) in enumerate(zip(activations, loss_grads, probe_grads)):
        h = h.detach()
        hrms = rms(h)
        q = number(np.float64(hrms) ** 2) if hrms is not None else None
        row = {"layer": index, "q": q, "mean": number(h.mean()),
               "variance": number(h.var(unbiased=False)), "probe_rms": rms(gp), "loss_rms": rms(gl),
               "zero_fraction": number((h == 0).double().mean()),
               "nonfinite_count": int((~torch.isfinite(h)).sum()),
               "subnormal_count": int(((h.abs() > 0) & (h.abs() < torch.finfo(h.dtype).tiny)).sum()),
               "neurons": [rms(h[:, j]) for j in range(min(12, config.width))]}
        row["status"] = {key: status(row[key]) for key in ("q", "probe_rms", "loss_rms")}
        rows.append(row)
        if index in selected:
            samples[str(index)] = [number(v) for v in h.flatten()[::max(1, h.numel() // sample_limit)][:sample_limit]]
    return {"config": asdict(config), "strategy": strategy, "multiplier": multiplier,
            "layers": rows, "samples": samples, "loss": number(loss.detach()),
            "weight_loss_rms": [rms(g) for g in loss_grads[len(activations):]]}


def relu_experiment(seed=58, n=100000):
    rng = np.random.default_rng(seed)
    half = rng.standard_normal(n // 2)
    z = np.concatenate([half, -half])  # Exact symmetry isolates the second-moment identity.
    a = np.maximum(z, 0)
    return {"before": z[::max(1, len(z) // 5000)].tolist(), "after": a[::max(1, len(a) // 5000)].tolist(),
            "n": len(z), "q_before": float(np.mean(z*z)), "q_after": float(np.mean(a*a)),
            "var_before": float(z.var()), "var_after": float(a.var())}


def normalization_experiment(seed=58, repetitions=80):
    generator = torch.Generator().manual_seed(seed)
    means = torch.tensor([-3., -1., 2., 6.], dtype=torch.float64)
    scales = torch.tensor([0.5, 1., 2., 3.], dtype=torch.float64)

    def batch(n):
        return torch.randn(n, 4, dtype=torch.float64, generator=generator) * scales + means

    x = batch(8)
    bn = nn.BatchNorm1d(4, affine=False, momentum=0.1, dtype=torch.float64)
    ln = nn.LayerNorm(4, elementwise_affine=False, dtype=torch.float64)
    bn_training = bn(x).detach()
    sensitivity = []
    for size in (2, 8, 32, 128):
        estimates = torch.stack([batch(size).mean(0) for _ in range(repetitions)])
        sensitivity.append({"batch": size, "feature_mean_sd": estimates.std(0, unbiased=False).tolist(),
                            "mean_estimates": estimates[:, 0].tolist()})
    anchor = torch.tensor([[0., 1., 2., 3.]], dtype=torch.float64)
    companions_a, companions_b = batch(7), batch(7) + 8
    a, b = torch.cat([anchor, companions_a]), torch.cat([anchor, companions_b])
    anchor_bn_a, anchor_bn_b = bn(a)[0].detach(), bn(b)[0].detach()
    # Populate buffers on actual training-mode passes from the reference population.
    bn.reset_running_stats()
    for _ in range(100):
        bn(batch(32))
    running_mean, running_var = bn.running_mean.clone(), bn.running_var.clone()
    test = batch(8)
    train_test = bn(test).detach()
    bn.eval()
    eval_test = bn(test).detach()
    return {"before": x.tolist(), "batchnorm": bn_training.tolist(), "layernorm": ln(x).tolist(),
            "sensitivity": sensitivity, "population_means": means.tolist(), "population_scales": scales.tolist(),
            "repetitions": repetitions, "anchor": anchor[0].tolist(),
            "anchor_bn_a": anchor_bn_a.tolist(), "anchor_bn_b": anchor_bn_b.tolist(),
            "anchor_ln_a": ln(a)[0].tolist(), "anchor_ln_b": ln(b)[0].tolist(),
            "train_test": train_test.tolist(), "eval_test": eval_test.tolist(), "eval_input": test.tolist(),
            "running_mean": bn.running_mean.tolist(), "running_var": bn.running_var.tolist(),
            "running_before_test_mean": running_mean.tolist(), "running_before_test_var": running_var.tolist(),
            "training_batches_for_eval": int(bn.num_batches_tracked), "eps": bn.eps}


def stability_surface(config):
    depths = sorted(set([1, 2, 5, 10, config.depth] + list(range(5, config.depth + 1, 5))))
    depths = [d for d in depths if d <= config.depth]
    scales = sorted(set(np.geomspace(0.2, 2., 17).tolist() + [1 / math.sqrt(2), 1.]))
    surface_config = replace(config, width=min(config.width, 48), batch=16,
                             activation="relu", normalization="none")
    forward, backward, flags = [], [], []
    for scale in scales:
        frow, brow, srow = [], [], []
        for depth in depths:
            result = diagnostic(replace(surface_config, depth=depth), multiplier=scale)
            final, first = result["layers"][-1], result["layers"][0]
            frow.append(final["q"])
            brow.append(first["probe_rms"])
            srow.append({"forward": final["status"]["q"], "backward": first["status"]["probe_rms"]})
        forward.append(frow)
        backward.append(brow)
        flags.append(srow)
    return {"depths": depths, "scales": scales, "forward": forward, "backward": backward,
            "status": flags, "config": asdict(surface_config)}


def global_norm(tensors):
    norms = [rms(t) for t in tensors]
    if any(n is None for n in norms):
        return None
    return number(math.hypot(*(n * math.sqrt(t.numel()) for n, t in zip(norms, tensors))))


def supervised_data(seed=5800, n=384):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, dtype=torch.float64, generator=generator)
    noise = torch.randn(n, 1, dtype=torch.float64, generator=generator) * 0.08
    y = torch.sin(1.5*x[:, :1]) + 0.4*x[:, 1:] + noise
    order = torch.randperm(n, generator=generator)
    cut = int(0.75*n)
    return x[order[:cut]], y[order[:cut]], x[order[cut:]], y[order[cut:]]


def train_one(*, seed=58, strategy="he", lr=0.01, epochs=60, width=32, depth=6, clip=None):
    if strategy not in SCALES or not 0 < lr <= 1 or not 1 <= epochs <= 200:
        raise ValueError("Invalid training strategy, learning rate or epoch budget")
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError("clip must be finite and positive")
    config = Config(depth=depth, width=width, seed=seed)
    net = SignalNet(config, strategy, input_size=2, supervised=True)
    x, y, xv, yv = supervised_data()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    history = []
    nonfinite = None
    for epoch in range(epochs + 1):
        with torch.no_grad():
            train_loss = number((net(x) - y).square().mean())
            validation_loss = number((net(xv) - yv).square().mean())
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
               "gradient_norm": None, "clipped_gradient_norm": None, "update_norm": None}
        history.append(row)
        if train_loss is None or validation_loss is None:
            nonfinite = {"epoch": epoch, "stage": "forward loss"}
            break
        if epoch == epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = (net(x) - y).square().mean()
        loss.backward()
        parameters = list(net.parameters())
        row["gradient_norm"] = global_norm([p.grad for p in parameters])
        if row["gradient_norm"] is None:
            nonfinite = {"epoch": epoch, "stage": "backward gradient"}
            break
        if clip is not None:
            nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
        row["clipped_gradient_norm"] = global_norm([p.grad for p in parameters])
        before = [p.detach().clone() for p in parameters]
        optimizer.step()
        row["update_norm"] = global_norm([p.detach() - b for p, b in zip(parameters, before)])
        if row["update_norm"] is None:
            nonfinite = {"epoch": epoch, "stage": "parameter update"}
            break
    return {"seed": seed, "strategy": strategy, "lr": lr, "clip": clip, "epochs_requested": epochs,
            "config": asdict(config), "history": history, "nonfinite": nonfinite,
            "split_seed": 5800, "train_n": len(x), "validation_n": len(xv),
            "data_sha256": hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest(),
            "optimizer": "full-batch SGD; no momentum or decay; no test set"}


def jacobian_experiment(seed=58):
    records = []
    for depth in (1, 5, 12):
        for strategy in SCALES:
            config = Config(depth=depth, width=12, batch=2, seed=seed)
            net = SignalNet(config, strategy)
            x = torch.randn(12, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
            jacobian = torch.autograd.functional.jacobian(lambda v: net(v[None])[0], x)
            singular = torch.linalg.svdvals(jacobian)
            tolerance = max(jacobian.shape) * torch.finfo(jacobian.dtype).eps * float(singular.max())
            output = net(x[None]).detach()
            records.append({"depth": depth, "strategy": strategy, "singular_values": singular.tolist(),
                            "numerical_rank": int((singular > tolerance).sum()), "rank_tolerance": tolerance,
                            "q_out": number(output.square().mean()), "width": 12})
    # Same mean squared singular value, very different directional sensitivities.
    bad = np.array([math.sqrt(12)] + [0.] * 11)
    return {"records": records, "controlled_spectra": {"identity": np.ones(12).tolist(), "rank_one": bad.tolist()},
            "control_note": "Constructed linear maps; equal Frobenius-squared/width=1, not trained networks"}


def build_bundle(config, *, full=False, progress=print):
    seeds = [config.seed + i for i in range(5 if full else 3)]
    progress("Measuring paired propagation and selectable configurations...", flush=True)
    baseline = {name: diagnostic(config, name) for name in SCALES}
    baseline["he_layer"] = diagnostic(replace(config, normalization="layer"))
    grid = []
    for width in sorted({max(2, config.width // 2), config.width}):
        for depth in sorted({min(10, config.depth), config.depth}):
            for activation in ("relu", "tanh"):
                for normalization in ("none", "layer", "batch"):
                    for seed in seeds[:2]:
                        c = replace(config, width=width, depth=depth, activation=activation,
                                    normalization=normalization, seed=seed)
                        grid.append({"config": asdict(c), "results": {s: diagnostic(c, s, sample_limit=128) for s in SCALES}})
    repeats = {s: [diagnostic(replace(config, seed=seed), s) for seed in seeds] for s in ("xavier", "he")}
    progress("Measuring depth/scale surface and normalization...", flush=True)
    surface = stability_surface(config)
    progress("Training matched synthetic regression configurations...", flush=True)
    training = [train_one(seed=seed, strategy=s, lr=lr, epochs=100 if full else 60,
                          width=min(config.width, 32), depth=6)
                for seed in seeds for lr in (0.01, 0.15) for s in SCALES]
    clipping = [train_one(seed=seed, strategy="large", lr=0.15, epochs=100 if full else 60,
                         width=min(config.width, 32), depth=6, clip=1.) for seed in seeds]
    return {"schema": 1, "config": asdict(config), "seeds": seeds, "baseline": baseline, "grid": grid,
            "repeats": repeats, "surface": surface, "relu": relu_experiment(config.seed),
            "normalization": normalization_experiment(config.seed), "training": training,
            "clipping": clipping, "jacobian": jacobian_experiment(config.seed) if full else None,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "platform": platform.platform()},
            "interpretation_status": "author review required"}


def cached_bundle(config, output_dir, *, full=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() + json.dumps(asdict(config), sort_keys=True).encode()
                                 + str(full).encode() + (torch.__version__ + np.__version__).encode()).hexdigest()[:16]
    path = output_dir / f"measurements-{fingerprint}.json"
    if path.exists():
        print(f"Reusing measurements: {path}", flush=True)
        return json.loads(path.read_text(encoding="utf-8")), path, True
    bundle = build_bundle(config, full=full)
    path.write_text(json.dumps(bundle, allow_nan=False, separators=(",", ":")), encoding="utf-8")
    return bundle, path, False
