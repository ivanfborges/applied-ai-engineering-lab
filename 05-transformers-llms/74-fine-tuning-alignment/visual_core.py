"""Deterministic synthetic post-training experiments, independent of rendering."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from example import LoRALinear, dpo_loss, evaluate, make_data, pair_logps, response_logps
from from_scratch import dpo_loss_gradient


DEMOS = {
    "landscape": "01 Pretraining and adaptation landscapes (analytical optimization)",
    "sft": "02 SFT response probabilities and token masking (trained + arithmetic)",
    "instruction": "03 Symbolic multitask adaptation and held-out variants (trained)",
    "reward": "04 Bradley-Terry reward modeling and annotation noise (trained)",
    "rlhf": "05 Exact finite-policy reward/KL optimization (not PPO)",
    "dpo": "06 DPO trajectories, beta and relative/absolute probabilities (trained)",
    "comparison": "07 Reward optimization versus population DPO (calculated + optimized)",
    "lora": "08 Low-rank matrix learning and truncated SVD (trained + analytical)",
    "geometry": "09 Projected rank constraints and full updates (optimized)",
    "efficiency": "10 Implemented bottleneck adapter and theoretical parameter/storage counts",
    "qlora": "11 Uniform quantization and frozen reconstructed base + LoRA (simulation)",
    "forgetting": "12 Sequential synthetic tasks, full/LoRA/replay (trained)",
    "hacking": "13 Misspecified proxy versus intended synthetic utility (optimized)",
    "pipeline": "14 Base, SFT-LoRA and DPO-LoRA with held-out metrics (trained)",
}


@dataclass(frozen=True)
class LabConfig:
    seed: int = 42
    quick: bool = False

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.quick, bool):
            raise ValueError("quick must be boolean")

    @property
    def steps(self):
        return 80 if self.quick else 240

    @property
    def frames(self):
        return 10 if self.quick else 25

    @property
    def samples(self):
        return 128 if self.quick else 512

    @property
    def dimension(self):
        return 16 if self.quick else 24

    @property
    def repetitions(self):
        return 1 if self.quick else 3


@contextmanager
def isolated_cpu(seed):
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.set_num_threads(threads)


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def softmax(logits):
    x = np.asarray(logits, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("logits must be a finite vector with at least two entries")
    e = np.exp(x - x.max())
    return e / e.sum()


def kl_divergence(policy, reference):
    p, q = np.asarray(policy, float), np.asarray(reference, float)
    if (p.shape != q.shape or p.ndim != 1 or not np.isfinite(p).all()
            or not np.isfinite(q).all() or (p < 0).any() or (q <= 0).any()
            or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1)):
        raise ValueError("probabilities must sum to one; reference must have full support")
    active = p > 0
    return float(np.sum(p[active] * (np.log(p[active]) - np.log(q[active]))))


def optimal_policy(reward, reference, beta):
    r, q = np.asarray(reward, float), np.asarray(reference, float)
    kl_divergence(q, q)
    if r.shape != q.shape or not np.isfinite(r).all() or not math.isfinite(beta) or beta < 0:
        raise ValueError("reward must match reference; beta must be finite and nonnegative")
    if beta == 0:
        p = np.zeros_like(q)
        winners = r == r.max()
        p[winners] = q[winners] / q[winners].sum()
        return p
    return softmax(np.log(q) + (r - r.max()) / beta)


def policy_trajectory(reward, reference, beta, steps):
    """Exponentiated-gradient/mirror ascent with exact expectations; no sampling/PPO."""
    optimum = optimal_policy(reward, reference, beta)
    q, r = np.asarray(reference, float), np.asarray(reward, float)
    logits = np.log(q)
    records = []
    for step in range(steps + 1):
        p = softmax(logits)
        kl = kl_divergence(p, q)
        records.append({"step": step, "probabilities": p.copy(), "reward": float(p @ r),
                        "kl": kl, "objective": float(p @ r - beta * kl)})
        # Natural logit direction, equivalent to an exponentiated policy update.
        logits += 0.2 * (r - beta * (np.log(p) - np.log(q)))
        logits -= logits.mean()
    return {"reference": q, "reward": r, "beta": beta, "records": records,
            "optimum": optimum, "max_abs_optimum_error": float(abs(p - optimum).max())}


def landscape_demo(c):
    rng = np.random.default_rng(c.seed)
    start = rng.uniform(-3.5, -2.0, 2)
    centers = [np.array([0., 0.]), np.array([1.7, -0.9])]
    matrices = [np.array([[1.2, .3], [.3, .8]]), np.array([[.7, -.2], [-.2, 1.5]])]
    position = start.copy()
    traces = []
    for name, center, hessian in zip(("pretraining", "fine-tuning"), centers, matrices):
        for step in range(c.steps + 1):
            delta = position - center
            traces.append({"stage": name, "step": step, "position": position.copy(),
                           "loss": float(.5 * delta @ hessian @ delta)})
            if step < c.steps:
                position -= .08 * hessian @ delta
    return {"centers": centers, "hessians": matrices, "records": traces,
            "limitation": "Analytical two-parameter quadratics, not a neural loss landscape"}


def sft_demo(c):
    model = nn.Linear(2, 4)
    x = torch.tensor([[1., .4]])
    target = torch.tensor([2])
    optimizer = torch.optim.SGD(model.parameters(), lr=.12)
    records = []
    for step in range(c.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        records.append({"step": step, "logits": array(logits[0]),
                        "probabilities": array(logits.softmax(-1)[0]), "loss": float(loss.detach()),
                        "gradient": float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())))})
        if step < c.steps:
            optimizer.step()
    logits = torch.randn(1, 6, 8)
    targets = torch.tensor([[1, 2, 3, 4, 5, 0]])
    mask = torch.tensor([[False, False, False, True, True, False]])
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return {"target": 2, "records": records, "labels": ["wrong task", "verbose", "chosen", "irrelevant"],
            "token": {"inputs": ["<bos>", "answer", "2+2", "<assistant>", "4", "<eos>"],
                      "targets": ["answer", "2+2", "<assistant>", "4", "<eos>", "<pad>"],
                      "mask": array(mask[0]), "cross_entropy": array(losses[0]),
                      "selected_sum": float(-response_logps(logits, targets, mask)[0])},
            "limitation": "Single synthetic instruction/category; token logits are a separate arithmetic fixture"}


class MultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(7, 16)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.hidden(x)))


def instruction_data(n, seed, heldout=False):
    g = torch.Generator().manual_seed(seed)
    task = torch.arange(n) % 4
    xy = torch.randn(n, 2, generator=g)
    rules = torch.stack((xy[:, 0], xy[:, 1], xy[:, 0] + xy[:, 1], xy[:, 0] - xy[:, 1]), -1)
    y = (rules.gather(1, task[:, None]).squeeze(1) > 0).long()
    variants = torch.randint(0, 2, (n,), generator=g).float() * .5 + (1. if heldout else 0.)
    return torch.cat((F.one_hot(task, 4).float(), xy, variants[:, None]), -1), y, task


def instruction_demo(c):
    train = instruction_data(c.samples * 4, c.seed + 101)
    test = instruction_data(c.samples * 4, c.seed + 102, True)
    base = MultiTask()
    optimizer = torch.optim.Adam(base.parameters(), lr=.02)
    # Base training sees only one common rule; it does not see task-specific labels.
    common_target = (train[0][:, 4] > 0).long()
    for _ in range(c.steps):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(base(train[0]), common_target).backward()
        optimizer.step()
    output = {}
    probes = test[0][torch.arange(32)]
    for name in ("base", "single-task", "multitask"):
        model = deepcopy(base)
        opt = torch.optim.Adam(model.parameters(), lr=.015)
        records = []
        for step in range(c.steps + 1):
            with torch.no_grad():
                predicted = model(test[0]).argmax(-1)
                accuracy = [float((predicted[test[2] == i] == test[1][test[2] == i]).float().mean()) for i in range(4)]
                records.append({"step": step, "accuracy": accuracy,
                                "hidden": array(torch.tanh(model.hidden(probes)))})
            if step < c.steps and name != "base":
                selected = train[2] == 1 if name == "single-task" else torch.ones_like(train[2], dtype=torch.bool)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(train[0][selected]), train[1][selected]).backward()
                opt.step()
        output[name] = records
    return {"models": output, "probe_tasks": array(test[2][:32]),
            "task_labels": ["classification", "extraction", "summarization", "QA"],
            "split_seeds": [c.seed + 101, c.seed + 102], "samples_per_split": len(train[0]),
            "limitation": "Names denote four numerical threshold rules, not language tasks; held-out scalar variant IDs are not unseen instructions"}


def reward_demo(c):
    rng = np.random.default_rng(c.seed + 200)
    utility_weights = np.array([2., .4, 1.5, 1.2, -.8])
    first, second = rng.uniform(0, 1, (2, c.samples * 2, 5))
    difference = first - second
    probability = 1 / (1 + np.exp(-(difference @ utility_weights)))
    labels = rng.binomial(1, probability).astype(float)
    test_diff = rng.uniform(0, 1, (c.samples, 5)) - rng.uniform(0, 1, (c.samples, 5))
    test_prob = 1 / (1 + np.exp(-(test_diff @ utility_weights)))
    output = {}
    for noise in (0., .2, .4):
        noisy = np.where(rng.uniform(size=len(labels)) < noise, 1 - labels, labels)
        model = nn.Linear(5, 1, bias=False)
        nn.init.zeros_(model.weight)
        opt = torch.optim.Adam(model.parameters(), lr=.04)
        x, y = torch.tensor(difference, dtype=torch.float32), torch.tensor(noisy, dtype=torch.float32)
        records = []
        for step in range(c.steps + 1):
            logits = model(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            w = array(model.weight[0])
            pred = 1 / (1 + np.exp(-(test_diff @ w)))
            expected_test_loss = -(test_prob * np.log(pred) + (1 - test_prob) * np.log(1 - pred)).mean()
            scores = np.stack((first[0] @ w, second[0] @ w))
            records.append({"step": step, "weights": w, "loss": float(loss.detach()),
                            "expected_test_loss": float(expected_test_loss), "probe_scores": scores,
                            "probe_preference_probability": float(1 / (1 + np.exp(-(scores[0] - scores[1]))))})
            if step < c.steps:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        output[str(noise)] = records
    return {"noise": output, "utility_weights": utility_weights,
            "features": ["factuality", "conciseness", "relevance", "compliance", "verbosity"],
            "probe_features": [first[0], second[0]], "samples": len(labels),
            "limitation": "Synthetic feature scores and stochastic BT labels; flips add inconsistent annotations, not real human judgments"}


def rlhf_demo(c):
    q = np.array([.6, .3, .1])
    reward = np.array([0., .6, 1.])
    return {"runs": {str(beta): policy_trajectory(reward, q, beta, c.steps * 4) for beta in (0., .2, .5, 2.)},
            "limitation": "Exact finite-action mirror ascent; supplied rewards, no learned reward model, sampling or PPO"}


def dpo_demo(c):
    q = np.array([.45, .35, .2])
    output = {}
    for beta in (.1, .5, 2.):
        logits = np.log(q)
        records = []
        for step in range(c.steps + 1):
            p = softmax(logits)
            loss, grad = dpo_loss_gradient(np.log(p[:2])[None, :], np.log(q[:2])[None, :], beta)
            margin = float(np.log(p[0] / p[1]) - np.log(q[0] / q[1]))
            records.append({"step": step, "logits": logits.copy(), "probabilities": p,
                            "margin": margin, "loss": loss, "gradient": float(np.linalg.norm(grad)),
                            "bt_preference_probability": float(1 / (1 + np.exp(-beta * margin)))})
            logits[:2] -= .2 * grad[0]
        output[str(beta)] = records
    return {"runs": output, "reference": q,
            "counterexample": {"reference": [.4, .3, .3], "policy": [.3, .1, .6]},
            "limitation": "One fixed offline chosen/rejected pair in three categories; BT probability is not response probability"}


def comparison_demo(c):
    q = np.array([.6, .3, .1])
    reward, beta = np.array([0., .6, 1.]), .5
    rlhf = policy_trajectory(reward, q, beta, c.steps * 4)
    logits = torch.tensor(np.log(q), dtype=torch.float64, requires_grad=True)
    first, second = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    preference = torch.sigmoid(torch.tensor(reward[first] - reward[second]))
    opt = torch.optim.Adam([logits], lr=.04)
    records = []
    for step in range(c.steps * 4 + 1):
        logps = logits.log_softmax(-1)
        relative = beta * ((logps[first] - logps[second])
                           - torch.tensor(np.log(q[first] / q[second])))
        loss = F.binary_cross_entropy_with_logits(relative, preference)
        p = array(logps.exp())
        kl = kl_divergence(p, q)
        records.append({"step": step, "probabilities": p, "loss": float(loss.detach()),
                        "reward": float(p @ reward), "kl": kl,
                        "objective": float(p @ reward - beta * kl),
                        "margin": float(np.log(p[2] / p[0]) - np.log(q[2] / q[0]))})
        if step < c.steps * 4:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return {"rlhf": rlhf, "dpo": records, "preference_targets": array(preference),
            "beta": beta, "reference": q, "reward": reward,
            "limitation": "Population BT probabilities for all three pairs, not a finite noisy preference dataset; different update geometries/budgets are not a speed or quality ranking"}


def truncated_svd(matrix, rank):
    value = np.asarray(matrix, float)
    if value.ndim != 2 or not np.isfinite(value).all() or not 1 <= rank <= min(value.shape):
        raise ValueError("finite matrix and rank inside its dimensions required")
    u, s, v = np.linalg.svd(value, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ v[:rank]


def lora_demo(c):
    rng = np.random.default_rng(c.seed + 300)
    d = c.dimension
    frozen = rng.normal(0, .2, (d, d))
    low = rng.normal(size=(d, 2)) @ rng.normal(size=(2, d)) / d
    full = rng.normal(size=(d, d)) / np.sqrt(d)
    output = {}
    for name, target in (("rank-2 target", low), ("full-rank target", full)):
        runs = {}
        for rank in (1, 2, 4, 8, 16):
            base = nn.Linear(d, d, bias=False).double()
            with torch.no_grad():
                base.weight.copy_(torch.tensor(frozen))
            layer = LoRALinear(base, rank, alpha=float(rank))
            optimizer = torch.optim.Adam([layer.A, layer.B], lr=.04)
            records = []
            target_tensor = torch.tensor(target)
            for step in range(c.steps + 1):
                update = layer.B @ layer.A
                loss = (update - target_tensor).square().sum()
                records.append({"step": step, "error": float(loss.detach().sqrt()),
                                "A": array(layer.A), "B": array(layer.B), "update": array(update)})
                if step < c.steps:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            runs[str(rank)] = {"records": records,
                               "svd_error": float(np.linalg.norm(target - truncated_svd(target, rank))),
                               "frozen_unchanged": bool(np.array_equal(array(layer.base.weight), frozen)),
                               "numerical_rank": int(np.linalg.matrix_rank(array(layer.B @ layer.A), tol=1e-8))}
        output[name] = {"target_update": target, "singular_values": np.linalg.svd(target, compute_uv=False), "ranks": runs}
    return {"base": frozen, "targets": output, "dimension": d, "alpha_over_rank": 1.,
            "limitation": "Frobenius matrix reconstruction, not LLM quality; finite optimization can remain above the SVD lower bound"}


def geometry_demo(c):
    target = np.array([[1., .6], [.4, -1.]])
    output = {}
    for rank in (1, 2):
        current = np.zeros((2, 2))
        records = []
        for step in range(c.steps + 1):
            records.append({"step": step, "matrix": current.copy(), "error": float(np.linalg.norm(current - target))})
            current = truncated_svd(current - .12 * (current - target), rank)
        output[str(rank)] = records
    return {"target": target, "runs": output, "optimal_rank1": truncated_svd(target, 1),
            "limitation": "Projected gradient on 2x2 matrices; three displayed coordinates discard the fourth; bounded-rank matrices form a nonlinear family, not a fixed linear subspace"}


class BottleneckAdapter(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        if not isinstance(width, int) or not isinstance(bottleneck, int) or min(width, bottleneck) < 1:
            raise ValueError("positive integer dimensions required")
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


def parameter_counts(width, rank, bottleneck, layers=1):
    for value in (width, rank, bottleneck, layers):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("dimensions/layers must be positive integers")
    return {"full": layers * width * width, "lora": layers * rank * 2 * width,
            "adapter": layers * (2 * width * bottleneck + bottleneck + width)}


def efficiency_demo(c):
    dims = np.array([32, 64, 128, 256, 512, 1024])
    ranks = np.array([1, 2, 4, 8, 16, 32])
    adapter = BottleneckAdapter(8, 2)
    x = torch.randn(1, 8)
    down, activated = adapter.down(x), F.relu(adapter.down(x))
    return {"dimensions": dims, "ranks": ranks,
            "counts": [parameter_counts(int(d), 8, 8, 4) for d in dims],
            "adapter_fixture": {"input": array(x[0]), "down": array(down[0]),
                                "relu": array(activated[0]), "up": array(adapter.up(activated)[0]),
                                "output": array(adapter(x)[0]),
                                "count": sum(p.numel() for p in adapter.parameters())},
            "storage_assumptions": {"trainable_precision_bits": 16, "layers": 4, "bias_policy": "full weight only; LoRA no bias; adapter down/up biases included"},
            "limitation": "Exact architecture counts and theoretical FP16 checkpoints, not total training memory, latency or GPU requirements"}


def quantize_uniform(weights, bits):
    w = np.asarray(weights, float)
    if not np.isfinite(w).all() or w.size == 0 or bits not in (4, 8):
        raise ValueError("nonempty finite weights and bits=4 or 8 required")
    bound = float(abs(w).max())
    scale = bound / (2 ** (bits - 1) - 1) if bound else 1.
    codes = np.clip(np.rint(w / scale), -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1).astype(np.int16)
    return {"codes": codes, "scale": scale, "reconstructed": codes * scale,
            "theoretical_bytes": math.ceil(w.size * bits / 8) + 4,
            "error": float(np.sqrt(np.mean((w - codes * scale) ** 2)))}


def qlora_demo(c):
    rng = np.random.default_rng(c.seed + 400)
    weights = rng.normal(0, .2, (c.dimension, c.dimension)).astype(np.float32)
    q4, q8 = quantize_uniform(weights, 4), quantize_uniform(weights, 8)
    reconstructed = q4["reconstructed"]
    base = nn.Linear(c.dimension, c.dimension, bias=False).double()
    with torch.no_grad():
        base.weight.copy_(torch.tensor(reconstructed))
    layer = LoRALinear(base, 2, 2.)
    target = torch.tensor(weights.astype(float))
    opt = torch.optim.Adam([layer.A, layer.B], lr=.02)
    records = []
    for step in range(c.steps + 1):
        effective = layer.base.weight + layer.B @ layer.A
        loss = (effective - target).square().mean()
        records.append({"step": step, "error": float(loss.detach().sqrt()), "effective": array(effective)})
        if step < c.steps:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return {"original": weights, "fp16": weights.astype(np.float16).astype(float),
            "q4": q4, "q8": q8, "records": records,
            "base_unchanged": bool(np.array_equal(array(layer.base.weight), reconstructed)),
            "limitation": "Symmetric uniform quantization with one FP32 scale; codes simulated in int16 and reconstructed FP64 training; NOT NF4, double quantization, paged optimizers or actual QLoRA memory"}


def forgetting_data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    return x, (x[:, 0] > 0).long(), (x[:, 1] > 0).long()


def forgetting_demo(c):
    x, a, b = forgetting_data(c.samples * 2, c.seed + 501)
    test, ta, tb = forgetting_data(c.samples * 2, c.seed + 502)
    base = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 2))
    opt = torch.optim.Adam(base.parameters(), lr=.03)
    for _ in range(c.steps):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(base(x), a).backward()
        opt.step()
    grid_axis = np.linspace(-2.5, 2.5, 31)
    gx, gy = np.meshgrid(grid_axis, grid_axis)
    grid = torch.tensor(np.column_stack((gx.ravel(), gy.ravel())), dtype=torch.float32)
    runs = {}
    for method in ("full", "lora", "replay"):
        model = deepcopy(base)
        if method == "lora":
            model.requires_grad_(False)
            model[2] = LoRALinear(model[2], 1, 2.)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.02)
        records = []
        for step in range(c.steps + 1):
            with torch.no_grad():
                pred = model(test).argmax(-1)
                records.append({"step": step, "task_a": float((pred == ta).float().mean()),
                                "task_b": float((pred == tb).float().mean()),
                                "boundary": array(model(grid).softmax(-1)[:, 1]).reshape(gx.shape)})
            if step < c.steps:
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x), b)
                if method == "replay":
                    loss = .5 * loss + .5 * F.cross_entropy(model(x), a)
                loss.backward()
                optimizer.step()
        runs[method] = records
    return {"runs": runs, "grid_axis": grid_axis, "split_seeds": [c.seed + 501, c.seed + 502],
            "limitation": "Conflicting rules A=x0>0 and B=x1>0 without task ID; replay deliberately trades off objectives; one seed is not a universal LoRA-retention claim"}


def hacking_demo(c):
    length = np.array([20., 60., 120.])
    truth = np.array([1., .8, .1])
    intended = truth - .002 * length
    proxy = length / length.max()
    reference = np.array([.6, .3, .1])
    runs = {"proxy beta=0": policy_trajectory(proxy, reference, 0., c.steps * 4),
            "proxy beta=0.5": policy_trajectory(proxy, reference, .5, c.steps * 4),
            "corrected beta=0.5": policy_trajectory(intended, reference, .5, c.steps * 4)}
    for report in runs.values():
        for row in report["records"]:
            p = row["probabilities"]
            row.update(proxy=float(p @ proxy), intended=float(p @ intended), length=float(p @ length))
    return {"runs": runs, "candidate_lengths": length, "intended_utility": intended, "proxy_reward": proxy,
            "limitation": "Constructed misspecification with supplied utility; KL constrains drift but cannot identify or repair wrong rewards"}


def pipeline_demo(c):
    datasets = [make_data(n, c.seed + 601 + i, probability) for i, (n, probability)
                in enumerate(((c.samples * 2, .5), (c.samples, .8), (c.samples, .5), (c.samples * 2, .5)))]
    base = nn.Linear(8, 4)
    snapshots = []
    ref = None
    frozen_snapshot = None
    for stage, data in zip(("base", "sft", "dpo"), datasets[:3]):
        if stage == "base":
            model = base
        elif stage == "sft":
            model = LoRALinear(base, 1, 2.)
            frozen_snapshot = deepcopy(model.base.state_dict())
        else:
            ref = deepcopy(model).requires_grad_(False).eval()
            ref_snapshot = deepcopy(ref.state_dict())
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.03 if stage != "dpo" else .01)
        stage_start = np.concatenate([array(p).ravel() for p in model.parameters() if p.requires_grad])
        for step in range(c.steps + 1):
            logits = model(data[0])
            if stage == "dpo":
                with torch.no_grad():
                    reference_pairs = pair_logps(ref(data[0]), data[2], data[3])
                loss = dpo_loss(pair_logps(logits, data[2], data[3]), reference_pairs, .5)
            else:
                loss = F.cross_entropy(logits, data[1])
            with torch.no_grad():
                trainables = [p for p in model.parameters() if p.requires_grad]
                snapshot = {"stage": stage, "step": step, "loss": float(loss.detach()),
                            "metrics": evaluate(model, datasets[3], ref or model, .5),
                            "probabilities": array(model(datasets[3][0][:2]).softmax(-1)),
                            "trainable_parameters": sum(p.numel() for p in trainables),
                            "weights": np.concatenate([array(p).ravel() for p in trainables]),
                            "weight_updates": np.concatenate([array(p).ravel() for p in trainables]) - stage_start,
                            "_evaluation_logps": array(model(datasets[3][0]).log_softmax(-1)),
                            "base_frozen_unchanged": stage == "base" or all(torch.equal(v, frozen_snapshot[k]) for k, v in model.base.state_dict().items())}
                snapshots.append(snapshot)
            if step < c.steps:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    # Retrospectively score all recorded distributions against the SAME SFT reference.
    with torch.no_grad():
        ref_logps = array(ref(datasets[3][0]).log_softmax(-1))
    chosen, rejected = array(datasets[3][2]), array(datasets[3][3])
    rows = np.arange(len(chosen))
    ref_pairs = ref_logps[rows[:, None], np.column_stack((chosen, rejected))]
    for row in snapshots:
        logps = row.pop("_evaluation_logps")
        pairs = logps[rows[:, None], np.column_stack((chosen, rejected))]
        row["metrics"]["kl_policy_to_sft_reference"] = float((np.exp(logps) * (logps - ref_logps)).sum(-1).mean())
        row["metrics"]["mean_reference_adjusted_margin"] = float((pairs[:, 0] - pairs[:, 1] - ref_pairs[:, 0] + ref_pairs[:, 1]).mean())
        row["metrics"]["dpo_loss_vs_sft_reference"] = dpo_loss_gradient(pairs, ref_pairs, .5)[0]
    return {"records": snapshots,
            "final_reference_unchanged": all(torch.equal(v, ref_snapshot[k]) for k, v in ref.state_dict().items()),
            "split_seeds": [c.seed + 601 + i for i in range(4)],
            "evaluation_size": len(datasets[3][0]),
            "limitation": "Task retention means only original category identity; all KL/margins use the final SFT reference; stage losses have different objectives and must not be ranked by magnitude"}


FUNCTIONS = {"landscape": landscape_demo, "sft": sft_demo, "instruction": instruction_demo,
             "reward": reward_demo, "rlhf": rlhf_demo, "dpo": dpo_demo,
             "comparison": comparison_demo, "lora": lora_demo, "geometry": geometry_demo,
             "efficiency": efficiency_demo, "qlora": qlora_demo, "forgetting": forgetting_demo,
             "hacking": hacking_demo, "pipeline": pipeline_demo}


def compute_demo(name, config):
    if name not in FUNCTIONS:
        raise ValueError(f"unknown demo: {name}")
    index = list(DEMOS).index(name)
    with isolated_cpu(config.seed + 1000 * index):
        result = FUNCTIONS[name](config)
    if name in ("instruction", "forgetting"):
        def final_scores(report):
            if name == "instruction":
                return {key: rows[-1]["accuracy"] for key, rows in report["models"].items()}
            return {key: [rows[-1]["task_a"], rows[-1]["task_b"]] for key, rows in report["runs"].items()}
        repetitions = [{"seed": config.seed, "final_scores": final_scores(result)}]
        for repetition in range(1, config.repetitions):
            repeated_config = replace(config, seed=config.seed + repetition * 10000)
            with isolated_cpu(repeated_config.seed + 1000 * index):
                repeated = FUNCTIONS[name](repeated_config)
            repetitions.append({"seed": repeated_config.seed, "final_scores": final_scores(repeated)})
        result["repetitions"] = repetitions
    result.update(configuration=asdict(config), demo=name, synthetic=True,
                  interpretation_status="author review required")
    return result


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value
