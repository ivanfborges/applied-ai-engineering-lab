"""CPU SFT/LoRA/DPO on synthetic response categories, not a language model."""

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 42
    rank: int = 1
    alpha: float = 2.0
    beta: float = 0.5
    base_steps: int = 200
    sft_steps: int = 80
    dpo_steps: int = 100
    base_lr: float = 0.03
    sft_lr: float = 0.03
    dpo_lr: float = 0.01

    def __post_init__(self):
        for name in ("seed", "rank", "base_steps", "sft_steps", "dpo_steps"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.rank > 4:
            raise ValueError("rank must not exceed min(input width 8, output width 4)")
        for name in ("alpha", "beta", "base_lr", "sft_lr", "dpo_lr"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def make_data(n, seed, concise_probability=0.5):
    """Balanced task IDs plus six nuisance features; all labels are synthetic."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2:
        raise ValueError("n must be a positive even integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(concise_probability) or not 0 <= concise_probability <= 1:
        raise ValueError("concise_probability must lie in [0, 1]")
    generator = torch.Generator().manual_seed(seed)
    task = (torch.arange(n) % 2)[torch.randperm(n, generator=generator)]
    x = torch.cat((F.one_hot(task, 2).float(),
                   0.2 * torch.randn(n, 6, generator=generator)), dim=1)
    chosen = task * 2
    rejected = chosen + 1
    concise = torch.rand(n, generator=generator) < concise_probability
    return x, torch.where(concise, chosen, rejected), chosen, rejected


class LoRALinear(nn.Module):
    """Floating-point LoRA with frozen base weight AND bias, without dropout."""

    def __init__(self, base, rank=1, alpha=2.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.device.type != "cpu":
            raise ValueError("base must be a CPU nn.Linear")
        if (isinstance(rank, bool) or not isinstance(rank, int)
                or not 1 <= rank <= min(base.in_features, base.out_features)):
            raise ValueError("rank must be an integer within the base dimensions")
        if isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.base = deepcopy(base).requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.normal_(self.A, std=0.02)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    def merged(self):
        """Return an independent inference layer; never mutate the frozen base."""
        layer = deepcopy(self.base)
        with torch.no_grad():
            layer.weight.add_(self.scale * (self.B @ self.A))
        return layer


def response_logps(logits, targets, response_mask):
    """Sum already shifted token log-probabilities for each nonempty response.

    Callers supply causal logits and aligned next-token IDs. Prompt/padding
    positions are excluded. Masked IDs may be -100 and are never gathered.
    """
    if logits.ndim != 3 or min(logits.shape) < 1 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite with shape (batch, tokens, vocab)")
    if targets.shape != logits.shape[:2] or response_mask.shape != targets.shape:
        raise ValueError("targets and mask must match the batch/token dimensions")
    if targets.dtype != torch.long or response_mask.dtype != torch.bool:
        raise ValueError("targets must be int64 and response_mask must be bool")
    if not response_mask.any(dim=1).all():
        raise ValueError("each example must have at least one response token")
    selected = targets[response_mask]
    if (selected < 0).any() or (selected >= logits.shape[-1]).any():
        raise ValueError("selected target IDs must lie inside the vocabulary")
    safe_targets = targets.masked_fill(~response_mask, 0)
    token_logps = logits.log_softmax(-1).gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_logps.masked_fill(~response_mask, 0).sum(-1)


def pair_logps(logits, chosen, rejected):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (nonempty batch, >=2 categories)")
    for ids in (chosen, rejected):
        if ids.dtype != torch.long or ids.shape != logits.shape[:1]:
            raise ValueError("response IDs must be int64 vectors matching the batch")
        if (ids < 0).any() or (ids >= logits.shape[1]).any():
            raise ValueError("response IDs must lie inside the category vocabulary")
    if (chosen == rejected).any() or not torch.isfinite(logits).all():
        raise ValueError("pairs must be distinct and logits finite")
    return logits.log_softmax(-1).gather(1, torch.stack((chosen, rejected), dim=1))


def dpo_loss(policy_logps, reference_logps, beta=0.5):
    if (policy_logps.ndim != 2 or policy_logps.shape[0] == 0
            or policy_logps.shape[1] != 2 or reference_logps.shape != policy_logps.shape):
        raise ValueError("policy and reference must have matching nonempty shape (pairs, 2)")
    if (not torch.isfinite(policy_logps).all() or not torch.isfinite(reference_logps).all()
            or (policy_logps > 0).any() or (reference_logps > 0).any()):
        raise ValueError("log-probabilities must be finite and nonpositive")
    if isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    difference = policy_logps - reference_logps.detach()
    scaled = beta * (difference[:, 0] - difference[:, 1])
    if not torch.isfinite(scaled).all():
        raise ValueError("scaled margins exceed the supported numeric range")
    return -F.logsigmoid(scaled).mean()


def train_sft(model, data, steps, learning_rate):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(data[0]), data[1])
        loss.backward()
        optimizer.step()


@torch.inference_mode()
def evaluate(model, data, reference, beta):
    model.eval()
    reference.eval()
    x, _, chosen, rejected = data
    logits = model(x)
    logps = logits.log_softmax(-1)
    ref_logps = reference(x).log_softmax(-1)
    pairs = pair_logps(logits, chosen, rejected)
    ref_pairs = pair_logps(reference(x), chosen, rejected)
    predicted = logits.argmax(-1)
    return {
        "task_accuracy": float((predicted // 2 == chosen // 2).float().mean()),
        "chosen_response_accuracy": float((predicted == chosen).float().mean()),
        "pair_accuracy": float((pairs[:, 0] > pairs[:, 1]).float().mean()),
        "mean_chosen_probability": float(logps.gather(1, chosen[:, None]).exp().mean()),
        "mean_pair_log_odds": float((pairs[:, 0] - pairs[:, 1]).mean()),
        "mean_reference_adjusted_margin": float(
            ((pairs[:, 0] - pairs[:, 1]) - (ref_pairs[:, 0] - ref_pairs[:, 1])).mean()),
        "kl_policy_to_sft_reference": float((logps.exp() * (logps - ref_logps)).sum(-1).mean()),
        "dpo_loss_vs_sft_reference": float(dpo_loss(pairs, ref_pairs, beta)),
    }


def run_experiment(config=Config()):
    """Fixed budgets; evaluation never chooses a checkpoint or hyperparameter."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            base_data = make_data(512, config.seed + 1, 0.5)
            sft_data = make_data(128, config.seed + 2, 0.8)
            preference_data = make_data(128, config.seed + 3)
            evaluation_data = make_data(256, config.seed + 4)
            base = nn.Linear(8, 4)
            train_sft(base, base_data, config.base_steps, config.base_lr)
            policy = LoRALinear(base, config.rank, config.alpha)
            base_snapshot = deepcopy(policy.base.state_dict())
            train_sft(policy, sft_data, config.sft_steps, config.sft_lr)
            # Freeze an independent copy of the entire SFT policy, including LoRA.
            reference = deepcopy(policy).requires_grad_(False).eval()
            reference_snapshot = deepcopy(reference.state_dict())
            stages = {"base": evaluate(base, evaluation_data, reference, config.beta),
                      "sft_lora": evaluate(policy, evaluation_data, reference, config.beta)}
            x, _, chosen, rejected = preference_data
            with torch.no_grad():
                reference_pairs = pair_logps(reference(x), chosen, rejected)
            optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
                                         lr=config.dpo_lr)
            policy.train()
            for _ in range(config.dpo_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = dpo_loss(pair_logps(policy(x), chosen, rejected), reference_pairs, config.beta)
                loss.backward()
                optimizer.step()
            stages["dpo_lora"] = evaluate(policy, evaluation_data, reference, config.beta)
            base_unchanged = all(torch.equal(v, base_snapshot[k])
                                 for k, v in policy.base.state_dict().items())
            reference_unchanged = all(torch.equal(v, reference_snapshot[k])
                                      for k, v in reference.state_dict().items())
            with torch.inference_mode():
                merge_error = float((policy(evaluation_data[0])
                                     - policy.merged()(evaluation_data[0])).abs().max())
            return {
                "data": "synthetic four-category response selection; no token generation",
                "config": asdict(config),
                "protocol": {"split_seeds": [config.seed + i for i in range(1, 5)],
                             "split_sizes": [512, 128, 128, 256],
                             "sft_concise_probability": 0.8,
                             "optimizer": "Adam; full batch; fresh state per stage",
                             "device": "cpu", "threads": 1,
                             "selection": "fixed final steps; evaluation unused in training"},
                "parameters": {"base_weight_and_bias": sum(p.numel() for p in base.parameters()),
                               "trainable_lora": sum(p.numel() for p in policy.parameters()
                                                     if p.requires_grad)},
                "stages": stages,
                "invariants": {"base_unchanged": base_unchanged,
                               "reference_unchanged": reference_unchanged,
                               "merged_max_abs_error": merge_error},
            }
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="New ignored JSON path; existing files are refused")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new dated path")
    config = Config(seed=args.seed, rank=args.rank, beta=args.beta)
    report = run_experiment(config)
    report["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "numpy": np.__version__, "platform": platform.platform()}
    report["hypothesis"] = "SFT and DPO can favor concise categories while retaining task identity"
    report["interpretation_candidate"] = "Compare task retention, probability, margins and KL together"
    report["review_status"] = "author review required"
    report["limitation"] = "Single seed; task is given as one-hot input; no human feedback or LLM"
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
