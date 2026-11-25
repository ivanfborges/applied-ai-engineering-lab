"""Synthetic CPU attention controls and a one-layer RoPE prefix-cache check."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from from_scratch import apply_rope, run_checks, sinusoidal_encoding

try:
    import torch
except ImportError as exc:
    raise ImportError('Install the canonical extra: python -m pip install -e ".[dev,deep-learning]"') from exc


def permutation_controls():
    """Keep weights fixed across token, coordinate and causal-mask interventions."""
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(69)
        x = torch.randn(1, 6, 16, dtype=torch.float64)
        layer = torch.nn.MultiheadAttention(16, 4, dropout=0.0, batch_first=True,
                                           dtype=torch.float64, device="cpu").eval()
    permutation = torch.tensor([2, 0, 5, 1, 4, 3])
    position = torch.arange(6, dtype=torch.float64)[:, None]
    frequency = torch.exp(-torch.arange(0, 16, 2, dtype=torch.float64) * np.log(10000.0) / 16)
    pe = torch.empty(6, 16, dtype=torch.float64)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)

    def forward(z, mask=None):
        return layer(z, z, z, attn_mask=mask, need_weights=False)[0]

    # MultiheadAttention bool True blocks a link. Keep this mask fixed to slots.
    causal = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    with torch.inference_mode():
        baseline = forward(x)
        positioned = forward(x + pe)
        metrics = {
            "no_position_equivariance_max_error": (forward(x[:, permutation]) - baseline[:, permutation]).abs().max().item(),
            "fixed_position_permutation_max_difference": (forward(x[:, permutation] + pe) - positioned[:, permutation]).abs().max().item(),
            "joint_token_position_permutation_max_error": (forward((x + pe)[:, permutation]) - positioned[:, permutation]).abs().max().item(),
            "fixed_causal_mask_permutation_max_difference": (forward(x[:, permutation], causal) - forward(x, causal)[:, permutation]).abs().max().item(),
            "numpy_torch_sinusoidal_max_error": float(np.max(np.abs(pe.numpy() - sinusoidal_encoding(np.arange(6), 16)))),
        }
    return metrics


def cache_offset_control():
    """Compare full causal attention and cached keys in one untrained head.

    Raw Q/K/V are held fixed. This is not a multi-layer decoding runtime.
    """
    rng = np.random.default_rng(169)
    q, k, v = rng.normal(size=(3, 8, 16))
    positions = np.arange(8)
    rotated_q, rotated_k = apply_rope(q, positions), apply_rope(k, positions)

    def attend(query, keys, values, query_positions):
        logits = query @ keys.T / np.sqrt(query.shape[-1])
        visible = positions[None, :] <= query_positions[:, None]
        logits = np.where(visible, logits, -np.inf)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        return weights @ values

    full = attend(rotated_q, rotated_k, v, positions)
    prefix = 5
    cached_keys = np.concatenate([rotated_k[:prefix], apply_rope(k[prefix:], positions[prefix:])])
    chunk = attend(apply_rope(q[prefix:], positions[prefix:]), cached_keys, v, positions[prefix:])
    reset_keys = np.concatenate([rotated_k[:prefix], apply_rope(k[prefix:], np.arange(3))])
    wrong = attend(apply_rope(q[prefix:], np.arange(3)), reset_keys, v, positions[prefix:])
    return {
        "correct_cache_offset_max_error": float(np.max(np.abs(chunk - full[prefix:]))),
        "reset_new_position_ids_max_difference": float(np.max(np.abs(wrong - full[prefix:]))),
    }


def run():
    return {
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "attention_configuration": {"seed": 69, "tokens": 6, "d_model": 16, "heads": 4,
                                    "dtype": "float64", "device": "cpu", "threads": 1,
                                    "dropout": 0, "training": False,
                                    "permutation": [2, 0, 5, 1, 4, 3], "embedding_scale": 1},
        "permutation_controls": permutation_controls(),
        "cache_configuration": {"seed": 169, "tokens": 8, "head_dim": 16, "prefix": 5,
                                "base": 10000.0, "dtype": "float64", "values_rotated": False},
        "cache_offset_control": cache_offset_control(),
        "numerical_checks": run_checks(),
        "interpretation_status": "author review required; synthetic, untrained architectural controls",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON record; use an ignored outputs/ path.")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output already exists; preserve the record and choose a new filename.")
    result = run()
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
