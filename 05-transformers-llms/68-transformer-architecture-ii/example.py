"""CPU numerical parity and controlled synthetic MHA calculations, without plots."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit('PyTorch is required; install the repository extra: pip install -e ".[deep-learning]"') from exc

from from_scratch import MultiHeadAttention


def torch_reference(layer, x, *, causal=False, blocked_mask=None, key_padding_mask=None):
    """Copy NumPy right-multiplication weights into PyTorch's transposed storage."""
    module = torch.nn.MultiheadAttention(
        layer.d_model, layer.num_heads, dropout=0.0, bias=False, batch_first=True,
        dtype=torch.float64, device="cpu",
    ).eval()
    batch, tokens, _ = x.shape
    # nn.MultiheadAttention accepts (B*H,T,T), not the NumPy layer's (B,H,T,T).
    blocked = np.zeros((batch, layer.num_heads, tokens, tokens), dtype=bool)
    if blocked_mask is not None:
        blocked |= blocked_mask
    if causal:
        blocked |= np.triu(np.ones((tokens, tokens), dtype=bool), k=1)
    with torch.inference_mode():
        module.in_proj_weight.copy_(torch.from_numpy(np.concatenate([layer.wq, layer.wk, layer.wv], axis=1).T.copy()))
        module.out_proj.weight.copy_(torch.from_numpy(layer.wo.T.copy()))
        tensor = torch.tensor(x, dtype=torch.float64, device="cpu")
        output, attention = module(
            tensor, tensor, tensor,
            attn_mask=torch.from_numpy(blocked.reshape(batch * layer.num_heads, tokens, tokens)),
            key_padding_mask=None if key_padding_mask is None else torch.from_numpy(key_padding_mask),
            need_weights=True, average_attn_weights=False,
        )
    return output.numpy(), attention.numpy()


def entropy(attention):
    """Mean row entropy in nats, using 0 log 0 = 0 for masked positions."""
    positive = attention > 0
    terms = np.zeros_like(attention)
    terms[positive] = attention[positive] * np.log(attention[positive])
    return float(-terms.sum(-1).mean())


def run(seed=68):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, 6, 32))
    layer = MultiHeadAttention(32, 4, seed=seed)
    parity = []
    padding = np.zeros((2, 6), dtype=bool)
    padding[0, -1] = True
    for causal in (False, True):
        output, attention = layer.forward(x, causal=causal, key_padding_mask=padding)
        reference_output, reference_attention = torch_reference(layer, x, causal=causal, key_padding_mask=padding)
        np.testing.assert_allclose(output, reference_output, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(attention, reference_attention, atol=1e-12, rtol=1e-12)
        parity.append({
            "causal": causal,
            "output_max_abs_error": float(np.max(abs(output - reference_output))),
            "attention_max_abs_error": float(np.max(abs(attention - reference_attention))),
            "row_sum_max_abs_error": float(np.max(abs(attention.sum(-1) - 1))),
            "blocked_key_max_weight": float(attention[0, :, :, -1].max()),
        })

    causal_output, causal_attention = layer.forward(x, causal=True)
    changed = x.copy()
    changed[:, 3:] += rng.normal(size=changed[:, 3:].shape) * 3
    changed_output, _ = layer.forward(changed, causal=True)
    full_output, _ = layer.forward(x)
    full_changed, _ = layer.forward(changed)
    prefix_error = float(np.max(abs(causal_output[:, :3] - changed_output[:, :3])))
    np.testing.assert_allclose(causal_output[:, :3], changed_output[:, :3], atol=1e-12, rtol=1e-12)

    # Every head-count configuration gets exactly the same packed weights and input.
    sweep_x = rng.normal(size=(2, 6, 64))
    sweep = []
    for heads in (1, 2, 4, 8, 16, 32):
        candidate = MultiHeadAttention(64, heads, seed=seed)
        _, scaled = candidate.forward(sweep_x)
        _, unscaled = candidate.forward(sweep_x, scaled=False)
        sweep.append({
            "heads": heads, "d_head": candidate.d_head,
            "projection_parameters": sum(w.size for w in (candidate.wq, candidate.wk, candidate.wv, candidate.wo)),
            "attention_entries": int(scaled.size),
            "scaled_mean_entropy_nats": entropy(scaled),
            "unscaled_mean_entropy_nats": entropy(unscaled),
        })
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(), "device": "cpu", "dtype": "float64"},
        "configuration": {"seed": seed, "batch": 2, "tokens": 6, "d_model": 32, "heads": 4, "dropout": 0, "bias": False, "updates": 0, "data": "synthetic independent standard-normal vectors", "sweep_d_model": 64, "sweep_mask": "bidirectional, no padding"},
        "parity": {
            "hypothesis": "Identical projections yield matching outputs and individual maps in NumPy and PyTorch.",
            "result": parity, "tolerance": {"atol": 1e-12, "rtol": 1e-12},
            "interpretation_candidate": "The tested forward calculations agree, including causal and padding masks.",
            "limitation": "Small CPU float64 cases; no training, mixed precision or optimized-kernel benchmark.",
        },
        "causal_control": {
            "hypothesis": "Changing tokens at positions 3+ cannot change earlier outputs under a causal mask.",
            "result": {"prefix_max_abs_change": prefix_error, "bidirectional_prefix_max_abs_change": float(np.max(abs(full_output[:, :3] - full_changed[:, :3]))), "future_max_weight": float(np.triu(causal_attention, k=1).max())},
            "interpretation_candidate": "The observed mask blocks future information in this attention layer.",
            "limitation": "Layer-level control; not a complete autoregressive training pipeline.",
        },
        "head_and_scaling_sweep": {
            "hypothesis": "Fixed-width head counts retain projection parameters, increase explicit map entries, and scaling reduces concentration for these fixed logits.",
            "result": sweep,
            "interpretation_candidate": "Head layout changes routing at fixed parameter count; entropy describes routing concentration only.",
            "limitation": "Untrained single-seed demonstration; changing head count also changes d_head and grouping, so this cannot rank model quality.",
        },
        "review_status": "author review required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--output", type=Path, help="JSON destination; default is a new ignored, dated outputs/run-*.json")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    torch.set_num_threads(1)
    record = run(args.seed)
    destination = args.output or Path(__file__).resolve().parent / "outputs" / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # An explicit path must also be new: preserve historical experiment records.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(record, indent=2, allow_nan=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
