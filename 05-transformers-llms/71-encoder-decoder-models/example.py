"""CPU information-flow controls on random blocks, not BERT/GPT/T5 replicas."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit('Install the canonical extra: python -m pip install -e ".[deep-learning]"') from exc

from from_scratch import attention, visibility


def causal_block_mask(length):
    """Transformer module semantics: True BLOCKS a query-key connection."""
    return torch.from_numpy(~visibility(length, length, causal=True))


def run_experiment(seed=71):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        source = torch.randn(1, 5, 16, dtype=torch.float64)
        target = torch.randn(1, 3, 16, dtype=torch.float64)
        encoder = nn.TransformerEncoderLayer(
            16, 4, dim_feedforward=64, dropout=0.0, batch_first=True, dtype=torch.float64
        ).eval()
        # A causal self-attention + FFN stack needs no encoder cross-attention.
        causal = deepcopy(encoder)
        decoder = nn.TransformerDecoderLayer(
            16, 4, dim_feedforward=64, dropout=0.0, batch_first=True, dtype=torch.float64
        ).eval()
        src_mask = causal_block_mask(5)
        tgt_mask = causal_block_mask(3)

        def differences(before, after):
            return torch.linalg.vector_norm(after - before, dim=-1)[0].tolist()

        with torch.inference_mode():
            memory = encoder(source)
            causal_output = causal(source, src_mask=src_mask)
            output = decoder(target, memory, tgt_mask=tgt_mask)
            changed_source = source.clone()
            changed_source[:, -1, 0] += 10.0
            changed_memory = encoder(changed_source)
            changed_target = target.clone()
            changed_target[:, -1, 0] += 10.0

            # Padding must be hidden during source encoding AND cross-attention.
            padding = torch.tensor([[False, False, False, False, True]])
            padded_memory = encoder(source, src_key_padding_mask=padding)
            changed_padded_memory = encoder(changed_source, src_key_padding_mask=padding)
            padded_output = decoder(target, padded_memory, tgt_mask=tgt_mask, memory_key_padding_mask=padding)
            changed_padded_output = decoder(
                target, changed_padded_memory, tgt_mask=tgt_mask, memory_key_padding_mask=padding
            )
            deltas = {
                "encoder_after_source_change": differences(memory, changed_memory),
                "causal_after_source_change": differences(
                    causal_output, causal(changed_source, src_mask=src_mask)
                ),
                "seq2seq_after_source_change": differences(
                    output, decoder(target, changed_memory, tgt_mask=tgt_mask)
                ),
                "seq2seq_after_target_change": differences(
                    output, decoder(changed_target, memory, tgt_mask=tgt_mask)
                ),
                "encoder_valid_after_padding_change": differences(
                    padded_memory[:, :4], changed_padded_memory[:, :4]
                ),
                "seq2seq_after_padding_change": differences(padded_output, changed_padded_output),
            }

            rng = np.random.default_rng(seed)
            q, k, v = rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 6))
            allowed = visibility(3, 5, key_valid=[True, True, True, True, False])
            reference, _ = attention(q, k, v, allowed)
            # Functional SDPA has the opposite Boolean convention to modules.
            actual = F.scaled_dot_product_attention(
                torch.from_numpy(q)[None, None], torch.from_numpy(k)[None, None],
                torch.from_numpy(v)[None, None], attn_mask=torch.from_numpy(allowed), dropout_p=0.0
            )[0, 0].numpy()

    tolerance = 1e-12
    checks = {
        "causal_prefix_unchanged": max(deltas["causal_after_source_change"][:-1]) <= tolerance,
        "target_prefix_unchanged": max(deltas["seq2seq_after_target_change"][:-1]) <= tolerance,
        "padded_source_hidden_from_valid_encoder": max(deltas["encoder_valid_after_padding_change"]) <= tolerance,
        "padded_source_hidden_from_decoder": max(deltas["seq2seq_after_padding_change"]) <= tolerance,
        "numpy_sdpa_agreement": float(np.max(np.abs(reference - actual))) <= tolerance,
    }
    if not all(checks.values()):
        raise RuntimeError(f"information-flow invariant failed: {checks}")
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Causal prefixes and correctly masked padding are invariant; visible source changes may reach targets.",
        "configuration": {
            "seed": seed, "device": "cpu", "dtype": "float64", "threads": 1,
            "batch": 1, "source_length": 5, "target_length": 3, "width": 16,
            "heads": 4, "ffn_width": 64, "layers_per_stack": 1,
            "dropout": 0.0, "norm_first": False, "activation": "relu", "perturbation": "+10 at last position, feature 0",
            "training_updates": 0, "data": "synthetic independent normal hidden vectors",
            "paired_encoder_causal_weights": True,
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "output_shapes": {"encoder": list(memory.shape), "causal": list(causal_output.shape), "seq2seq": list(output.shape)},
        "positionwise_l2_deltas": deltas,
        "numpy_sdpa_max_abs_error": float(np.max(np.abs(reference - actual))),
        "tolerance": tolerance, "checks": checks,
        "interpretation_candidate": "Observed invariance is consistent with the permitted information paths; author review required.",
        "limitation": "One random initialization; no tokenizer, positional encoding, task head, training, generation, KV cache or language-quality evaluation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, help="Optional new JSON record, normally under ignored outputs/.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new filename to preserve historical records")
    result = run_experiment(args.seed)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
