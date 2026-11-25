"""CPU encoder-decoder Transformer on synthetic fixed-length sequence reversal."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit('Install repository extras: python -m pip install -e ".[dev,deep-learning]"') from exc

from from_scratch import positive_int

BOS, EOS = 0, 1


@dataclass(frozen=True)
class Config:
    seed: int = 42
    vocab_size: int = 32
    sequence_length: int = 5
    train_size: int = 512
    test_size: int = 128
    d_model: int = 32
    heads: int = 4
    layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    steps: int = 400
    batch_size: int = 32
    learning_rate: float = 0.001

    def __post_init__(self):
        for name in ("vocab_size", "sequence_length", "train_size", "test_size", "d_model",
                     "heads", "layers", "d_ff", "steps", "batch_size"):
            positive_int(getattr(self, name), name)
        if self.vocab_size < 3 or self.d_model % self.heads:
            raise ValueError("vocab_size must be >= 3 and d_model divisible by heads")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        space = (self.vocab_size - 2) ** self.sequence_length
        if space > np.iinfo(np.int64).max or self.train_size + self.test_size > space:
            raise ValueError("requested unique split does not fit the supported sequence space")


def make_split(config):
    """Sample unique base-(vocab-2) sequence IDs without constructing the full space."""
    base = config.vocab_size - 2
    rng = np.random.default_rng(config.seed)
    ranks = rng.choice(base ** config.sequence_length,
                       config.train_size + config.test_size, replace=False)
    tokens = np.empty((len(ranks), config.sequence_length), dtype=np.int64)
    for position in range(config.sequence_length):
        tokens[:, position] = ranks % base + 2
        ranks //= base
    tokens = torch.from_numpy(tokens)
    return tokens[:config.train_size], tokens[config.train_size:]


def validate_tokens(tokens, vocab_size, name):
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
            or tokens.device.type != "cpu" or tokens.ndim != 2 or min(tokens.shape) < 1):
        raise ValueError(f"{name} must be a nonempty CPU long tensor with shape (batch, length)")
    if (tokens < 0).any() or (tokens >= vocab_size).any():
        raise ValueError(f"{name} token IDs must lie in [0, vocab_size)")


def reversal_targets(source):
    if (not isinstance(source, torch.Tensor) or source.dtype != torch.long
            or source.device.type != "cpu" or source.ndim != 2 or min(source.shape) < 1
            or (source < 2).any()):
        raise ValueError("source must be a nonempty CPU long tensor of content IDs >= 2")
    reverse = source.flip(1)
    boundary = torch.full((len(source), 1), BOS, dtype=torch.long)
    decoder_input = torch.cat((boundary, reverse), dim=1)
    target = torch.cat((reverse, torch.full_like(boundary, EOS)), dim=1)
    return decoder_input, target


def causal_mask(length):
    length = positive_int(length, "length")
    return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.sequence_length + 1, config.d_model)
        self.transformer = nn.Transformer(
            d_model=config.d_model, nhead=config.heads,
            num_encoder_layers=config.layers, num_decoder_layers=config.layers,
            dim_feedforward=config.d_ff, dropout=config.dropout,
            batch_first=True, norm_first=False, activation="relu",
        )
        self.output = nn.Linear(config.d_model, config.vocab_size)
        # Independently initialize matrices, including those in cloned stack layers.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def embed(self, tokens):
        validate_tokens(tokens, self.config.vocab_size, "tokens")
        if tokens.shape[1] > self.config.sequence_length + 1:
            raise ValueError("sequence exceeds the learned positional table")
        positions = torch.arange(tokens.shape[1])
        return self.embedding(tokens) * math.sqrt(self.config.d_model) + self.position(positions)

    def forward(self, source, decoder_input, *, causal=True):
        if not isinstance(causal, bool):
            raise ValueError("causal must be boolean")
        src, tgt = self.embed(source), self.embed(decoder_input)
        if len(src) != len(tgt):
            raise ValueError("source and decoder batch sizes must agree")
        mask = causal_mask(decoder_input.shape[1]) if causal else None
        return self.output(self.transformer(src, tgt, tgt_mask=mask))

    @torch.no_grad()
    def generate(self, source):
        """Fixed length + EOS slot; encode once, recompute the decoder prefix (no KV cache)."""
        if self.training:
            raise ValueError("call model.eval() before generation")
        validate_tokens(source, self.config.vocab_size, "source")
        if source.shape[1] != self.config.sequence_length or (source < 2).any():
            raise ValueError("generation requires configured-length content sequences")
        memory = self.transformer.encoder(self.embed(source))
        prefix = torch.full((len(source), 1), BOS, dtype=torch.long)
        for _ in range(self.config.sequence_length + 1):
            hidden = self.transformer.decoder(self.embed(prefix), memory,
                                               tgt_mask=causal_mask(prefix.shape[1]))
            next_token = self.output(hidden[:, -1]).argmax(dim=-1, keepdim=True)
            prefix = torch.cat((prefix, next_token), dim=1)
        return prefix[:, 1:]


def sequence_metrics(prediction, target):
    if prediction.shape != target.shape or target.ndim != 2 or min(target.shape) < 1:
        raise ValueError("prediction and target must have the same nonempty (batch, length) shape")
    correct = prediction == target
    return {"token_accuracy": correct.float().mean().item(),
            "exact_match": correct.all(dim=1).float().mean().item()}


def run(config):
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    train, test = make_split(config)
    model = Seq2SeqTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 1)
    train_input, train_target = reversal_targets(train)
    model.eval()
    with torch.no_grad():
        initial_loss = nn.functional.cross_entropy(
            model(train, train_input).flatten(0, 1), train_target.flatten()).item()
    for _ in range(config.steps):
        model.train()
        indices = torch.randint(len(train), (config.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(train[indices], train_input[indices])
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), train_target[indices].flatten())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    # Fixed budget: no held-out feedback, early stopping or checkpoint selection.
    model.eval()
    test_input, test_target = reversal_targets(test)
    with torch.no_grad():
        final_loss = nn.functional.cross_entropy(
            model(train, train_input).flatten(0, 1), train_target.flatten()).item()
        logits = model(test, test_input)
        test_loss = nn.functional.cross_entropy(logits.flatten(0, 1), test_target.flatten()).item()
    generated = model.generate(test)
    return {
        "data": "synthetic unique fixed-length reversal sequences; no external dataset",
        "config": asdict(config),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "platform": platform.platform(),
                        "device": "cpu", "torch_threads": torch.get_num_threads()},
        "train_eval_cross_entropy_before": initial_loss,
        "train_eval_cross_entropy_after": final_loss,
        "test_teacher_forced_cross_entropy": test_loss,
        "test_teacher_forced": sequence_metrics(logits.argmax(-1), test_target),
        "test_autoregressive": sequence_metrics(generated, test_target),
        "first_test_source": test[0].tolist(), "first_test_target": test_target[0].tolist(),
        "first_test_generated": generated[0].tolist(),
        "interpretation_status": "author review required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    config = Config(steps=args.steps, seed=args.seed)
    report = run(config)
    report["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output_dir / f"run_{stamp}_seed{config.seed}_steps{config.steps}.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Local run record: {destination}")


if __name__ == "__main__":
    main()
