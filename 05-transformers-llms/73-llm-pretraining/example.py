"""CPU synthetic CLM/MLM training; no pretrained weights or external corpus."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
import platform
from time import perf_counter

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit('Install the canonical extra: python -m pip install -e ".[deep-learning]"') from exc


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Config:
    objective: str = "clm"
    seed: int = 42
    steps: int = 200
    width: int = 32
    heads: int = 2
    learning_rate: float = 0.01
    masked_tokens: int = 1

    def __post_init__(self):
        if self.objective not in ("clm", "mlm"):
            raise ValueError("objective must be clm or mlm")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("steps", "width", "heads", "masked_tokens"):
            positive_integer(getattr(self, name), name)
        if self.width % self.heads or self.masked_tokens > 4:
            raise ValueError("width must be divisible by heads; masked_tokens must be <= 4")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (float, int))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")


def make_corpus():
    """Hold out the four diagonal animal-food pairs, including all their tokens in train."""
    train, validation = [], []
    for i, animal in enumerate(("cat", "dog", "bird", "rabbit")):
        for j, food in enumerate(("fish", "meat", "seeds", "carrots")):
            sentence = ["<bos>", "the", animal, "likes", food, "<eos>"]
            (validation if i == j else train).append(sentence)
    vocabulary = sorted({token for sentence in train for token in sentence} | {"<mask>"})
    stoi = {token: i for i, token in enumerate(vocabulary)}

    def encode(sentences):
        return torch.tensor([[stoi[token] for token in sentence] for sentence in sentences])

    return encode(train), encode(validation), vocabulary


def validate_sequences(tokens, minimum_length=2):
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
            or tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < minimum_length
            or (tokens < 0).any()):
        raise ValueError(f"tokens must be a nonempty batch of length >= {minimum_length} with nonnegative long IDs")


def clm_batch(tokens):
    validate_sequences(tokens)
    return tokens[:, :-1].clone(), tokens[:, 1:].clone()


def mlm_batch(tokens, selected, mask_id):
    """All selected tokens become <mask>; unselected targets use ignore_index=-100."""
    validate_sequences(tokens)
    if (not isinstance(selected, torch.Tensor) or selected.dtype != torch.bool
            or selected.shape != tokens.shape or not selected.any(dim=1).all()
            or selected[:, 0].any() or selected[:, -1].any()):
        raise ValueError("select at least one interior token per sequence, excluding BOS/EOS")
    positive_integer(mask_id, "mask_id")
    if (tokens == mask_id).any():
        raise ValueError("uncorrupted sequences must not contain the reserved mask ID")
    corrupted, targets = tokens.clone(), torch.full_like(tokens, -100)
    corrupted[selected], targets[selected] = mask_id, tokens[selected]
    return corrupted, targets


def fixed_mlm_batch(tokens, mask_id, masked_tokens):
    """Enumerate all interior mask combinations, giving deterministic evaluation."""
    validate_sequences(tokens)
    positive_integer(masked_tokens, "masked_tokens")
    if masked_tokens > tokens.shape[1] - 2:
        raise ValueError("masked_tokens exceeds eligible positions")
    patterns = list(combinations(range(1, tokens.shape[1] - 1), masked_tokens))
    expanded = tokens.repeat_interleave(len(patterns), dim=0)
    selected = torch.zeros_like(expanded, dtype=torch.bool)
    for row in range(len(expanded)):
        selected[row, list(patterns[row % len(patterns)])] = True
    return mlm_batch(expanded, selected, mask_id)


class TinyLanguageModel(nn.Module):
    """One self-attention/FFN block; fixed-length inputs have no padding."""

    def __init__(self, vocabulary_size, width=32, heads=2):
        super().__init__()
        for value, name in ((vocabulary_size, "vocabulary_size"), (width, "width"), (heads, "heads")):
            positive_integer(value, name)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.token_embedding = nn.Embedding(vocabulary_size, width)
        self.position_embedding = nn.Embedding(6, width)
        self.block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=2 * width, dropout=0.0, batch_first=True,
        )
        self.output = nn.Linear(width, vocabulary_size)

    def forward(self, tokens, causal):
        validate_sequences(tokens, minimum_length=1)
        if not isinstance(causal, bool):
            raise ValueError("causal must be Boolean")
        length = tokens.shape[1]
        if length > 6 or (tokens >= self.token_embedding.num_embeddings).any():
            raise ValueError("input exceeds the six-position context or vocabulary")
        hidden = self.token_embedding(tokens) + self.position_embedding(torch.arange(length))
        # Transformer module Boolean masks use True to BLOCK future keys.
        blocked = torch.ones(length, length, dtype=torch.bool).triu(1) if causal else None
        return self.output(self.block(hidden, src_mask=blocked))


def token_loss(logits, targets):
    if (logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long
            or not torch.isfinite(logits).all() or not (targets != -100).any()
            or ((targets != -100) & ((targets < 0) | (targets >= logits.shape[-1]))).any()):
        raise ValueError("finite batch/sequence/vocabulary logits and valid nonempty targets required")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def evaluate(model, batch, causal):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            inputs, targets = batch
            logits = model(inputs, causal)
            valid = targets != -100
            result = {"nll": float(token_loss(logits, targets)),
                      "accuracy": float((logits.argmax(-1)[valid] == targets[valid]).float().mean()),
                      "supervised_tokens": int(valid.sum())}
            if causal:
                result["perplexity"] = math.exp(result["nll"])
            return result
    finally:
        model.train(was_training)


def run_experiment(config=Config()):
    torch.set_num_threads(1)
    train, validation, vocabulary = make_corpus()
    mask_id = vocabulary.index("<mask>")
    causal = config.objective == "clm"
    train_eval = clm_batch(train) if causal else fixed_mlm_batch(train, mask_id, config.masked_tokens)
    val_eval = clm_batch(validation) if causal else fixed_mlm_batch(validation, mask_id, config.masked_tokens)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        generator = torch.Generator().manual_seed(config.seed + 1)
        model = TinyLanguageModel(len(vocabulary), config.width, config.heads)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        initial_train = evaluate(model, train_eval, causal)
        initial_validation = evaluate(model, val_eval, causal)
        started = perf_counter()
        model.train()
        for _ in range(config.steps):
            if causal:
                batch = clm_batch(train)
            else:
                # Fresh uniform subsets of interior positions for every update.
                ranks = torch.rand((len(train), 4), generator=generator).argsort(dim=1)
                selected = torch.zeros_like(train, dtype=torch.bool)
                selected.scatter_(1, ranks[:, :config.masked_tokens] + 1, True)
                batch = mlm_batch(train, selected, mask_id)
            optimizer.zero_grad(set_to_none=True)
            loss = token_loss(model(batch[0], causal), batch[1])
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step()
        training_seconds = perf_counter() - started
        final_train = evaluate(model, train_eval, causal)
        final_validation = evaluate(model, val_eval, causal)
        parameters = sum(p.numel() for p in model.parameters())
        nonembedding = parameters - model.token_embedding.weight.numel() - model.position_embedding.weight.numel()
    processed = config.steps * len(train) * (5 if causal else 6)
    supervised = config.steps * len(train) * (5 if causal else config.masked_tokens)
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A small self-supervised model can fit observed templates; held-out pair prediction may remain weak.",
        "configuration": {**asdict(config), "device": "cpu", "dtype": "float32", "threads": 1,
                          "layers": 1, "ffn_width": 2 * config.width, "dropout": 0.0,
                          "weight_decay": 0.01, "batch_size": len(train), "checkpoint": "final update"},
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {"source": "synthetic animal-food Cartesian product", "vocabulary": vocabulary,
                 "train_sentences": len(train), "validation_sentences": len(validation),
                 "split": "i != j train; i == j validation; no external dataset"},
        "budget": {"parameters_total": parameters, "parameters_nonembedding": nonembedding,
                   "unique_train_sequences": len(train), "unique_train_token_positions": train.numel(),
                   "processed_input_tokens": processed, "supervised_tokens": supervised,
                   "training_seconds": training_seconds,
                   "approximate_6ND_flops_nonembedding": 6 * nonembedding * processed,
                   "compute_note": "Theoretical dense proxy, not profiled FLOPs; repeated exposures are not new data."},
        "initial_train": initial_train, "initial_validation": initial_validation,
        "final_train": final_train, "final_validation": final_validation,
        "interpretation_candidate": "Fitting templates need not transfer to unseen pairs; author review required.",
        "limitation": "One seed, 12 training sentences, no test set or scaling-law fit; CLM/MLM NLL are different objectives.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("clm", "mlm"), default="clm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--masked-tokens", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON path, normally inside ignored outputs/")
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        config = Config(**args)
    except ValueError as exc:
        parser.error(str(exc))
    if output is not None and output.exists():
        parser.error("output already exists; use a new dated path to preserve historical records")
    report = run_experiment(config)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
