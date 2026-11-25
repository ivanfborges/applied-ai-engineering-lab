"""Inspect two pretrained byte-BPE encodings and hypothetical request budgets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform


ENCODINGS = ("cl100k_base", "o200k_base")
# Handwritten synthetic probes, not a parallel corpus or an OCR benchmark.
SAMPLES = {
    "English": "Machine learning is transforming industries.",
    "Portuguese": "O aprendizado de máquina está transformando indústrias.",
    "Technical": "Retrieval-augmented generation with contextual embeddings.",
    "Python": "def predict(x):\n    return model.predict(x)",
    "JSON": '{"model": "demo", "temperature": 0.2}',
    "Clean": "Environmental regulation",
    "OCR noise": "Env1ronmentaI regu1at ion",
    "Unicode": "Inteligência artificial 🚀 中文",
}


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def context_budget(
    components: dict[str, int], output_reserve: int, context_limit: int,
    *, max_input: int | None = None, max_output: int | None = None,
) -> dict:
    """Arithmetic for already-counted components of a hypothetical shared window."""
    if not isinstance(components, dict) or any(
        not isinstance(name, str) or not name for name in components
    ):
        raise ValueError("components must map nonempty names to token counts")
    for name, count in components.items():
        nonnegative_int(count, name)
    nonnegative_int(output_reserve, "output_reserve")
    nonnegative_int(context_limit, "context_limit")
    if context_limit == 0:
        raise ValueError("context_limit must be positive")
    for name, limit in (("max_input", max_input), ("max_output", max_output)):
        if limit is not None:
            nonnegative_int(limit, name)
    input_tokens = sum(components.values())
    remaining = context_limit - input_tokens - output_reserve
    fits = (remaining >= 0
            and (max_input is None or input_tokens <= max_input)
            and (max_output is None or output_reserve <= max_output))
    return {"input_tokens": input_tokens, "output_reserve": output_reserve,
            "remaining": remaining, "fits": fits}


def estimated_cost(input_tokens: int, output_tokens: int,
                   input_price: Decimal | int | float,
                   output_price: Decimal | int | float) -> Decimal:
    """Hypothetical prices per million tokens; excludes cache/tools/reasoning fees."""
    nonnegative_int(input_tokens, "input_tokens")
    nonnegative_int(output_tokens, "output_tokens")
    prices = []
    for value in (input_price, output_price):
        if isinstance(value, bool) or not isinstance(value, (Decimal, int, float)):
            raise ValueError("prices must be finite nonnegative numbers")
        price = Decimal(str(value))
        if not price.is_finite() or price < 0:
            raise ValueError("prices must be finite nonnegative numbers")
        prices.append(price)
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def vocabulary_budget(vocabulary_size: int, width: int, *, tied: bool = True) -> dict:
    """Weight-only calculation for an embedding and a bias-free output projection."""
    for name, value in (("vocabulary_size", vocabulary_size), ("width", width)):
        nonnegative_int(value, name)
        if value == 0:
            raise ValueError(f"{name} must be positive")
    if type(tied) is not bool:
        raise ValueError("tied must be boolean")
    parameters = vocabulary_size * width * (1 if tied else 2)
    return {"parameters": parameters, "fp16_weight_bytes": 2 * parameters}


def inspect_text(encoder, text: str) -> dict:
    """Treat all input as ordinary text; verify full UTF-8 byte reconstruction."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    # Reject unpaired surrogates instead of accepting tiktoken's repair behavior.
    try:
        original = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("text must be valid UTF-8 encodable Unicode") from error
    ids = encoder.encode_ordinary(text)
    pieces = [encoder.decode_single_token_bytes(token) for token in ids]
    if b"".join(pieces) != original or encoder.decode(ids, errors="strict") != text:
        raise AssertionError("complete encode/decode reconstruction failed")
    return {"tokens": len(ids), "characters": len(text), "utf8_bytes": len(original),
            "characters_per_token": len(text) / len(ids) if ids else None,
            "bytes_per_token": len(original) / len(ids) if ids else None,
            "ids": ids, "token_bytes_hex": [piece.hex() for piece in pieces]}


def run_experiment() -> dict:
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError('Install the canonical extra: python -m pip install -e ".[dev,nlp]"') from error
    results = {}
    for name in ENCODINGS:
        encoder = tiktoken.get_encoding(name)
        results[name] = {
            "id_capacity": encoder.n_vocab,
            "mergeable_tokens": len(encoder.token_byte_values()),
            "special_tokens": len(encoder.special_tokens_set),
            "samples": {label: inspect_text(encoder, text) for label, text in SAMPLES.items()},
        }
    components = {"system": 1500, "history": 4000, "query": 300,
                  "retrieval": 14000, "overhead": 500}
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {"python": platform.python_version(),
                     "tiktoken": importlib.metadata.version("tiktoken")},
        "data": "handwritten synthetic probes; no downstream task or API call",
        "samples": SAMPLES,
        "samples_sha256": hashlib.sha256(json.dumps(SAMPLES, ensure_ascii=True,
            sort_keys=True).encode("utf-8")).hexdigest(),
        "encodings": results,
        "hypothetical_budget": context_budget(components, 4000, 32768),
        "hypothetical_cost_usd": str(estimated_cost(6000, 1000, 2, 8)),
        "vocabulary_50000_width4096": vocabulary_budget(50000, 4096),
        "interpretation_status": "author review required",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON record; use an ignored outputs/ directory")
    args = parser.parse_args()
    report = run_experiment()
    serialized = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
    print(serialized)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Preserve the previous run; choose a new path: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
