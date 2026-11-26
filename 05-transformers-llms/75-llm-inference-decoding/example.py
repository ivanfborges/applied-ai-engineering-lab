"""Run synthetic decoding controls; retain a new dated, ignored JSON record."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from from_scratch import decoding_distribution, entropy_bits, generate, greeting_model


def run_experiments(seed=42, samples=10_000):
    base = np.array([0.85, 0.07, 0.04, 0.03, 0.01])
    temperature_rows = []
    for temperature in (0.2, 0.7, 1.0, 1.5, 2.0):
        probabilities = decoding_distribution(np.log(base), temperature=temperature)
        draws = np.random.default_rng(seed).choice(5, samples, p=probabilities)
        empirical = np.bincount(draws, minlength=5) / samples
        temperature_rows.append(dict(
            temperature=temperature, probabilities=probabilities.tolist(),
            entropy_bits=entropy_bits(probabilities), empirical=empirical.tolist(),
            max_absolute_frequency_error=float(np.max(np.abs(empirical - probabilities))),
        ))
    filtering_rows = []
    for name, distribution in (("peaked", base), ("uniform", np.full(5, 0.2))):
        for temperature in (0.7, 1.0, 2.0):
            for top_k in (None, 3):
                for top_p in (0.5, 0.9, 1.0):
                    probabilities = decoding_distribution(
                        np.log(distribution), temperature=temperature,
                        top_k=top_k, top_p=top_p,
                    )
                    support = probabilities > 0
                    filtering_rows.append(dict(
                        distribution=name, temperature=temperature, top_k=top_k, top_p=top_p,
                        candidates=int(support.sum()),
                        retained_original_mass=float(distribution[support].sum()),
                        entropy_bits=entropy_bits(probabilities),
                    ))
    vocabulary = ("A", "B", "<EOS>")

    def branching_model(context):
        if len(context) >= 12:
            return [-10.0, -10.0, 10.0]
        return [1.0, 0.0, -10.0] if not context or context[-1] == 0 else [0.0, 1.0, -10.0]

    runs = [generate(branching_model, (), vocabulary, eos_id=2, seed=value)
            for value in (seed, seed, seed + 1)]
    greedy = [generate(branching_model, (), vocabulary, eos_id=2, seed=value, temperature=0)
              for value in (seed, seed + 1)]
    greeting = ("Hello", " world", "!", "<EOS>")
    stop_runs = {name: asdict(generate(greeting_model, (), greeting, eos_id=3, temperature=0, **options))
                for name, options in (
                    ("eos", {}), ("stop", {"stop_sequences": ("!",)}),
                    ("cross_fragment", {"stop_sequences": ("lo wo",)}),
                    ("budget", {"max_new_tokens": 2}),
                    ("zero_budget", {"max_new_tokens": 0}),
                )}
    return [
        dict(name="temperature", hypothesis="Higher T increases unfiltered entropy for fixed nonuniform logits.",
             configuration=dict(base_probabilities=base.tolist(), seed=seed, samples_per_temperature=samples),
             result=temperature_rows,
             interpretation_candidate="Compare monotonic theoretical entropy and finite-sample frequency errors.",
             limitation="Independent single-step draws; no language quality or convergence-rate estimate."),
        dict(name="filters", hypothesis="Nucleus size adapts to distribution and preceding transformations.",
             configuration=dict(order="temperature -> top-k -> renormalize -> top-p", cases=36),
             result=filtering_rows,
             interpretation_candidate="Compare candidate counts while keeping distribution and other settings fixed.",
             limitation="Five synthetic logits; retained original mass differs from mass after temperature/top-k."),
        dict(name="reproducibility", hypothesis="Same seed and call sequence reproduce this controlled generator.",
             configuration=dict(seeds=[seed, seed, seed + 1], max_new_tokens=20, transition_limit=12),
             result=dict(sampled_runs=[asdict(run) for run in runs], same_seed_equal=runs[0] == runs[1],
                         different_seed_equal=runs[0] == runs[2], greedy_seed_equal=greedy[0] == greedy[1]),
             interpretation_candidate="Separate reproducibility of a local generator from output diversity.",
             limitation="Different seeds can coincide; no cross-version or hosted-inference guarantee."),
        dict(name="termination", hypothesis="EOS, completed stop strings and budget exhaustions have distinct outcomes.",
             configuration=dict(vocabulary=list(greeting), temperature=0, seed=seed), result=stop_runs,
             interpretation_candidate="Compare visible text with sampled token counts, including cross-fragment stops.",
             limitation="Concatenated fragments only; no tokenizer, streaming, KV cache or context-window enforcement."),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if args.seed < 0 or args.samples < 1:
        parser.error("seed must be >= 0 and samples must be >= 1")
    now = datetime.now(timezone.utc)
    record = dict(created_at_utc=now.isoformat(), data="synthetic; no pretrained model or external dataset",
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   platform=platform.platform(), bit_generator="PCG64"),
                  review_status="author review required",
                  experiments=run_experiments(args.seed, args.samples))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"run-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Record: {path}")
    for row in record["experiments"][0]["result"]:
        print(f"T={row['temperature']}: H={row['entropy_bits']:.6f} bits; "
              f"max frequency error={row['max_absolute_frequency_error']:.6f}")
    print(json.dumps(record["experiments"][2]["result"], indent=2))
    print(json.dumps(record["experiments"][3]["result"], indent=2))
    print("Interpretation candidates: author review required.")


if __name__ == "__main__":
    main()
