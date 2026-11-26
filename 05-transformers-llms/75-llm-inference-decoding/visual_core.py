"""Synthetic experiments shared by exports and the live decoding playground."""

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np

from from_scratch import decoding_distribution, entropy_bits, generate, softmax, _integer

TOKENS = ("river", "forest", "city", "ocean", "desert", "garden", "mountain", "island", "village", "meadow")
PRESETS = {
    "Peaked": np.array([4.5, 1.5, 1.0, 0.7, 0.2, 0.0, -0.2, -0.5, -0.8, -1.0]),
    "Moderate": np.array([2.0, 1.6, 1.3, 1.0, 0.7, 0.4, 0.1, -0.2, -0.5, -0.8]),
    "Diffuse": np.array([0.25, 0.20, 0.15, 0.10, 0.05, 0.0, -0.05, -0.10, -0.15, -0.20]),
}
GROUPS = ("temperature", "top-k", "top-p", "interactions", "generation", "stopping", "reproducibility", "sampling", "cache")
STRUCTURED_VOCABULARY = ('{"ok":', ' true', '}', '<EN', 'D>', '<EOS>')
STRATEGIES = {
    "Greedy": {"temperature": 0.0},
    "Full sampling": {},
    "Temperature sampling": {"temperature": 1.7},
    "Top-k sampling": {"top_k": 1},
    "Top-p sampling": {"top_p": 0.7},
}


@dataclass(frozen=True)
class LabConfig:
    seed: int = 275
    quick: bool = False
    samples: int = 10_000
    beam_width: int = 2

    def __post_init__(self):
        _integer(self.seed, "seed")
        _integer(self.samples, "samples", 1)
        _integer(self.beam_width, "beam_width", 1)
        if self.beam_width > 16:
            raise ValueError("beam_width must be <= 16 for this bounded laboratory.")

    @property
    def frames(self):
        return 10 if self.quick else 32

    @property
    def grid_size(self):
        return 13 if self.quick else 31


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def stages(logits, temperature=1.0, top_k=None, top_p=1.0):
    """Use the analytical decoder for every stage; no second filter implementation."""
    z = np.asarray(logits, dtype=float)
    final = decoding_distribution(z, temperature=temperature, top_k=top_k, top_p=top_p)
    original = softmax(z)
    tempered = softmax(z, temperature)
    after_k = decoding_distribution(z, temperature=temperature, top_k=top_k)
    raw_k = np.where(after_k > 0, tempered, 0)
    raw_p = np.where(final > 0, after_k, 0)
    entropy = entropy_bits(final)
    return dict(logits=z, scaled_logits=None if temperature == 0 else z / temperature,
                original=original, tempered=tempered, raw_k=raw_k, after_k=after_k,
                raw_p=raw_p, final=final, temperature=temperature, top_k=top_k, top_p=top_p,
                entropy=entropy, effective=2 ** entropy, maximum=float(final.max()),
                candidates=int(np.count_nonzero(final)),
                retained_original_mass=float(original[final > 0].sum()))


def landscape(logits, temperatures, thresholds, top_k=None):
    records = [[stages(logits, float(t), top_k, float(p)) for t in temperatures] for p in thresholds]
    return dict(temperatures=np.asarray(temperatures), thresholds=np.asarray(thresholds), top_k=top_k,
                **{key: np.array([[row[key] for row in rows] for rows in records])
                   for key in ("entropy", "effective", "maximum", "candidates")})


def binary_logits(context):
    """A/B tree: root .6/.4, then .55/.45 after A or .95/.05 after B."""
    if not context:
        probabilities = [0.6, 0.4]
    elif len(context) == 1:
        probabilities = [0.55, 0.45] if context[0] == 0 else [0.95, 0.05]
    else:
        probabilities = [0.8, 0.2] if context[-1] == 0 else [0.3, 0.7]
    return np.log(probabilities)


def sequence_score(ids):
    score = 0.0
    context = ()
    for token in ids:
        score += float(np.log(softmax(binary_logits(context))[token]))
        context += (token,)
    return score


def exhaustive_search(depth=3):
    _integer(depth, "depth", 1)
    if depth > 8:
        raise ValueError("Exhaustive tree depth must be <= 8.")
    records = [dict(ids=ids, log_probability=sequence_score(ids)) for ids in product(range(2), repeat=depth)]
    return sorted(records, key=lambda row: (-row["log_probability"], row["ids"]))


def beam_search(depth=3, width=2):
    _integer(depth, "depth", 1)
    _integer(width, "width", 1)
    if depth > 8 or width > 256:
        raise ValueError("Bounded beam search requires depth <= 8 and width <= 256.")
    beam, history = [((), 0.0)], []
    for _ in range(depth):
        candidates = [(ids + (token,), logp + float(np.log(softmax(binary_logits(ids))[token])))
                      for ids, logp in beam for token in range(2)]
        beam = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:width]
        history.append([dict(ids=ids, log_probability=logp) for ids, logp in beam])
    return dict(best=dict(ids=beam[0][0], log_probability=beam[0][1]), history=history)


def sample_path(seed=275, depth=3, **decoding):
    _integer(seed, "seed")
    _integer(depth, "depth", 1)
    if depth > 8:
        raise ValueError("Sampled tree depth must be <= 8.")
    rng = np.random.default_rng(seed)
    ids, records = (), []
    model_logp = policy_logp = 0.0
    for step in range(depth):
        logits = binary_logits(ids)
        state = stages(logits, **decoding)
        token = int(rng.choice(2, p=state["final"]))
        model_logp += float(np.log(state["original"][token]))
        policy_logp += float(np.log(state["final"][token]))
        records.append(dict(step=step, context=ids, selected=token, state=state,
                            model_log_probability=model_logp, policy_log_probability=policy_logp))
        ids += (token,)
    return dict(ids=ids, log_probability=model_logp, policy_log_probability=policy_logp, steps=records)


def eos_logits(context):
    """Variable-length fixture: EOS becomes more likely as context grows."""
    eos_probability = min(0.05 + 0.07 * len(context), 0.8)
    weight = 0.7 if not context or context[-1] == 0 else 0.3
    return np.log([(1 - eos_probability) * weight, (1 - eos_probability) * (1 - weight), eos_probability])


def structured_logits(context, delimiter=False):
    values = np.full(6, -20.0)
    next_id = min(len(context), 5) if delimiter or len(context) < 3 else 5
    values[next_id] = 20
    return values


def trace_generation(model, vocabulary, eos_id, **options):
    captured = []

    def recording_model(context):
        logits = model(context)
        captured.append((context, np.asarray(logits)))
        return logits

    result = generate(recording_model, (), vocabulary, eos_id=eos_id, **options)
    trace = []
    for step, ((context, logits), token) in enumerate(zip(captured, result.sampled_ids)):
        current_ids = result.sampled_ids[:step + 1]
        text = "".join(vocabulary[i] for i in current_ids if i != eos_id)
        if step == len(captured) - 1:
            text = result.text
        state = stages(logits, **{k: options[k] for k in ("temperature", "top_k", "top_p") if k in options})
        trace.append(dict(context=context, selected=token, state=state, text=text,
                          remaining=options.get("max_new_tokens", 20) - step - 1,
                          reason=result.finish_reason if step == len(captured) - 1 else "continue"))
    return dict(result=asdict(result), trace=trace)


def sequence_agreement(sequences):
    """All unordered run pairs; token agreement includes EOS and excludes double padding."""
    if len(sequences) < 2:
        raise ValueError("Need at least two sequences for pairwise agreement.")
    lengths = np.array([len(row) for row in sequences])
    matrix = np.full((len(sequences), max(lengths, default=0)), -1, dtype=int)
    for i, row in enumerate(sequences):
        matrix[i, :len(row)] = row
    exact, token, count = 0, 0.0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            exact += tuple(sequences[i]) == tuple(sequences[j])
            active = (matrix[i] >= 0) | (matrix[j] >= 0)
            token += float(np.mean(matrix[i, active] == matrix[j, active])) if active.any() else 1.0
            count += 1
    return dict(matrix=matrix, lengths=lengths, exact_agreement=exact / count,
                token_agreement=token / count, unique=len({tuple(row) for row in sequences}))


def reproducibility(config):
    rows = {}
    shared = np.random.default_rng(config.seed)
    for mode in ("Greedy", "Reset same seed", "Different seeds", "Shared RNG"):
        sequences = []
        for run in range(16):
            options = dict(temperature=0 if mode == "Greedy" else 1,
                           seed=config.seed + run if mode == "Different seeds" else config.seed,
                           max_new_tokens=12)
            if mode == "Shared RNG":
                options["rng"] = shared
            result = generate(eos_logits, (), ("A", "B", "<EOS>"), eos_id=2, **options)
            sequences.append(result.sampled_ids)
        rows[mode] = sequence_agreement(sequences)
    return rows


def sampling_experiment(config):
    records = []
    for name, options in (("T=0.4", {"temperature": 0.4}), ("T=2", {"temperature": 2}),
                          ("k=2", {"top_k": 2}), ("k=6", {"top_k": 6}),
                          ("p=0.5", {"top_p": 0.5}), ("p=0.9", {"top_p": 0.9})):
        probabilities = decoding_distribution(PRESETS["Moderate"], **options)
        draws = np.random.default_rng(config.seed).choice(len(TOKENS), config.samples, p=probabilities)
        snapshots = []
        sizes = np.unique(np.geomspace(10, config.samples, config.frames).astype(int))
        sizes = np.unique(np.append(sizes[sizes <= config.samples], config.samples))
        for size in sizes:
            empirical = np.bincount(draws[:size], minlength=len(TOKENS)) / size
            snapshots.append(dict(samples=int(size), empirical=empirical,
                                  total_variation=float(0.5 * np.abs(empirical - probabilities).sum())))
        records.append(dict(name=name, options=options, probabilities=probabilities, snapshots=snapshots))
    return records


def kv_budget(prompt=6, output=8, layers=2, kv_heads=2, head_dim=8, bytes_per_value=4):
    """Theoretical cached K/V storage and per-layer projection position counts.

    Prefill predicts output token 1. To predict output token j+1, feed token j.
    A causal one-layer fixed-projection numerical check accompanies the diagram.
    """
    for name, value in locals().copy().items():
        _integer(value, name, 1)
    lengths = prompt + np.arange(output)
    new_positions = np.r_[prompt, np.ones(output - 1, dtype=int)]
    reused_positions = np.r_[0, lengths[:-1]]
    return dict(prompt=prompt, output=output, layers=layers, kv_heads=kv_heads,
                head_dim=head_dim, bytes_per_value=bytes_per_value, lengths=lengths,
                new_positions=new_positions, reused_positions=reused_positions,
                uncached_projection_positions=lengths,
                cached_kv_bytes=2 * layers * kv_heads * head_dim * bytes_per_value * lengths,
                cached_attention_pairs=np.r_[prompt * (prompt + 1) // 2, lengths[1:]],
                uncached_attention_pairs=lengths * (lengths + 1) // 2)


def kv_equivalence(seed=275, prompt=6, output=8, dimension=8):
    """Actual single-head one-layer attention; fixed synthetic inputs, no training."""
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(prompt + output - 1, dimension))
    wq, wk, wv = rng.normal(size=(3, dimension, dimension)) / np.sqrt(dimension)
    cached_k, cached_v, errors = None, None, []
    for step in range(output):
        end = prompt + step
        visible = inputs[:end]
        q = visible[-1] @ wq
        full_k, full_v = visible @ wk, visible @ wv
        if step == 0:
            cached_k, cached_v = full_k.copy(), full_v.copy()
        else:
            cached_k = np.vstack((cached_k, visible[-1] @ wk))
            cached_v = np.vstack((cached_v, visible[-1] @ wv))
        full = softmax(q @ full_k.T / np.sqrt(dimension)) @ full_v
        cached = softmax(q @ cached_k.T / np.sqrt(dimension)) @ cached_v
        errors.append(float(np.max(np.abs(full - cached))))
    return dict(max_absolute_error=max(errors), step_errors=errors,
                limitation="Single layer, fixed input vectors; no multilayer token-generation benchmark.")


def compute_group(name, config):
    logits = PRESETS["Moderate"]
    if name == "temperature":
        return [stages(logits, float(t)) for t in np.r_[0.0, np.linspace(0.1, 3, config.frames)]]
    if name == "top-k":
        rng = np.random.default_rng(config.seed)
        return [dict(state=stages(logits, top_k=k), selected=int(rng.choice(10, p=decoding_distribution(logits, top_k=k))))
                for k in range(1, 11)]
    if name == "top-p":
        return {preset: [stages(values, top_p=float(p)) for p in np.linspace(0.05, 1, config.frames)]
                for preset, values in PRESETS.items()}
    if name == "interactions":
        return [landscape(logits, np.linspace(0.1, 3, config.grid_size),
                          np.linspace(0.05, 1, config.grid_size), k) for k in (None, 3, 6)]
    if name == "generation":
        return dict(paths={name: sample_path(config.seed, **options) for name, options in STRATEGIES.items()},
                    exact=exhaustive_search(), beam=beam_search(width=config.beam_width),
                    strategies=STRATEGIES, depth=3, beam_width=config.beam_width)
    if name == "stopping":
        runs = {}
        for key, model, options in (
            ("Natural EOS", structured_logits, {"max_new_tokens": 8}),
            ("Output budget", structured_logits, {"max_new_tokens": 2}),
            ("Cross-token stop", lambda context: structured_logits(context, True),
             {"max_new_tokens": 8, "stop_sequences": ("<END>",)}),
        ):
            runs[key] = trace_generation(model, STRUCTURED_VOCABULARY, 5, temperature=0, **options)
        return dict(vocabulary=STRUCTURED_VOCABULARY, runs=runs,
                    complete_budget=trace_generation(structured_logits, STRUCTURED_VOCABULARY, 5,
                                                     temperature=0, max_new_tokens=3))
    if name == "reproducibility":
        return reproducibility(config)
    if name == "sampling":
        return sampling_experiment(config)
    if name == "cache":
        return dict(theory=kv_budget(), numerical_check=kv_equivalence(config.seed))
    raise ValueError(f"Unknown group: {name}")
