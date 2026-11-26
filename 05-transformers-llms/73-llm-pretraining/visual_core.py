"""Measured synthetic pretraining and analytical budgets, independent of rendering."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from example import clm_batch, mlm_batch, token_loss


DOMAINS = {
    "animals": {"subjects": ["cat", "dog", "bird", "rabbit"],
                "verbs": ["likes", "eats", "finds", "sees"],
                "objects": ["fish", "meat", "seeds", "carrots"]},
    "technical": {"subjects": ["model", "engineer", "system", "scientist"],
                  "verbs": ["learns", "analyzes", "finds", "uses"],
                  "objects": ["patterns", "data", "tokens", "signals"]},
}


def finite_positive(value, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class LabConfig:
    seed: int = 73
    steps: int = 240
    width: int = 32
    heads: int = 2
    layers: int = 2
    ffn: int = 64
    batch_size: int = 32
    learning_rate: float = 0.005
    masking_probability: float = 0.25
    snapshots: int = 13
    device: str = "cpu"

    def __post_init__(self):
        for name in ("steps", "width", "heads", "layers", "ffn", "batch_size", "snapshots"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width < 3 or self.width % self.heads or self.snapshots < 2:
            raise ValueError("width >= 3 must divide by heads; snapshots must be >= 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        finite_positive(self.learning_rate, "learning_rate")
        finite_positive(self.masking_probability, "masking_probability")
        if self.masking_probability > 1 or self.device not in ("cpu", "cuda"):
            raise ValueError("masking_probability must be <= 1; device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")


def make_lab_corpus(seed=73):
    """Split complete subject/verb combinations before any batching. No windows overlap."""
    rng = np.random.default_rng(seed)
    rows = []
    categories = {"<bos>": "special", "<eos>": "special", "<mask>": "special", "the": "function"}
    for domain, terms in DOMAINS.items():
        for key, category in (("subjects", "subject"), ("verbs", "verb"), ("objects", "object")):
            categories.update({token: category for token in terms[key]})
        for i, subject in enumerate(terms["subjects"]):
            for j, verb in enumerate(terms["verbs"]):
                for obj in terms["objects"]:
                    for grammar in (0, 1):
                        middle = ["the", subject, verb, obj] if grammar == 0 else [subject, verb, "the", obj]
                        rows.append({"tokens": ["<bos>", *middle, "<eos>"], "domain": domain,
                                     "grammar": grammar, "held_out": (i + j) % 4 == 0})
    training = [r for r in rows if not r["held_out"]]
    training = [training[i] for i in rng.permutation(len(training))]
    validation = [r for r in rows if r["held_out"]]
    vocabulary = sorted(categories)
    stoi = {token: i for i, token in enumerate(vocabulary)}

    def encode(records):
        return [[stoi[t] for t in r["tokens"]] for r in records]

    base = training[:96]
    duplicated = base[:24] * 4
    single = [r for r in training if r["grammar"] == 0]
    animal = [r for r in training if r["domain"] == "animals"]
    technical = [r for r in training if r["domain"] == "technical"]
    conflict_ids = encode(base)
    # Intentional inconsistent token-role associations, not factual truth labels.
    subject_ids = [stoi[t] for terms in DOMAINS.values() for t in terms["subjects"]]
    for i in range(0, len(conflict_ids), 4):
        conflict_ids[i][4] = subject_ids[i % len(subject_ids)]
    datasets = {"diverse": encode(base), "restricted": encode(base[:24]),
                "more_unique": encode(training), "duplicated": encode(duplicated),
                "single_grammar": encode(single), "conflicting": conflict_ids,
                "domain_imbalanced": encode(animal[:86] + technical[:10])}
    return {"vocabulary": vocabulary, "categories": [categories[t] for t in vocabulary],
            "datasets": datasets, "validation": encode(validation),
            "split": "Within each domain (subject_index + verb_index) % 4 == 0 held out; both grammars reserved.",
            "source": "Local synthetic symbolic grammar; associations are not claims about diets or facts."}


def corpus_statistics(sequences, vocabulary):
    array = np.asarray(sequences, dtype=int)
    counts = np.bincount(array.ravel(), minlength=len(vocabulary))
    unique = len({tuple(row) for row in sequences})
    technical = {vocabulary.index(t) for t in DOMAINS["technical"]["subjects"]}
    animal = {vocabulary.index(t) for t in DOMAINS["animals"]["subjects"]}
    tech_rows = sum(any(token in technical for token in row[:4]) for row in sequences)
    animal_rows = sum(any(token in animal for token in row[:4]) for row in sequences)
    return {"examples": len(sequences), "unique_examples": unique,
            "duplicate_fraction": 1 - unique / len(sequences),
            "token_positions": int(array.size), "token_counts": counts.tolist(),
            "technical_examples": tech_rows, "animal_examples": animal_rows,
            "grammars": len({row.index(vocabulary.index("the")) for row in sequences})}


class ExposedBlock(nn.Module):
    """Post-norm block whose returned per-head weights belong to its actual forward pass."""

    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ffn = nn.Sequential(nn.Linear(config.width, config.ffn), nn.ReLU(),
                                 nn.Linear(config.ffn, config.width))

    def forward(self, hidden, blocked, capture=False):
        context, weights = self.attention(hidden, hidden, hidden, attn_mask=blocked,
                                          need_weights=capture, average_attn_weights=False)
        hidden = self.norm1(hidden + context)
        return self.norm2(hidden + self.ffn(hidden)), weights


class ExposedTransformer(nn.Module):
    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.width)
        self.position = nn.Embedding(6, config.width)
        self.blocks = nn.ModuleList([ExposedBlock(config) for _ in range(config.layers)])
        self.output = nn.Linear(config.width, vocabulary_size)

    def forward(self, tokens, causal=True, capture=False):
        if (tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[0] < 1
                or not 1 <= tokens.shape[1] <= 6 or (tokens < 0).any()
                or (tokens >= self.embedding.num_embeddings).any()):
            raise ValueError("tokens must be a nonempty long batch within vocabulary and context length 1..6")
        if not isinstance(causal, bool):
            raise ValueError("causal must be Boolean")
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        blocked = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1) if causal else None
        weights = []
        for block in self.blocks:
            hidden, attention = block(hidden, blocked, capture)
            if capture:
                weights.append(attention)
        return self.output(hidden), weights, hidden


def random_mlm_batch(tokens, mask_id, probability, generator):
    finite_positive(probability, "masking_probability")
    if probability > 1:
        raise ValueError("masking_probability must be <= 1")
    selected = torch.zeros_like(tokens, dtype=torch.bool)
    selected[:, 1:-1] = torch.rand(tokens.shape[0], tokens.shape[1] - 2,
                                  generator=generator, device=tokens.device) < probability
    missing = ~selected.any(1)
    fallback = torch.randint(1, tokens.shape[1] - 1, (int(missing.sum()),),
                             generator=generator, device=tokens.device)
    selected[missing, fallback] = True
    return mlm_batch(tokens, selected, mask_id)


def evaluate_lab(model, batch, causal):
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            inputs, targets = batch
            logits, _, _ = model(inputs, causal)
            valid = targets != -100
            nll = float(token_loss(logits, targets))
            result = {"nll": nll, "accuracy": float((logits.argmax(-1)[valid] == targets[valid]).float().mean()),
                      "targets": int(valid.sum())}
            if causal:
                result["perplexity"] = math.exp(nll)
            return result
    finally:
        model.train(was_training)


def prefix_evaluation(model, sequences, causal):
    """Every target is scored with only its preceding prefix, for both ablation arms."""
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            losses, correct = [], []
            for position in range(1, sequences.shape[1]):
                logits, _, _ = model(sequences[:, :position], causal)
                scores = logits[:, -1]
                target = sequences[:, position]
                losses.append(F.cross_entropy(scores, target, reduction="none"))
                correct.append(scores.argmax(-1) == target)
            nll = float(torch.cat(losses).mean())
            return {"nll": nll, "accuracy": float(torch.cat(correct).float().mean()),
                    "perplexity": math.exp(nll)}
    finally:
        model.train(was_training)


def greedy_generate(model, prefix, causal, max_length=6):
    with torch.inference_mode():
        tokens = prefix.clone()
        while tokens.shape[1] < max_length:
            logits, _, _ = model(tokens, causal)
            tokens = torch.cat([tokens, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return tokens


def train_visual_run(config, corpus, dataset="diverse", objective="clm", causal=True):
    if objective not in ("clm", "mlm"):
        raise ValueError("objective must be clm or mlm")
    torch.set_num_threads(1)
    device = torch.device(config.device)
    train = torch.tensor(corpus["datasets"][dataset], device=device)
    val = torch.tensor(corpus["validation"], device=device)
    vocabulary = corpus["vocabulary"]
    stoi = {t: i for i, t in enumerate(vocabulary)}
    probe_tokens = [["<bos>", "the", "cat", verb, "fish", "<eos>"] for verb in ("eats", "likes")]
    probes = torch.tensor([[stoi[t] for t in seq] for seq in probe_tokens], device=device)
    mask_id = stoi["<mask>"]
    mask_generator = torch.Generator(device=device).manual_seed(config.seed + 10)
    fixed_train = clm_batch(train) if objective == "clm" else random_mlm_batch(
        train, mask_id, config.masking_probability, mask_generator)
    fixed_val = clm_batch(val) if objective == "clm" else random_mlm_batch(
        val, mask_id, config.masking_probability, mask_generator)
    capture_steps = set(np.unique(np.linspace(0, config.steps, min(config.snapshots, config.steps + 1), dtype=int)))
    devices = [] if config.device == "cpu" else [torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        if config.device == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        generator = torch.Generator(device=device).manual_seed(config.seed + 1)
        model = ExposedTransformer(len(vocabulary), config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        snapshots, trace = [], []
        started = perf_counter()

        def synchronize():
            if config.device == "cuda":
                torch.cuda.synchronize()

        def capture(step):
            model.eval()
            train_metrics = evaluate_lab(model, fixed_train, causal if objective == "clm" else False)
            val_metrics = evaluate_lab(model, fixed_val, causal if objective == "clm" else False)
            with torch.inference_mode():
                if objective == "clm":
                    inputs = probes[:, :5]
                else:
                    selected = torch.zeros_like(probes, dtype=torch.bool)
                    selected[:, 4] = True
                    inputs, _ = mlm_batch(probes, selected, mask_id)
                logits, attention, hidden = model(inputs, causal if objective == "clm" else False, capture=True)
                # Position 3 predicts object at 4 in CLM; MLM reconstructs position 4.
                position = 3 if objective == "clm" else 4
                probabilities = torch.softmax(logits[:, position], -1)
                synchronize()
                record = {"step": step, "elapsed_seconds": perf_counter() - started,
                          "train": train_metrics, "validation": val_metrics,
                          "embedding": model.embedding.weight.detach().cpu().tolist(),
                          "probabilities": probabilities.cpu().tolist(),
                          "entropy": (-(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)).cpu().tolist(),
                          "reference_token_nll": (-probabilities[:, stoi["fish"]].clamp_min(1e-30).log()).cpu().tolist(),
                          "attention": attention[0].cpu().tolist(), "attention_layer": 0,
                          "probe_inputs": inputs.cpu().tolist(), "hidden": hidden.cpu().tolist()}
                if objective == "mlm" and step == config.steps:
                    reconstruction, _, _ = model(fixed_val[0][:4], False)
                    record["reconstruction_examples"] = {
                        "inputs": fixed_val[0][:4].cpu().tolist(), "targets": fixed_val[1][:4].cpu().tolist(),
                        "probabilities": torch.softmax(reconstruction, -1).cpu().tolist()}
            snapshots.append(record)
            model.train()

        capture(0)
        for step in range(1, config.steps + 1):
            indices = torch.randint(len(train), (config.batch_size,), generator=generator, device=device)
            sequences = train[indices]
            batch = clm_batch(sequences) if objective == "clm" else random_mlm_batch(
                sequences, mask_id, config.masking_probability, generator)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(batch[0], causal if objective == "clm" else False)
            loss = token_loss(logits, batch[1])
            loss.backward()
            norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient norm")
            optimizer.step()
            synchronize()
            trace.append({"step": step, "batch_nll": float(loss.detach()), "gradient_norm": float(norm),
                          "processed_tokens": step * config.batch_size * batch[0].shape[1],
                          "supervised_tokens_this_step": int((batch[1] != -100).sum()),
                          "elapsed_seconds": perf_counter() - started})
            if step in capture_steps:
                capture(step)
        model.eval()
        prefix = prefix_evaluation(model, val, causal) if objective == "clm" else None
        generated = greedy_generate(model, probes[:, :4], causal).cpu().tolist() if objective == "clm" else None
        parameters = sum(p.numel() for p in model.parameters())
        nonembedding = parameters - model.embedding.weight.numel() - model.position.weight.numel()
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    report = {"schema": 1, "executed_at_utc": datetime.now(timezone.utc).isoformat(),
              "hypothesis": "Self-supervision may fit synthetic templates without reliable transfer to held-out contexts.",
              "configuration": {**asdict(config), "dataset": dataset, "objective": objective,
                                "causal": causal, "dropout": 0, "weight_decay": 0.01,
                                "evaluation": "fixed corruption for MLM; final checkpoint; no validation selection"},
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "corpus": corpus, "dataset_statistics": corpus_statistics(corpus["datasets"][dataset], vocabulary),
              "parameters": parameters, "nonembedding_parameters": nonembedding,
              "processed_training_tokens": trace[-1]["processed_tokens"],
              "supervised_training_tokens": sum(r["supervised_tokens_this_step"] for r in trace),
              "duration_seconds": perf_counter() - started, "trace": trace, "snapshots": snapshots,
              "prefix_only_validation": prefix, "generated_from_probe_prefixes": generated,
              "probe_labels": ["Observed context: the cat eats", "Held-out context: the cat likes"],
              "probe_reference": "fish is one valid symbolic continuation, not a uniquely correct fact",
              "interpretation_status": "author review required",
              "limitations": "One seed; symbolic distribution shift; fixed-length six-token sequences; no real text or scaling-law fit."}
    return report, state


def experiment_specs(config):
    return {
        "clm": (config, "diverse", "clm", True),
        "mlm": (config, "diverse", "mlm", False),
        "large_restricted": (replace(config, width=2 * config.width, ffn=2 * config.ffn), "restricted", "clm", True),
        "longer": (replace(config, steps=2 * config.steps), "diverse", "clm", True),
        "more_unique": (config, "more_unique", "clm", True),
        "duplicated": (config, "duplicated", "clm", True),
        "single_grammar": (config, "single_grammar", "clm", True),
        "conflicting": (config, "conflicting", "clm", True),
        "domain_imbalanced": (config, "domain_imbalanced", "clm", True),
        "unmasked": (config, "diverse", "clm", False),
    }


def cached_experiments(config, folder, names):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    corpus = make_lab_corpus(config.seed)
    reports = {}
    source_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("example.py").read_bytes()).hexdigest()
    for name in names:
        run_config, dataset, objective, causal = experiment_specs(config)[name]
        signature = hashlib.sha256(json.dumps([asdict(run_config), dataset, objective, causal, corpus,
                                               source_hash, torch.__version__], sort_keys=True).encode()).hexdigest()
        path = folder / f"{name}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("signature") != signature:
                raise ValueError(f"Cached configuration/code mismatch: {path}; choose a new output directory")
            print(f"  reuse {name}: {path}", flush=True)
        else:
            print(f"  train {name}: {run_config.steps} steps, width {run_config.width}, {dataset}", flush=True)
            report, state = train_visual_run(run_config, corpus, dataset, objective, causal)
            report["signature"] = signature
            with path.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, allow_nan=False)
            torch.save(state, folder / f"{name}-final.pt")
            print(f"    final train={report['snapshots'][-1]['train']['nll']:.4f} val={report['snapshots'][-1]['validation']['nll']:.4f}", flush=True)
        reports[name] = report
    return reports


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] < 2 or not np.isfinite(logits).all():
        raise ValueError("logits must be finite with at least two vocabulary entries")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)


def cross_entropy_logits(logits, target):
    values = np.asarray(logits, dtype=np.float64)
    probabilities = softmax(values)
    if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < values.shape[-1]:
        raise ValueError("target must be an integer vocabulary ID")
    shifted = values - values.max(axis=-1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(-1)) - shifted[..., target]
    one_hot = np.eye(values.shape[-1])[target]
    return loss, probabilities - one_hot


def logit_descent(steps=30, learning_rate=0.2):
    """Actual two-coordinate descent; third logit remains fixed at zero."""
    finite_positive(learning_rate, "learning_rate")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    logits = np.array([-2., 2., 0.])
    records = []
    for step in range(steps + 1):
        loss, gradient = cross_entropy_logits(logits, 0)
        records.append({"step": step, "logits": logits.tolist(), "probabilities": softmax(logits).tolist(),
                        "loss": float(loss), "gradient": gradient.tolist()})
        logits[:2] -= learning_rate * gradient[:2]
    return records


@dataclass(frozen=True)
class Scaling:
    E: float = 1.0
    A: float = 4.0
    B: float = 4.0
    alpha: float = 0.34
    beta: float = 0.28

    def __post_init__(self):
        finite_positive(self.E, "E")
        for name in ("A", "B", "alpha", "beta"):
            finite_positive(getattr(self, name), name)


def scaling_loss(parameters, tokens, scaling=Scaling()):
    parameters, tokens = np.asarray(parameters), np.asarray(tokens)
    if (not np.isfinite(parameters).all() or not np.isfinite(tokens).all()
            or (parameters <= 0).any() or (tokens <= 0).any()):
        raise ValueError("parameters and tokens must be finite and positive")
    # Coefficients are explicitly illustrative; N and D are expressed in billions here.
    return scaling.E + scaling.A / (parameters / 1e9) ** scaling.alpha + scaling.B / (tokens / 1e9) ** scaling.beta


def optimal_allocation(compute, scaling=Scaling()):
    finite_positive(compute, "compute")
    # C = 6 * (1e9*n) * (1e9*d); differentiate the assumed normalized loss.
    budget = compute / 6e18
    n = ((scaling.alpha * scaling.A / (scaling.beta * scaling.B))
         * budget ** scaling.beta) ** (1 / (scaling.alpha + scaling.beta))
    parameters = 1e9 * n
    tokens = compute / (6 * parameters)
    return {"parameters": parameters, "tokens": tokens,
            "loss": float(scaling_loss(parameters, tokens, scaling))}


def compute_budget(parameters, tokens, accelerators, peak_tflops, utilization, hourly_cost):
    for name, value in (("parameters", parameters), ("tokens", tokens), ("peak_tflops", peak_tflops),
                        ("utilization", utilization)):
        finite_positive(value, name)
    if isinstance(accelerators, bool) or not isinstance(accelerators, int) or accelerators < 1:
        raise ValueError("accelerators must be a positive integer")
    if utilization > 1 or not np.isfinite(hourly_cost) or hourly_cost < 0:
        raise ValueError("utilization must be <= 1 and hourly cost must be finite and nonnegative")
    flops = 6 * parameters * tokens
    throughput = accelerators * peak_tflops * 1e12 * utilization
    seconds = flops / throughput
    hours = seconds / 3600 * accelerators
    return {"training_flops": flops, "sustained_flops_per_second": throughput,
            "seconds": seconds, "accelerator_hours": hours, "accelerator_cost": hours * hourly_cost,
            "tokens_per_parameter": tokens / parameters}


def pooled_pca(embeddings):
    matrices = np.asarray(embeddings, dtype=float)
    if matrices.ndim != 3 or min(matrices.shape) < 1 or matrices.shape[-1] < 3 or not np.isfinite(matrices).all():
        raise ValueError("finite checkpoint/token/feature embedding matrices with feature width >= 3 required")
    from sklearn.decomposition import PCA
    projection = PCA(n_components=3, svd_solver="full")
    coords = projection.fit_transform(matrices.reshape(-1, matrices.shape[-1])).reshape(*matrices.shape[:2], 3)
    return coords, projection.explained_variance_ratio_


def cosine_matrix(embedding):
    embedding = np.asarray(embedding, dtype=float)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    if not np.isfinite(embedding).all() or (norms == 0).any():
        raise ValueError("finite nonzero embedding rows required")
    normalized = embedding / norms
    return normalized @ normalized.T
