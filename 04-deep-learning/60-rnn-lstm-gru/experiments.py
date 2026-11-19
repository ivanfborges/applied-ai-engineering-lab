"""Cached, independent visual protocol. Does not rewrite the analytical study."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import numpy as np
import torch

from example import SequenceClassifier, evaluate, fit_model, make_data
from from_scratch import integer, valid_seed
from visual_core import input_gradients, pca_states


ROOT = Path(__file__).resolve().parent


def parameter_count(kind, hidden):
    if kind not in ("rnn", "lstm", "gru"):
        raise ValueError("unknown cell")
    hidden = integer(hidden, "hidden")
    return {"rnn": 1, "lstm": 4, "gru": 3}[kind] * hidden * (hidden + 4) + hidden + 1


def matched_hidden(kind, target):
    target = integer(target, "target")
    return min(range(3, 257), key=lambda h: abs(parameter_count(kind, h) - target))


def configuration(full=False, matched=False):
    return dict(protocol="visual-v1", seeds=[160, 161, 162] if full else [160],
                lengths=[10, 20, 50], epochs=20 if full else 12, hidden=16 if full else 12,
                train_size=1024 if full else 512, validation_size=256 if full else 128,
                test_size=512 if full else 256, batch_size=64, learning_rate=0.005, clip_norm=1.0,
                matched=bool(matched), probe_size=32, trajectory_per_class=2)


def cache_signature(config):
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode())
    for name in ("example.py", "from_scratch.py", "visual_core.py", "experiments.py"):
        digest.update((ROOT / name).read_bytes())
    digest.update(f"{torch.__version__}|{np.__version__}|{platform.python_version()}|{platform.platform()}".encode())
    return digest.hexdigest()[:20]


def run_comparison(config, progress=print):
    required = set(configuration())
    if set(config) != required or config["protocol"] != "visual-v1":
        raise ValueError("configuration must follow the visual-v1 schema")
    seeds = [valid_seed(s) for s in config["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds) or max(seeds) > (2**32 - 5) // 10:
        raise ValueError("provide unique seeds leaving room for derived streams")
    lengths = [integer(t, "length", 2) for t in config["lengths"]]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("provide unique nonempty lengths")
    rows, weights = [], {}
    target = parameter_count("lstm", config["hidden"])
    for seed in seeds:
        base = 10 * seed
        for length in lengths:
            train = make_data(config["train_size"], length, base)
            validation = make_data(config["validation_size"], length, base + 1)
            test = make_data(config["test_size"], length, base + 2)
            # A separate fixed probe is used for gradients and representation inspection.
            probe_x, probe_y = make_data(max(64, config["probe_size"]), length, base + 5)
            for kind in ("rnn", "lstm", "gru"):
                hidden = matched_hidden(kind, target) if config["matched"] else config["hidden"]
                progress(f"Training seed={seed} T={length} {kind.upper()} H={hidden}")
                start = perf_counter()
                model, training = fit_model(kind, train, validation, hidden=hidden, seed=base + 3,
                                            shuffle_seed=base + 4, epochs=config["epochs"],
                                            batch_size=config["batch_size"], learning_rate=config["learning_rate"],
                                            clip_norm=config["clip_norm"])
                elapsed = perf_counter() - start
                key = f"s{seed}_t{length}_{kind}"
                for name, value in model.state_dict().items():
                    weights[f"{key}/{name}"] = value.detach().numpy().copy()
                indices = []
                for label in (0, 1):
                    found = torch.where(probe_y == label)[0][:config["trajectory_per_class"]]
                    if len(found) != config["trajectory_per_class"]:
                        raise ValueError("probe does not contain enough examples of both classes")
                    indices.extend(found.tolist())
                with torch.no_grad():
                    states, _ = model.encode(probe_x[indices])
                    states = torch.cat([torch.zeros(len(indices), 1, hidden), states], dim=1).numpy()
                projected, ratios = pca_states(states)
                n_probe = min(config["probe_size"], len(probe_y))
                baseline = int(train[1].mean() >= 0.5)
                rows.append(dict(key=key, seed=seed, length=length, cell=kind, hidden=hidden,
                                 parameters=parameter_count(kind, hidden), training_seconds=elapsed,
                                 **training, test=evaluate(model, *test),
                                 majority_accuracy=(test[1] == baseline).float().mean().item(),
                                 first_sign_accuracy=((test[0][:, 0, 0] > 0) == test[1].bool()).float().mean().item(),
                                 last_sign_accuracy=((test[0][:, -1, 0] > 0) == test[1].bool()).float().mean().item(),
                                 input_gradient_norm=input_gradients(model, probe_x[:n_probe], probe_y[:n_probe]).tolist(),
                                 projection=projected.tolist(), explained_variance=ratios.tolist(),
                                 trajectory_labels=probe_y[indices].int().tolist(),
                                 trajectory_logits=model(probe_x[indices]).detach().tolist(),
                                 trajectory_probe_indices=indices))
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
                  hypothesis="Gating may aid delayed recall, but optimization and budget can reverse or obscure the expected comparison.",
                  environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                                   platform=platform.platform(), processor=platform.processor(),
                                   threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled()),
                  selection="Earliest minimum validation BCE per predefined cell/length/seed; no test-based selection",
                  timing="Wall-clock fit_model including initialization, validation and restoration; excludes probes and rendering; not inference latency",
                  gradient_definition="Mean over probe examples of L2 norm across two features of d BCE_i / d x_it; summed loss gives individual-example scaling",
                  pca_definition="Separate NumPy SVD PCA per checkpoint over four probe trajectories including zero initial state; axes are not aligned across models",
                  interpretation_candidate="Assess same-generator recall against controls; single-seed differences and timing depend on configuration/hardware. PCA separation and sensitivities are not explanations or causal evidence.",
                  interpretation_status="author review required",
                  limitations="Synthetic IID data; equal hidden size is unequal capacity unless matched mode; no validation search, real time series, temporal extrapolation or serving benchmark. Repeated seeds summarize training variation, not population uncertainty.",
                  results=rows)
    return report, weights


def load_or_run(cache_dir, config, progress=print):
    """Cache identity binds configuration, numerical source and environment.

    Incomplete or altered caches fail visibly. New successful runs are moved
    into place only after both a JSON record and NumPy checkpoints are written.
    """
    cache_dir = Path(cache_dir)
    key = cache_signature(config)
    folder = cache_dir / key
    if folder.exists():
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        checkpoint = folder / "weights.npz"
        if report.get("cache_key") != key or report["checkpoint_sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError("cache identity/checksum mismatch; preserve it and choose another cache directory")
        progress(f"Cache hit: {folder}; no training")
        return report, folder
    report, weights = run_comparison(config, progress)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pending_", dir=cache_dir) as temporary:
        temporary = Path(temporary)
        checkpoint = temporary / "weights.npz"
        np.savez_compressed(checkpoint, **weights)
        report.update(cache_key=key, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (temporary / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        folder.mkdir()
        checkpoint.rename(folder / "weights.npz")
        (temporary / "report.json").rename(folder / "report.json")
    return report, folder


def load_model(folder, row):
    model = SequenceClassifier(row["cell"], row["hidden"])
    with np.load(Path(folder) / "weights.npz", allow_pickle=False) as archive:
        state = {name: torch.from_numpy(archive[f"{row['key']}/{name}"].copy()) for name in model.state_dict()}
    model.load_state_dict(state)
    return model.eval()
