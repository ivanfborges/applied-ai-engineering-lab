"""Numerical instrumentation and bounded experiments for the Day 67 visual lab.

The model and data contracts come from example.py. Importing this module never
trains, writes checkpoints or creates figures. All attention capture is in eval mode.
"""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.nn import functional as F

from example import Config, Seq2SeqTransformer, causal_mask, make_split, reversal_targets, sequence_metrics
from from_scratch import Attention, sinusoidal_positions

COLORS = {"encoder": "#2563eb", "decoder": "#d97706", "attention": "#7c3aed",
          "ffn": "#15803d", "residual": "#db2777", "norm": "#0891b2", "muted": "#64748b"}
SCHEMA = 1


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


def masked_softmax(scores, blocked=None):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim < 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite array with query/key axes")
    if blocked is not None:
        blocked = np.asarray(blocked)
        if blocked.dtype != bool or blocked.shape != scores.shape[-2:]:
            raise ValueError("blocked must be a boolean query by key matrix")
        if blocked.all(axis=-1).any():
            raise ValueError("fully masked rows have no defined probability distribution")
        scores = np.where(blocked, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)


@torch.no_grad()
def capture_attention(module, query, memory, blocked=None):
    """Expose actual Q/K/V, individual heads, concat and output; assert MHA agreement."""
    if module.training or query.device.type != "cpu" or memory.device.type != "cpu":
        raise ValueError("attention capture requires eval-mode CPU tensors")
    if not module._qkv_same_embed_dim or module.bias_k is not None or module.add_zero_attn:
        raise ValueError("capture supports the study's standard equal-width MHA only")
    width, heads = module.embed_dim, module.num_heads
    weights = module.in_proj_weight.chunk(3)
    biases = module.in_proj_bias.chunk(3) if module.in_proj_bias is not None else [None] * 3
    q, k, v = [F.linear(x, w, b).reshape(len(x), x.shape[1], heads, width // heads).transpose(1, 2)
               for x, w, b in zip((query, memory, memory), weights, biases)]
    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(width // heads)
    if blocked is not None:
        # Also explicitly checks the all-masked edge case before torch softmax.
        masked_softmax(array(scaled), array(blocked))
    logits = scaled.masked_fill(blocked, -torch.inf) if blocked is not None else scaled
    probability = torch.softmax(logits, dim=-1)
    head_output = probability @ v
    concatenated = head_output.transpose(1, 2).reshape(len(query), query.shape[1], width)
    output = module.out_proj(concatenated)
    reference, per_head = module(query, memory, memory, attn_mask=blocked,
                                 need_weights=True, average_attn_weights=False)
    torch.testing.assert_close(output, reference, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(probability, per_head, atol=3e-6, rtol=3e-5)
    snapshot = {name: array(value[0]) for name, value in
                (("q", q), ("k", k), ("v", v), ("raw", raw), ("scores", scaled),
                 ("probability", probability), ("head_output", head_output))}
    snapshot.update(input=array(query[0]), memory=array(memory[0]),
                    concat=array(concatenated[0]), output=array(output[0]),
                    blocked=array(blocked) if blocked is not None else np.zeros(scaled.shape[-2:], dtype=bool))
    return output, snapshot


def embedded(model, tokens, positions=True):
    full = model.embed(tokens)
    return full if positions else model.embedding(tokens) * math.sqrt(model.config.d_model)


@torch.no_grad()
def capture_model(model, source, decoder_input, *, causal=True, positions=True):
    """Trace registered layer modules, including final stack norms, without hooks/private forwards."""
    if model.training:
        raise ValueError("model.eval() is required for stable activation capture")
    src = embedded(model, source, positions)
    tgt = embedded(model, decoder_input, positions)
    stages = [{"name": "Scaled token embeddings", "values": array(model.embedding(source)[0]) * math.sqrt(model.config.d_model)},
              {"name": "Embeddings + learned positions" if positions else "Positions disabled", "values": array(src[0])}]
    attentions, residuals = [], []
    x = src
    for index, layer in enumerate(model.transformer.encoder.layers):
        before = x
        attention_input = layer.norm1(x) if layer.norm_first else x
        update, snapshot = capture_attention(layer.self_attn, attention_input, attention_input)
        snapshot.update(mechanism="encoder", layer=index)
        attentions.append(snapshot)
        residuals.append({"input": array(before[0]), "update": array(update[0]), "sum": array((before + update)[0])})
        stages.append({"name": f"Encoder {index + 1}: attention branch", "values": array(update[0])})
        x = x + update if layer.norm_first else layer.norm1(x + update)
        stages.append({"name": f"Encoder {index + 1}: first residual + norm", "values": array(x[0])})
        ffn_input = layer.norm2(x) if layer.norm_first else x
        expanded = layer.linear1(ffn_input)
        activated = layer.activation(expanded)
        update = layer.linear2(activated)
        stages.append({"name": f"Encoder {index + 1}: FFN branch", "values": array(update[0])})
        x = x + update if layer.norm_first else layer.norm2(x + update)
        stages.append({"name": f"Encoder {index + 1}: complete layer", "values": array(x[0])})
    memory = model.transformer.encoder.norm(x) if model.transformer.encoder.norm is not None else x
    stages.append({"name": "Final encoder memory", "values": array(memory[0])})
    y = tgt
    mask = causal_mask(tgt.shape[1]) if causal else None
    for index, layer in enumerate(model.transformer.decoder.layers):
        value = layer.norm1(y) if layer.norm_first else y
        update, snapshot = capture_attention(layer.self_attn, value, value, mask)
        snapshot.update(mechanism="decoder", layer=index)
        attentions.append(snapshot)
        y = y + update if layer.norm_first else layer.norm1(y + update)
        value = layer.norm2(y) if layer.norm_first else y
        update, snapshot = capture_attention(layer.multihead_attn, value, memory)
        snapshot.update(mechanism="cross", layer=index)
        attentions.append(snapshot)
        y = y + update if layer.norm_first else layer.norm2(y + update)
        value = layer.norm3(y) if layer.norm_first else y
        update = layer.linear2(layer.activation(layer.linear1(value)))
        y = y + update if layer.norm_first else layer.norm3(y + update)
    if model.transformer.decoder.norm is not None:
        y = model.transformer.decoder.norm(y)
    logits = model.output(y)
    reference = model.output(model.transformer(src, tgt, tgt_mask=mask))
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=5e-5)
    if positions:
        torch.testing.assert_close(logits, model(source, decoder_input, causal=causal), atol=1e-5, rtol=5e-5)
    return {"stages": stages, "attention": attentions, "residuals": residuals,
            "logits": array(logits[0]), "probability": array(logits.softmax(-1)[0]),
            "source": source[0].tolist(), "decoder_input": decoder_input[0].tolist(),
            "positions": positions, "causal": causal,
            "norm": "pre" if model.transformer.encoder.layers[0].norm_first else "post",
            "instrumentation_max_error": float((logits - reference).abs().max())}


def shared_pca(stages):
    values = [np.asarray(stage["values"], dtype=float) for stage in stages]
    if not values or len({value.shape for value in values}) != 1 or min(values[0].shape) < 3:
        raise ValueError("PCA needs equally shaped stages with >=3 rows/features")
    fitted = PCA(n_components=3, svd_solver="full").fit(np.concatenate(values))
    projected = np.stack([fitted.transform(value) for value in values])
    return {"coordinates": projected, "variance_ratio": fitted.explained_variance_ratio_,
            "mean": fitted.mean_, "components": fitted.components_,
            "names": [stage["name"] for stage in stages]}


def equivariance_demo(seed=67):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(5, 8))
    permutation = np.array([2, 0, 4, 1, 3])
    attention = Attention(8, rng)
    y, _ = attention(x, x)
    reordered, _ = attention(x[permutation], x[permutation])
    positional, _ = attention(x + sinusoidal_positions(5, 8), x + sinusoidal_positions(5, 8))
    reordered_pos, _ = attention(x[permutation] + sinusoidal_positions(5, 8), x[permutation] + sinusoidal_positions(5, 8))
    return {"without_positions_error": float(abs(reordered - y[permutation]).max()),
            "with_positions_change": float(abs(reordered_pos - positional[permutation]).max()),
            "embeddings": x, "positioned": x + sinusoidal_positions(5, 8), "permutation": permutation}


def memory_bytes(source_length, target_length, heads=4, batch=1, bytes_per_value=4):
    for value in (source_length, target_length, heads, batch, bytes_per_value):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("memory dimensions must be positive integers")
    return {"encoder": batch * heads * source_length ** 2 * bytes_per_value,
            "decoder": batch * heads * target_length ** 2 * bytes_per_value,
            "cross": batch * heads * source_length * target_length * bytes_per_value}


@torch.no_grad()
def generate_prefix(model, source, causal=True):
    """Decode only available predictions; the unmasked arm also has no future labels here."""
    if model.training:
        raise ValueError("generation requires evaluation mode")
    memory = model.transformer.encoder(model.embed(source))
    prefix = torch.zeros((len(source), 1), dtype=torch.long)
    distributions = []
    for _ in range(model.config.sequence_length + 1):
        hidden = model.transformer.decoder(model.embed(prefix), memory,
                    tgt_mask=causal_mask(prefix.shape[1]) if causal else None)
        probability = model.output(hidden[:, -1]).softmax(-1)
        distributions.append(array(probability[0]))
        prefix = torch.cat((prefix, probability.argmax(-1, keepdim=True)), dim=1)
    return prefix[:, 1:], np.stack(distributions)


@torch.no_grad()
def evaluate(model, train, test, causal):
    model.eval()
    train_input, train_target = reversal_targets(train)
    decoder, target = reversal_targets(test)
    train_loss = F.cross_entropy(model(train, train_input, causal=causal).flatten(0, 1), train_target.flatten())
    logits = model(test, decoder, causal=causal)
    loss = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    generated, probabilities = generate_prefix(model, test, causal)
    return {"train_loss": train_loss.item(), "teacher_forced_loss": loss.item(),
            "teacher_forced": sequence_metrics(logits.argmax(-1), target),
            "autoregressive": sequence_metrics(generated, target),
            "source_examples": test[:4].tolist(), "target_examples": target[:4].tolist(),
            "generated_examples": generated[:4].tolist(), "first_probabilities": probabilities.tolist()}


def cache_key(config):
    source = b"".join(Path(__file__).with_name(name).read_bytes() for name in ("example.py", "transformer_lab.py", "from_scratch.py"))
    payload = json.dumps({"config": asdict(config), "torch": str(torch.__version__), "schema": SCHEMA}, sort_keys=True).encode()
    return hashlib.sha256(source + payload).hexdigest()[:20]


def train_comparison(config, cache_dir, skip_training=False):
    """Matched two-arm experiment. Snapshots are descriptive; no checkpoint is selected."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cache_dir = Path(cache_dir)
    path = cache_dir / f"comparison_{cache_key(config)}.pt"
    if path.exists():
        result = torch.load(path, map_location="cpu", weights_only=True)
        if result["config"] != asdict(config) or result["schema"] != SCHEMA:
            raise ValueError("checkpoint metadata does not match the requested experiment")
        return result, path, True
    if skip_training:
        return None, path, False
    train, test = make_split(config)
    decoder, target = reversal_targets(train)
    checkpoints = sorted({0, max(1, config.steps // 10), max(1, config.steps // 2), config.steps})
    result = {"schema": SCHEMA, "config": asdict(config), "history": {}, "states": {},
              "hypothesis": "Missing causal masking can favor teacher-forced scores over free-running generation.",
              "limitations": "Single seed, fixed length, no model selection; held-out curves are descriptive diagnostics, not an untouched final test after inspection.",
              "interpretation": "author review required", "checkpoints": checkpoints}
    for name, causal in (("masked", True), ("unmasked", False)):
        torch.manual_seed(config.seed)
        model = Seq2SeqTransformer(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        generator = torch.Generator().manual_seed(config.seed + 1)
        history, states = [], []
        for step in range(config.steps + 1):
            if step in checkpoints:
                measured = evaluate(model, train, test, causal)
                history.append({"step": step, **measured})
                states.append({key: value.detach().clone() for key, value in model.state_dict().items()})
            if step == config.steps:
                break
            model.train()
            indices = torch.randint(len(train), (config.batch_size,), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train[indices], decoder[indices], causal=causal).flatten(0, 1), target[indices].flatten())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        result["history"][name], result["states"][name] = history, states
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(result, handle)
    return result, path, False


def model_at(comparison, arm="masked", checkpoint=-1):
    config = Config(**comparison["config"])
    # Avoid changing the caller's random stream when reconstructing a snapshot.
    with torch.random.fork_rng():
        model = Seq2SeqTransformer(config)
    model.load_state_dict(comparison["states"][arm][checkpoint])
    return model.eval()


def default_trace(config, comparison=None):
    torch.set_num_threads(1)
    if comparison is None:
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            model = Seq2SeqTransformer(config).eval()
    else:
        model = model_at(comparison)
    _, test = make_split(config)
    source = test[:1]
    decoder, _ = reversal_targets(source)
    trace = capture_model(model, source, decoder)
    trace["classification"] = "Trained model observation" if comparison is not None else "Untrained numerical demonstration"
    return model, trace


def playground_cases(config, comparison=None):
    """Small real, precomputed forwards: changed controls select these, not fictitious retraining."""
    cases = []
    snapshots = [0, len(comparison["checkpoints"]) - 1] if comparison else [None]
    for checkpoint in snapshots:
        model = model_at(comparison, checkpoint=checkpoint) if comparison else default_trace(config)[0]
        _, test = make_split(config)
        for length in sorted({3, config.sequence_length}):
            if length > config.sequence_length:
                continue
            source = test[:1, :length]
            decoder, _ = reversal_targets(source)
            for positions in (True, False):
                for causal in (True, False):
                    for norm in ("post", "pre"):
                        for layer in list(model.transformer.encoder.layers) + list(model.transformer.decoder.layers):
                            layer.norm_first = norm == "pre"
                        trace = capture_model(model, source, decoder, causal=causal, positions=positions)
                        # Compact state: preserve head/feature values, not duplicated Q/K/V inputs.
                        trace["attention"] = [{key: value for key, value in attention.items()
                                               if key in ("mechanism", "layer", "scores", "probability", "output", "head_output")}
                                              for attention in trace["attention"]]
                        trace.pop("residuals")
                        trace["checkpoint"] = comparison["checkpoints"][checkpoint] if comparison else 0
                        trace["classification"] = "Untrained numerical demonstration" if not checkpoint else "Trained model observation"
                        trace["counterfactual"] = norm == "pre" or not causal or not positions or length != config.sequence_length
                        cases.append(trace)
    return cases
