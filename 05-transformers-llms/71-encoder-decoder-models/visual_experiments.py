"""Computed synthetic experiments shared by Day 71 visual exports and tests."""

from copy import deepcopy
import re

import numpy as np
import torch
from torch import nn

from from_scratch import attention, shift_right, visibility
from example import causal_block_mask

SEED = 271
SOURCE = ["The", "bank", "approved", "the", "customer's", "loan."]
TARGET = ["O", "banco", "aprovou", "o", "empréstimo", "do", "cliente."]
COLORS = {"encoder": "#27834c", "decoder": "#2473bd", "cross": "#8154af",
          "active": "#dc791c", "masked": "#a4aab2", "ink": "#223047"}


def softmax(logits, temperature=1.0):
    values = np.asarray(logits, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("logits must be nonempty and finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = (values - values.max(axis=-1, keepdims=True)) / temperature
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def positions(length, width=16, offset=0):
    if length < 1 or width < 2 or width % 2 or offset < 0:
        raise ValueError("positive length, nonnegative offset and positive even width required")
    angles = np.arange(offset, offset + length)[:, None] / 10000 ** (np.arange(0, width, 2) / width)
    output = np.empty((length, width))
    output[:, 0::2], output[:, 1::2] = np.sin(angles), np.cos(angles)
    return output


def block_trace(layer, inputs, mask=None, memory=None):
    """Actual Post-LN operations using the supplied PyTorch layer's parameters."""
    trace = {"Embeddings + positions": inputs}
    if memory is None:
        attn = layer.self_attn(inputs, inputs, inputs, attn_mask=mask, need_weights=False)[0]
    else:
        attn = layer.self_attn(inputs, inputs, inputs, attn_mask=mask, need_weights=False)[0]
    trace["Self-attention"] = attn
    trace["Residual 1"] = inputs + attn
    x = layer.norm1(trace["Residual 1"])
    trace["Normalization 1"] = x
    if memory is not None:
        cross = layer.multihead_attn(x, memory, memory, need_weights=False)[0]
        trace["Cross-attention"] = cross
        trace["Residual cross"] = x + cross
        x = layer.norm2(x + cross)
        trace["Normalization cross"] = x
    ffn = layer.linear2(layer.activation(layer.linear1(x)))
    trace["Feed-forward"] = ffn
    trace["Residual FFN"] = x + ffn
    output = (layer.norm3 if memory is not None else layer.norm2)(x + ffn)
    trace["Contextual states"] = output
    return {key: value[0].detach().numpy().copy() for key, value in trace.items()}


def model_experiment(seed=SEED, generation_steps=4):
    """Paired full blocks, interventions and untrained greedy continuation."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    vocab = list(dict.fromkeys(["<start>", *SOURCE, *TARGET, "<eos>"]))
    table = rng.normal(size=(len(vocab), 16))
    src_embedding = table[[vocab.index(token) for token in SOURCE]]
    tgt_embedding = table[[vocab.index(token) for token in ["<start>", *TARGET[:-1]]]]
    src = torch.from_numpy(src_embedding + positions(len(SOURCE)))[None]
    tgt = torch.from_numpy(tgt_embedding + positions(len(TARGET)))[None]
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(seed)
        enc = nn.TransformerEncoderLayer(16, 4, 48, dropout=0, batch_first=True, dtype=torch.float64).eval()
        causal = deepcopy(enc)
        dec = nn.TransformerDecoderLayer(16, 4, 48, dropout=0, batch_first=True, dtype=torch.float64).eval()
        memory = enc(src)
        outputs = {"encoder": memory, "decoder": causal(src, src_mask=causal_block_mask(len(SOURCE))),
                   "cross": dec(tgt, memory, tgt_mask=causal_block_mask(len(TARGET)))}
        modified_src = src.clone()
        modified_src[0, -1, 0] += 10
        modified_tgt = tgt.clone()
        modified_tgt[0, -1, 0] += 10
        modified = {"encoder": enc(modified_src),
                    "decoder": causal(modified_src, src_mask=causal_block_mask(len(SOURCE))),
                    "cross": dec(tgt, enc(modified_src), tgt_mask=causal_block_mask(len(TARGET)))}
        target_modified = dec(modified_tgt, memory, tgt_mask=causal_block_mask(len(TARGET)))
        traces = {"encoder": block_trace(enc, src),
                  "decoder": block_trace(causal, src, causal_block_mask(len(SOURCE))),
                  "cross": block_trace(dec, tgt, causal_block_mask(len(TARGET)), memory)}
        trace_errors = {key: float(np.max(np.abs(traces[key]["Contextual states"] - value[0].numpy())))
                        for key, value in outputs.items()}
        head = rng.normal(scale=0.25, size=(16, len(vocab)))
        generated = {}
        for family in ("decoder", "cross"):
            ids = [vocab.index(token) for token in SOURCE] if family == "decoder" else [0]
            steps = []
            for _ in range(generation_steps):
                x = torch.from_numpy(table[ids] + positions(len(ids)))[None]
                hidden = causal(x, src_mask=causal_block_mask(len(ids))) if family == "decoder" else dec(
                    x, memory, tgt_mask=causal_block_mask(len(ids)))
                logits = hidden[0, -1].numpy() @ head
                probs = softmax(logits)
                chosen = int(probs.argmax())
                steps.append({"context": [vocab[i] for i in ids], "logits": logits,
                              "probabilities": probs, "chosen": chosen, "token": vocab[chosen]})
                ids.append(chosen)
            generated[family] = steps
    result = {"seed": seed, "source": SOURCE, "target": TARGET, "vocab": vocab,
              "embeddings": {"encoder": src_embedding, "decoder": src_embedding, "cross": tgt_embedding},
              "inputs": {"encoder": src[0].numpy(), "decoder": src[0].numpy(), "cross": tgt[0].numpy()},
              "baseline": {key: value[0].numpy() for key, value in outputs.items()},
              "perturbed": {key: value[0].numpy() for key, value in modified.items()},
              "target_perturbed": target_modified[0].numpy(), "traces": traces,
              "trace_errors": trace_errors, "generated": generated}
    result["deltas"] = {key: np.linalg.norm(result["perturbed"][key] - value, axis=1)
                        for key, value in result["baseline"].items()}
    result["target_deltas"] = np.linalg.norm(result["target_perturbed"] - result["baseline"]["cross"], axis=1)
    return result


def attention_experiment(model=None, seed=SEED):
    model = model if model is not None else model_experiment(seed)
    rng = np.random.default_rng(seed + 1)
    wq, wk, wv = (rng.normal(scale=0.25, size=(16, 8)) for _ in range(3))
    source = model["inputs"]["encoder"]
    data = {}
    for family in ("encoder", "decoder", "cross"):
        query_input = model["inputs"]["cross"] if family == "cross" else source
        memory = model["baseline"]["encoder"] if family == "cross" else source
        q, k, v = query_input @ wq, memory @ wk, memory @ wv
        mask = visibility(len(q), len(k), causal=family == "decoder")
        output, weights = attention(q, k, v, mask)
        logits = q @ k.T / np.sqrt(q.shape[1])
        data[family] = {"q": q, "k": k, "v": v, "logits": logits, "allowed": mask,
                        "masked_logits": np.where(mask, logits, -np.inf), "weights": weights,
                        "output": output, "query_labels": ["<start>", *TARGET[:-1]] if family == "cross" else SOURCE,
                        "key_labels": SOURCE}
    return data


def shared_pca(arrays):
    matrices = [np.asarray(matrix, dtype=float) for matrix in arrays]
    if not matrices or any(x.ndim != 2 or x.shape[1] < 3 or not np.isfinite(x).all() for x in matrices):
        raise ValueError("finite matrices with at least three features required")
    combined = np.concatenate(matrices)
    mean = combined.mean(axis=0)
    _, singular, vt = np.linalg.svd(combined - mean, full_matrices=False)
    basis = vt[:3].T
    # Fix SVD sign ambiguity for reproducible display coordinates.
    for column in range(3):
        if basis[np.abs(basis[:, column]).argmax(), column] < 0:
            basis[:, column] *= -1
    total = np.sum(singular**2)
    return {"coordinates": [(matrix - mean) @ basis for matrix in matrices], "mean": mean,
            "basis": basis, "explained_ratio": singular[:3]**2 / total if total else np.zeros(3)}


def corrupt_spans(tokens, spans):
    if not spans or any(a < 0 or b <= a or b > len(tokens) for a, b in spans):
        raise ValueError("spans must be nonempty valid half-open ranges")
    if any(spans[i][1] >= spans[i+1][0] for i in range(len(spans)-1)):
        raise ValueError("spans must be ordered, disjoint and nonadjacent")
    source, target, cursor = [], [], 0
    for i, (start, stop) in enumerate(spans):
        sentinel = f"<s{i}>"
        source.extend([*tokens[cursor:start], sentinel])
        target.extend([sentinel, *tokens[start:stop]])
        cursor = stop
    source.extend(tokens[cursor:])
    target.append(f"<s{len(spans)}>")
    return source, target


def objective_experiment(seed=SEED):
    rng = np.random.default_rng(seed + 2)
    vocab = ["<start>", "<mask>", "<eos>", "<s0>", "<s1>", "<s2>", *SOURCE, "river"]
    table = rng.normal(size=(len(vocab), 16))
    head = rng.normal(scale=0.2, size=(16, len(vocab)))

    def predict(input_tokens, labels, causal=False, source=None):
        ids = [vocab.index(t) for t in input_tokens]
        x = table[ids] + positions(len(ids))
        context, _ = attention(x, x, x, visibility(len(x), len(x), causal=causal))
        if source is not None:
            memory = table[[vocab.index(t) for t in source]] + positions(len(source))
            conditioned, _ = attention(context, memory, memory, visibility(len(x), len(memory)))
            context = context + conditioned
        logits = context @ head
        probabilities = softmax(logits)
        label_ids = [vocab.index(t) for t in labels]
        losses = np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(labels)), label_ids]
        return {"inputs": input_tokens, "labels": labels, "probabilities": probabilities,
                "losses": losses, "logits": logits}

    selected = [1, 3, 5]
    corrupted = SOURCE.copy()
    corrupted[1], corrupted[3] = "<mask>", "river"
    mlm = predict(corrupted, SOURCE)
    mlm.update({"selected": selected, "modes": ["mask", "random replacement", "unchanged"],
                "mean_selected_loss": float(mlm["losses"][selected].mean())})
    sequence = ["<start>", *SOURCE, "<eos>"]
    causal = predict(sequence[:-1], sequence[1:], causal=True)
    unmasked = predict(sequence[:-1], sequence[1:])
    src, target = corrupt_spans(SOURCE, [(1, 3), (5, 6)])
    target_ids = np.array([vocab.index(t) for t in target] + [vocab.index("<eos>")])
    decoder_ids = shift_right(target_ids, 0)
    seq2seq = predict([vocab[i] for i in decoder_ids], [vocab[i] for i in target_ids], True, src)
    return {"vocab": vocab, "original": SOURCE, "mlm": mlm, "causal": causal,
            "unmasked": unmasked, "span": seq2seq, "span_source": src,
            "span_target": [vocab[i] for i in target_ids],
            "leak_pairs": [(i, i+1) for i in range(len(sequence)-2)]}


def generation_experiment(seed=SEED, steps=6):
    """Fixed random bigram logits: a probability generator, not a trained GPT."""
    vocab = ["<start>", "bank", "loan", "approved", "river", "today", "<eos>"]
    rng = np.random.default_rng(seed + 3)
    transitions = rng.normal(size=(len(vocab), len(vocab)))
    runs = []
    for mode in ("greedy", "sampling"):
        for temperature in (0.25, 0.5, 1.0, 2.0):
            sampler = np.random.default_rng(seed + 4)
            context, records = [0, 1], []
            for _ in range(steps):
                logits = transitions[context[-1]].copy()
                probabilities = softmax(logits, temperature)
                chosen = int(probabilities.argmax()) if mode == "greedy" else int(sampler.choice(len(vocab), p=probabilities))
                records.append({"context": [vocab[i] for i in context], "logits": logits,
                                "probabilities": probabilities, "chosen": chosen, "token": vocab[chosen]})
                context.append(chosen)
                if vocab[chosen] == "<eos>":
                    break
            runs.append({"mode": mode, "temperature": temperature, "steps": records})
    return {"vocab": vocab, "transition_logits": transitions, "runs": runs}


def cache_experiment(seed=SEED, prompt=3, total=12):
    if not 1 <= prompt <= total:
        raise ValueError("require 1 <= prompt <= total")
    rng = np.random.default_rng(seed + 5)
    x = rng.normal(size=(total, 16)) + positions(total)
    wq, wk, wv = (rng.normal(scale=0.2, size=(16, 8)) for _ in range(3))
    kc, vc, cached, uncached, sizes = [], [], [], [], []
    cached_work, uncached_work, repeated = [], [], 0
    for length in range(prompt, total+1):
        query = x[length-1:length] @ wq
        full_k, full_v = x[:length] @ wk, x[:length] @ wv
        start = 0 if length == prompt else length-1
        kc.extend(x[start:length] @ wk)
        vc.extend(x[start:length] @ wv)
        cached.append(attention(query, np.asarray(kc), np.asarray(vc), visibility(1, length))[0][0])
        uncached.append(attention(query, full_k, full_v, visibility(1, length))[0][0])
        repeated += 2*length
        cached_work.append(2*length)
        uncached_work.append(repeated)
        sizes.append(length)
    return {"lengths": np.array(sizes), "cached_outputs": np.array(cached), "uncached_outputs": np.array(uncached),
            "cached_projection_vectors": np.array(cached_work), "uncached_projection_vectors": np.array(uncached_work),
            "max_abs_error": float(np.max(np.abs(np.array(cached)-np.array(uncached)))),
            "cache_bytes": np.array(sizes)*2*8*8, "head_width": 8, "bytes_per_element": 8, "prompt": prompt}


def cache_bytes(length, layers, kv_heads, head_dim, precision_bytes, batch=1):
    values = [length, layers, kv_heads, head_dim, precision_bytes, batch]
    if any(np.any(np.asarray(x) <= 0) for x in values):
        raise ValueError("all cache dimensions and precision bytes must be positive")
    return 2*np.asarray(length)*layers*kv_heads*head_dim*precision_bytes*batch


def connection_counts(n, m):
    n, m = np.asarray(n, dtype=np.float64), np.asarray(m, dtype=np.float64)
    if np.any(n < 1) or np.any(m < 0) or not np.isfinite(n).all() or not np.isfinite(m).all():
        raise ValueError("finite n >= 1 and m >= 0 required")
    return {"encoder": np.broadcast_arrays(n*n, n+m)[0], "decoder": (n+m)*(n+m+1)/2,
            "cross": n*n + m*(m+1)/2 + n*m}


def rag_experiment():
    docs = ["The bank approved the loan.", "The river bank flooded today.",
            "Loan approval requires income documents.", "The customer requested a bank loan.",
            "A recipe uses flour and milk.", "Football teams play today."]
    query = "bank loan approved"
    tokenize = lambda text: re.findall(r"[a-z]+", text.lower())
    vocabulary = sorted(set(tokenize(query + ' ' + ' '.join(docs))))

    def vector(text):
        terms = tokenize(text)
        counts = np.array([terms.count(term) for term in vocabulary], dtype=float)
        return counts / max(np.linalg.norm(counts), 1)

    vectors = np.array([vector(doc) for doc in docs])
    q = vector(query)
    cosine = vectors @ q
    order = np.argsort(-cosine, kind="stable")[:4]
    # Explicit toy pair features; no trained cross-encoder is executed.
    phrase = np.array([float("bank" in doc.lower() and "loan" in doc.lower()) for doc in docs])
    approval = np.array([float(re.search(r"\b(?:approved|approval)\b", doc.lower()) is not None) for doc in docs])
    joint = 0.4*cosine + 0.4*phrase + 0.2*approval
    reranked = order[np.argsort(-joint[order], kind="stable")]
    return {"query": query, "documents": docs, "vocabulary": vocabulary, "query_vector": q,
            "document_vectors": vectors, "cosine": cosine, "joint_scores": joint,
            "retrieved": order, "reranked": reranked,
            "pair_features": {"bank_and_loan": phrase, "approval_term": approval}}


def validate_experiments(models, attention_data, cache):
    tolerance = 1e-12
    checks = {}
    for family, item in attention_data.items():
        checks[f"{family}_rows_sum_to_one"] = bool(np.max(np.abs(item["weights"].sum(axis=1)-1)) <= tolerance)
        checks[f"{family}_masked_probability_zero"] = bool(np.all(item["weights"][~item["allowed"]] == 0))
        checks[f"{family}_weighted_values"] = bool(np.allclose(item["output"], item["weights"] @ item["v"], atol=tolerance))
    checks.update({
        "causal_lower_triangular": bool(np.all(attention_data["decoder"]["weights"][np.triu_indices(len(SOURCE), 1)] == 0)),
        "rectangular_cross_shape": attention_data["cross"]["weights"].shape == (len(TARGET), len(SOURCE)),
        "future_source_causal_invariance": bool(np.max(models["deltas"]["decoder"][:-1]) <= tolerance),
        "future_target_causal_invariance": bool(np.max(models["target_deltas"][:-1]) <= tolerance),
        "cached_uncached_equivalence": cache["max_abs_error"] <= tolerance,
        "manual_traces_match_official_layers": max(models["trace_errors"].values()) <= tolerance,
    })
    if not all(checks.values()):
        raise AssertionError(f"Numerical validation failed: {checks}")
    return {"tolerance": tolerance, "checks": checks, "cache_max_abs_error": cache["max_abs_error"],
            "causal_source_prefix_max_l2": float(models["deltas"]["decoder"][:-1].max()),
            "causal_target_prefix_max_l2": float(models["target_deltas"][:-1].max())}
