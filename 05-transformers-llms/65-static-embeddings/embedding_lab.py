"""Synthetic data, real optimization traces and bounded embedding experiments."""

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
import logging
import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from from_scratch import (
    EmbeddingFit, analogy, cosine, nearest, positive_float, positive_int,
    predictive_loss_grad, train_glove, train_predictive, training_data, unit_rows,
)


CATEGORIES = {
    "healthcare": "doctor physician nurse hospital patient medicine healthcare surgeon".split(),
    "technology": "developer programmer engineer software computer technology code architect".split(),
    "transport": "car automobile truck bus road vehicle driver transport".split(),
    "geography": "brazil brasilia france paris germany berlin country capital".split(),
    "royalty": "king queen prince princess man woman palace crown royal".split(),
    "food": "apple banana orange fruit food fresh ripe".split(),
    "finance": "money finance loan customer deposit bank".split(),
    "river": "river water fisherman shore overflow stream".split(),
    "attributes": "hot cold large small temperature size".split(),
}
COLORS = dict(zip(CATEGORIES, ["#2563eb", "#7c3aed", "#d97706", "#0d9488", "#be185d", "#65a30d", "#ca8a04", "#0284c7", "#c2410c"]))
RARE_WORDS = ("surgeon", "architect")
ANALOGIES = (("man", "king", "woman", "queen"), ("france", "paris", "germany", "berlin"))


def category(word: str) -> str:
    return next((name for name, words in CATEGORIES.items() if word in words), "other")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def make_corpus(size=180, repetitions=1, seed=65, finance_ratio=0.5) -> list[list[str]]:
    """Multiple grammatical templates; rare entries occur once per repetition.

    Size controls the common-word portion. Repetition duplicates the generated
    collection; it does not create independent linguistic evidence.
    """
    positive_int(size, "size")
    positive_int(repetitions, "repetitions")
    if not np.isfinite(finance_ratio) or not 0 <= finance_ratio <= 1:
        raise ValueError("finance_ratio must be in [0, 1]")
    pools = [
        [f"the {who} treats a patient in the hospital" for who in ("doctor", "physician", "nurse")]
        + [f"the patient asks the {who} for medicine" for who in ("doctor", "physician", "nurse")]
        + ["healthcare staff work at the hospital", "medicine helps the patient"],
        [f"the {who} writes software on a computer" for who in ("developer", "programmer", "engineer")]
        + [f"the team asks the {who} to review code" for who in ("developer", "programmer", "engineer")]
        + ["technology teams build computer software", "the computer runs new code"],
        [f"the {what} carries a driver along the road" for what in ("car", "automobile", "truck", "bus")]
        + [f"the driver repairs the {what} vehicle" for what in ("car", "truck", "bus")]
        + ["transport workers inspect the road"],
        [f"{city} is the capital of {country}" for country, city in (("brazil", "brasilia"), ("france", "paris"), ("germany", "berlin"))]
        + [f"the country {country} has capital {city}" for country, city in (("brazil", "brasilia"), ("france", "paris"), ("germany", "berlin"))],
        [f"the {who} wears a royal crown in the palace" for who in ("king", "queen", "prince", "princess")]
        + ["the king is a man", "the queen is a woman", "the prince is a man", "the princess is a woman"],
        [f"the {fruit} is a fresh fruit" for fruit in ("apple", "banana", "orange")]
        + [f"the customer eats a ripe {fruit}" for fruit in ("apple", "banana", "orange")]
        + ["fresh food includes fruit"],
        [f"the water is {value} today" for value in ("hot", "cold")]
        + [f"the vehicle is {value} today" for value in ("large", "small")]
        + ["temperature describes hot and cold water", "size describes large and small objects"],
    ]
    financial = ["the bank approved the loan", "the customer deposited money at the bank", "finance staff inspect the bank deposit"]
    river = ["the river overflowed its bank", "the fisherman sat on the river bank", "water moves along the shore near the bank"]
    # Include the support vocabulary before random draws, without seeding rare words repeatedly.
    support = [sentence for pool in pools for sentence in pool] + financial + river
    support += ["the surgeon treats a patient", "the architect reviews computer software", "the company complied with regulations", "the company did not comply with regulations"]
    rng = np.random.default_rng(seed)
    common = [sentence for pool in pools for sentence in pool]
    result = support + [common[int(rng.integers(len(common)))] for _ in range(size)]
    bank_count = max(12, size // 3)
    n_finance = round(bank_count * finance_ratio)
    for n, pool in ((n_finance, financial), (bank_count - n_finance, river)):
        result.extend(pool[int(rng.integers(len(pool)))] for _ in range(n))
    rng.shuffle(result)
    return [tokenize(sentence) for sentence in result] * repetitions


def noise_distribution(frequencies, exponent=0.75) -> np.ndarray:
    counts = np.asarray(frequencies, dtype=float)
    if counts.ndim != 1 or len(counts) == 0 or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("frequencies must be a finite nonnegative vector with positive mass")
    positive_float(exponent, "exponent")
    weights = counts**exponent
    return weights / weights.sum()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -np.asarray(x)))


def sgns_loss_grad(inputs, outputs, centers, contexts, negatives):
    """Mean loss per positive event, summing its K negative contributions.

    Repeated sampled IDs accumulate; noise is independent and may equal a
    positive context. Do not silently resample collisions: that changes q.
    """
    v, positive, negative = inputs[centers], outputs[contexts], outputs[negatives]
    z_pos = np.einsum("bd,bd->b", v, positive)
    z_neg = np.einsum("bd,bkd->bk", v, negative)
    loss = np.mean(np.logaddexp(0, -z_pos) + np.logaddexp(0, z_neg).sum(axis=1))
    dp = (sigmoid(z_pos) - 1) / len(centers)
    dn = sigmoid(z_neg) / len(centers)
    dv, du = np.zeros_like(inputs), np.zeros_like(outputs)
    np.add.at(dv, centers, dp[:, None] * positive + np.einsum("bk,bkd->bd", dn, negative))
    np.add.at(du, contexts, dp[:, None] * v)
    np.add.at(du, negatives.ravel(), (dn[:, :, None] * v[:, None, :]).reshape(-1, inputs.shape[1]))
    return float(loss), (dv, du)


@dataclass
class Snapshot:
    epoch: int
    parameters: list[np.ndarray]
    loss: float


@dataclass
class LabFit:
    mode: str
    fit: EmbeddingFit
    snapshots: list[Snapshot]
    seconds: float
    configuration: dict


def train_model(corpus, mode="cbow", dim=10, window=3, epochs=80, seed=65, lr=0.8, negative=5) -> LabFit:
    for value, name in ((dim, "dim"), (epochs, "epochs"), (window, "window"), (negative, "negative")):
        positive_int(value, name)
    positive_float(lr, "lr")
    if mode not in ("cbow", "skipgram", "glove", "sgns"):
        raise ValueError("unknown embedding architecture")
    snapshots = []
    capture = set(np.linspace(0, epochs, min(16, epochs + 1), dtype=int))
    def recorder(epoch, parameters, loss):
        if epoch in capture:
            snapshots.append(Snapshot(epoch, [p.copy() for p in parameters], float(loss)))
    start = perf_counter()
    if mode == "glove":
        fit = train_glove(corpus, dim, window, epochs, lr, seed, recorder=recorder)
    elif mode in ("cbow", "skipgram"):
        fit = train_predictive(corpus, mode, dim, window, epochs, lr, seed, recorder=recorder)
    else:
        words, _, _, pairs, _ = training_data(corpus, window)
        freq = Counter(w for sentence in corpus for w in sentence)
        q = noise_distribution([freq[w] for w in words])
        rng = np.random.default_rng(seed)
        v = rng.normal(0, 0.1, (len(words), dim))
        u = rng.normal(0, 0.1, v.shape)
        probe = np.random.default_rng(seed + 10000).choice(len(words), (len(pairs), negative), p=q)
        losses = []
        for epoch in range(epochs + 1):
            loss, _ = sgns_loss_grad(v, u, pairs[:, 0], pairs[:, 1], probe)
            if not np.isfinite(loss):
                raise FloatingPointError("SGNS diverged")
            losses.append(loss)
            recorder(epoch, [v, u], loss)
            if epoch == epochs:
                break
            for ids in np.array_split(rng.permutation(len(pairs)), max(1, int(np.ceil(len(pairs) / 256)))):
                batch = pairs[ids]
                negatives = rng.choice(len(words), (len(batch), negative), p=q)
                _, (dv, du) = sgns_loss_grad(v, u, batch[:, 0], batch[:, 1], negatives)
                v -= lr * dv
                u -= lr * du
        fit = EmbeddingFit(words, v.copy(), np.asarray(losses))
    config = {"mode": mode, "dim": dim, "window": window, "epochs": epochs, "seed": seed, "lr": lr,
              "negative": negative if mode == "sgns" else None, "noise_exponent": 0.75 if mode == "sgns" else None,
              "representation": "input+output" if mode == "glove" else "input", "snapshot_epochs": sorted(capture)}
    return LabFit(mode, fit, snapshots, perf_counter() - start, config)


def snapshot_vectors(model: LabFit) -> list[np.ndarray]:
    return [s.parameters[0] + s.parameters[1] if model.mode == "glove" else s.parameters[0] for s in model.snapshots]


def fixed_projection(matrices, components=2):
    """One PCA fitted across all recorded states, never separately per frame."""
    if not matrices or any(m.ndim != 2 or m.shape != matrices[0].shape for m in matrices):
        raise ValueError("snapshots must be aligned matrices")
    all_rows = np.vstack(matrices)
    n = min(components, all_rows.shape[1], all_rows.shape[0])
    pca = PCA(n_components=n, svd_solver="full").fit(all_rows)
    coordinates = [np.pad(pca.transform(m), ((0, 0), (0, components - n))) for m in matrices]
    return coordinates, np.pad(pca.explained_variance_ratio_, (0, components - n)), pca


def association_matrices(counts):
    x = np.asarray(counts, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all() or np.any(x < 0) or x.sum() <= 0:
        raise ValueError("counts must be a nonnegative matrix with positive mass")
    observed = x > 0
    logs = np.full(x.shape, np.nan)
    logs[observed] = np.log(x[observed])
    conditional = np.divide(x, x.sum(axis=1, keepdims=True), out=np.zeros_like(x), where=x.sum(axis=1, keepdims=True) > 0)
    denominator = x.sum(axis=1)[:, None] * x.sum(axis=0)[None, :]
    pmi = np.full(x.shape, np.nan)
    pmi[observed] = np.log(x[observed] * x.sum() / denominator[observed])
    # Unobserved PMI is -infinity with positive marginals; display it masked.
    # PPMI has the conventional zero for those entries, not an invented finite PMI.
    ppmi = np.maximum(np.nan_to_num(pmi, nan=0), 0)
    return {"counts": x, "log_counts": logs, "conditional": conditional, "pmi": pmi, "ppmi": ppmi}


def expected_sgns_loss_grad(v, u, counts, q, negative):
    """Exact expectation over noise for a controlled finite-count experiment."""
    scores = v @ u.T
    positive = counts / counts.sum()
    noise_mass = negative * (counts.sum(axis=1) / counts.sum())[:, None] * q[None, :]
    loss = np.sum(positive * np.logaddexp(0, -scores) + noise_mass * np.logaddexp(0, scores))
    delta = positive * (sigmoid(scores) - 1) + noise_mass * sigmoid(scores)
    return float(loss), (delta @ u, delta.T @ v)


def pmi_experiment(seed=65, steps=1200):
    counts = np.array([[35., 8, 4], [5, 28, 10], [8, 3, 20]])
    results = []
    for exponent in (1., 0.75):
        q = noise_distribution(counts.sum(axis=0), exponent)
        conditional = counts / counts.sum(axis=1, keepdims=True)
        target = np.log(conditional / (5 * q[None, :]))
        rng = np.random.default_rng(seed)
        v, u = rng.normal(0, .2, (3, 2)), rng.normal(0, .2, (3, 2))
        for _ in range(steps):
            _, (dv, du) = expected_sgns_loss_grad(v, u, counts, q, 5)
            v -= .8 * dv
            u -= .8 * du
        learned = v @ u.T
        results.append({"exponent": exponent, "target": target, "learned": learned,
                        "rmse": float(np.sqrt(np.mean((target - learned)**2))), "steps": steps,
                        "objective": "exact expected SGNS; no sampled noise; rank 2"})
    return results


def negative_trace(seed=65, steps=20, lr=.25):
    rng = np.random.default_rng(seed)
    v, u = rng.normal(0, .5, (3, 2)), rng.normal(0, .5, (3, 2))
    result = []
    for step in range(steps):
        center, context = np.array([0]), np.array([1])
        noise = np.array([[2, 2]]) if step % 2 == 0 else np.array([[0, 2]])
        loss, gradients = sgns_loss_grad(v, u, center, context, noise)
        before = [v.copy(), u.copy()]
        v -= lr * gradients[0]
        u -= lr * gradients[1]
        result.append({"step": step, "before": before, "after": [v.copy(), u.copy()], "gradients": gradients,
                       "negatives": noise[0].tolist(), "loss_before": loss, "lr": lr,
                       "dot_before": float(before[0][0] @ before[1][1]), "dot_after": float(v[0] @ u[1]),
                       "cos_before": cosine(before[0][0], before[1][1]), "cos_after": cosine(v[0], u[1])})
    return result


def forward_trace(mode="cbow", seed=65):
    corpus = [["doctor", "treats", "patient"], ["physician", "treats", "patient"]]
    words, features, targets, pairs, _ = training_data(corpus, 1)
    if mode == "skipgram":
        center = words.index("treats")
        ids = np.where(pairs[:, 0] == center)[0][:2]
        features, targets = np.eye(len(words))[pairs[ids, 0]], pairs[ids, 1]
    else:
        features, targets = features[[1]], targets[[1]]
    rng = np.random.default_rng(seed)
    v, u = rng.normal(0, .4, (len(words), 2)), rng.normal(0, .4, (len(words), 2))
    hidden = features @ v
    logits = hidden @ u.T
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    loss, (dv, du) = predictive_loss_grad(v, u, features, targets)
    after_v, after_u = v - .4 * dv, u - .4 * du
    return dict(words=words, features=features, targets=targets, inputs=v, outputs=u, hidden=hidden,
                logits=logits, probabilities=probs, loss=loss, dv=dv, du=du,
                after_inputs=after_v, after_outputs=after_u, after_loss=predictive_loss_grad(after_v, after_u, features, targets)[0])


def diagnostics(model: LabFit, frequencies, k=3):
    fit = model.fit
    unit = unit_rows(fit.vectors)
    categories = [category(w) for w in fit.words]
    mask = np.array([c != "other" for c in categories])
    similarity = unit @ unit.T
    within, between, purity, rare = [], [], [], []
    for i, word in enumerate(fit.words):
        if not mask[i]:
            continue
        others = [j for j in range(len(fit.words)) if mask[j] and j != i]
        for j in others:
            (within if categories[j] == categories[i] else between).append(similarity[i, j])
        # Ranking uses the entire vocabulary; stop words are not hidden candidates.
        ranking = nearest(fit.words, fit.vectors, fit.vectors[i], k, (word,))
        score = np.mean([category(w) == categories[i] for w, _ in ranking])
        purity.append(score)
        if word in RARE_WORDS:
            rare.append(score)
    covered = [row for row in ANALOGIES if all(w in fit.words for w in row)]
    accuracy = np.mean([analogy(fit, a, b, c, 1)[0][0] == answer for a, b, c, answer in covered]) if covered else 0.
    return {"within_cosine": float(np.mean(within)), "between_cosine": float(np.mean(between)),
            "purity_at3": float(np.mean(purity)), "rare_purity_at3": float(np.mean(rare)) if rare else 0.,
            "analogy_accuracy": float(accuracy), "analogy_coverage": len(covered) / len(ANALOGIES),
            "category_coverage": sum(w in fit.words for group in CATEGORIES.values() for w in group) / sum(map(len, CATEGORIES.values())),
            "seconds": model.seconds, "input_table_bytes": int(fit.vectors.nbytes),
            "two_training_tables_bytes": int(2 * fit.vectors.nbytes), "vocabulary": len(fit.words),
            "rare_frequencies": {w: int(frequencies.get(w, 0)) for w in RARE_WORDS}}


def ranking_metrics(ranking, relevance, k=3):
    positive_int(k, "k")
    if len(ranking) != len(set(ranking)) or any(g < 0 or not np.isfinite(g) for g in relevance.values()):
        raise ValueError("ranking IDs must be unique; relevance grades finite and nonnegative")
    relevant = {doc for doc, grade in relevance.items() if grade > 0}
    top = ranking[:k]
    recall = len(set(top) & relevant) / len(relevant) if relevant else 0.
    rr = next((1 / (i + 1) for i, doc in enumerate(ranking) if doc in relevant), 0.)
    dcg = sum((2**relevance.get(doc, 0) - 1) / np.log2(i + 2) for i, doc in enumerate(top))
    ideal = sum((2**g - 1) / np.log2(i + 2) for i, g in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    return {f"recall_at{k}": float(recall), "rr": float(rr), f"ndcg_at{k}": float(dcg / ideal) if ideal else 0.}


DOCUMENTS = [
    "the doctor treats the patient in the hospital",
    "the programmer writes computer software and code",
    "the company complied with regulations",
    "the company did not comply with regulations",
    "the customer deposited money at the bank",
    "the fisherman sat on the river bank near water",
    "the car carries the driver along the road",
    "the truck carries food along the road",
]
QUERIES = [
    ("physician hospital", {0: 2}, "shared context"),
    ("company complied regulations", {2: 2, 3: 0}, "negation"),
    ("bank river water", {5: 2, 4: 0}, "polysemy"),
    ("software code", {1: 2}, "exact terminology"),
    ("vehicle driver road", {6: 2, 7: 1}, "graded transport"),
    ("kubernetes", {1: 1}, "fully OOV query"),
]


def mean_vectors(texts, fit, idf=None):
    index = {w: i for i, w in enumerate(fit.words)}
    vectors, coverage = [], []
    for text in texts:
        tokens = tokenize(text)
        known = [w for w in tokens if w in index]
        coverage.append(len(known) / len(tokens) if tokens else 0.)
        weights = np.array([idf.get(w, 1.) if idf else 1. for w in known])
        vectors.append(np.average(fit.vectors[[index[w] for w in known]], axis=0, weights=weights) if known else np.zeros(fit.vectors.shape[1]))
    return np.asarray(vectors), np.asarray(coverage)


def safe_cosine_matrix(a, b):
    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    return np.clip(normalize(a) @ normalize(b).T, -1, 1)


def retrieval_experiment(models):
    texts = [row[0] for row in QUERIES]
    tfidf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None).fit(DOCUMENTS)
    docs, queries = tfidf.transform(DOCUMENTS), tfidf.transform(texts)
    scores = {"tfidf": (queries @ docs.T).toarray()}
    coverage = {"tfidf": [float(any(w in tfidf.vocabulary_ for w in tokenize(q))) for q in texts]}
    vectors = {}
    for mode in ("skipgram", "glove"):
        d, _ = mean_vectors(DOCUMENTS, models[mode].fit)
        q, cov = mean_vectors(texts, models[mode].fit)
        vectors[mode] = {"documents": d, "queries": q}
        scores[mode] = safe_cosine_matrix(q, d)
        coverage[mode] = cov.tolist()
    idf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    d, _ = mean_vectors(DOCUMENTS, models["skipgram"].fit, idf)
    q, _ = mean_vectors(texts, models["skipgram"].fit, idf)
    scores["idf_weighted_skipgram"] = safe_cosine_matrix(q, d)
    coverage["idf_weighted_skipgram"] = coverage["skipgram"]
    results, details = {}, {}
    for mode, matrix in scores.items():
        rows = []
        details[mode] = []
        for i, (query, grades, failure) in enumerate(QUERIES):
            # Abstain when the query is unrepresented; exact score ties use doc ID.
            ranking = sorted(range(len(DOCUMENTS)), key=lambda j: (-matrix[i, j], j)) if coverage[mode][i] > 0 else []
            row = ranking_metrics(ranking, grades)
            rows.append(row)
            details[mode].append({"query": query, "case": failure, "ranking": ranking, "scores": matrix[i].tolist(),
                                  "coverage": coverage[mode][i], "grades": grades, **row})
        results[mode] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    return {"summary": results, "details": details, "scores": scores, "vectors": vectors,
            "documents": DOCUMENTS, "queries": QUERIES, "k": 3,
            "protocol": "hand-authored synthetic judgments; TF-IDF fits document collection only; embeddings fit separate synthetic corpus; no tuning on judgments"}


def sweep_experiments(corpus, seed=65, epochs=40, extended=False):
    """One-factor controls; timings are observations, not matched tuning budgets."""
    rows = []
    frequencies = Counter(w for sentence in corpus for w in sentence)
    seeds = [seed, seed + 1, seed + 2] if extended else [seed, seed + 1]
    def run(group, value, mode="skipgram", local=corpus, **kwargs):
        config = dict(dim=10, window=3, epochs=epochs, seed=seed)
        config.update(kwargs)
        logging.getLogger("day65").info("Sweep %s=%s: %s dim=%s window=%s epochs=%s seed=%s",group,value,mode,config["dim"],config["window"],config["epochs"],config["seed"])
        model = train_model(local, mode=mode, **config)
        rows.append({"group": group, "value": value, **model.configuration,
                     **diagnostics(model, Counter(w for sentence in local for w in sentence))})
    for trial_seed in seeds:
        for mode in ("cbow", "skipgram", "glove"):
            run("architecture", mode, mode, seed=trial_seed)
    for window in (1, 3, 5):
        run("window", window, window=window)
    for dim in (2, 10, 30, 100):
        run("dimension", dim, dim=dim)
    for k in (1, 5, 15):
        run("negative", k, "sgns", negative=k, lr=.3)
    for n in (max(15, len(corpus) // 4), len(corpus) // 2, len(corpus)):
        # Rebuild a supported smaller corpus instead of silently dropping evaluation words.
        local = make_corpus(size=n, seed=seed)
        run("corpus_size", len(local), local=local)
    for count in sorted({max(5, epochs // 4), epochs, epochs * 2}):
        run("epochs", count, epochs=count)
    return rows
