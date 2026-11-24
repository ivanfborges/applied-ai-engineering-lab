"""Compare three educational objectives on one explicitly synthetic corpus."""

from datetime import datetime, timezone
import hashlib
import json
import platform

import numpy as np

from from_scratch import analogy, cosine, nearest, train_glove, train_predictive, training_data, word_vector


def build_corpus():
    """Invented contexts intentionally give selected words the same roles."""
    groups = (
        (("doctor", "physician"), ("patient", "hospital", "medicine")),
        (("developer", "programmer"), ("software", "computer", "code")),
        (("car", "automobile"), ("road", "driver", "vehicle")),
        (("king", "queen"), ("palace", "crown", "royal")),
    )
    corpus = [[subject, "uses", context] for subjects, contexts in groups for subject in subjects for context in contexts]
    corpus += [sentence.split() for sentence in (
        "king is man", "queen is woman", "prince is man", "princess is woman",
        "prince uses palace", "princess uses palace", "bank holds money", "bank borders river",
    )]
    return corpus


def run_experiment():
    corpus = build_corpus()
    words, contexts, _, pairs, counts = training_data(corpus)
    fits = {
        "cbow": train_predictive(corpus, mode="cbow"),
        "skipgram": train_predictive(corpus, mode="skipgram"),
        "glove": train_glove(corpus),
    }
    models = {}
    for name, fit in fits.items():
        related = cosine(word_vector(fit, "doctor"), word_vector(fit, "physician"))
        unrelated = cosine(word_vector(fit, "doctor"), word_vector(fit, "developer"))
        ranking = analogy(fit, "man", "king", "woman", k=len(words))
        models[name] = {
            "objective": "mean observed-pair weighted squared log-count residual" if name == "glove" else "mean full-softmax NLL per prediction event",
            "representation": "input + output" if name == "glove" else "input table",
            "shape": list(fit.vectors.shape),
            "initial_loss": float(fit.losses[0]), "final_loss": float(fit.losses[-1]),
            "doctor_physician_cosine": related, "doctor_developer_cosine": unrelated,
            "shared_context_margin": related - unrelated,
            "doctor_neighbors": nearest(words, fit.vectors, word_vector(fit, "doctor"), exclude=("doctor",)),
            "bank_neighbors": nearest(words, fit.vectors, word_vector(fit, "bank"), exclude=("bank",)),
            "man_king_woman_top3": ranking[:3],
            "queen_rank": next(i + 1 for i, (word, _) in enumerate(ranking) if word == "queen"),
            "analogy_top1_is_queen": ranking[0][0] == "queen",
        }
    digest = hashlib.sha256(json.dumps(corpus, separators=(",", ":")).encode()).hexdigest()
    return {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Each objective can reduce its own training loss; shared contexts may yield a positive doctor similarity margin. Analogy success is not assumed.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "data": {"source": "code-defined synthetic corpus; no external dataset", "sha256": digest,
                 "sentences": len(corpus), "tokens": sum(map(len, corpus)), "vocabulary": len(words),
                 "cbow_events": len(contexts), "skipgram_pairs": len(pairs), "glove_observed_pairs": int(np.count_nonzero(counts))},
        "configuration": {"dim": 8, "window": 2, "epochs": 400, "lr": 0.8, "seed": 65,
                          "optimizer": "float64 full-batch gradient descent, constant lr",
                          "glove_x_max": 10.0, "glove_alpha": 0.75,
                          "predictive_contexts": "fixed symmetric window; uniform weighting; no subsampling",
                          "glove_counts": "symmetric directed inverse-distance counts; sentence boundaries respected"},
        "models": models,
        "interpretation_candidate": "Loss reduction describes fitting. Shared-context margins and analogy rankings are fixture-specific; author review required.",
        "limitations": "One seed, engineered contexts, training-only diagnostics, unequal event counts and objectives, no held-out semantic judgments or production benchmark. Bank always has one vector.",
    }


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2, allow_nan=False))
