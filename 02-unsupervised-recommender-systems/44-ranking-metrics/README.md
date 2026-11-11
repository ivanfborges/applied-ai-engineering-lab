# Day 44 — Ranking Metrics

**Can two systems find the same relevant items yet produce different ranking
quality? And can a perfect reranker hide evidence lost during retrieval?**

Precision@K and Recall@K measure top-K purity and coverage. AP/MAP reward
earlier placement of multiple relevant items, RR/MRR focus on the first hit,
and NDCG measures discounted graded utility relative to an ideal ordering.
These distinctions matter in recommendation, search and RAG when the final
display or context budget is smaller than the candidate pool.

[Day 43](../43-recommender-systems-ii/) used a single held-out positive per user.
This study adds multiple relevant items, graded judgments and explicit
normalization universes. It evaluates **hand-constructed synthetic rankings**;
it does not train a retriever or measure a deployed application.

## Run and inspect

From the repository root with Python 3.11+:

```bash
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/44-ranking-metrics/example.py --save
python -m pytest -q 02-unsupervised-recommender-systems/44-ranking-metrics/tests/test_ranking_metrics.py
```

Dependencies follow [pyproject.toml](../../pyproject.toml); no new packages or
extras are needed. The core uses NumPy, and the example/tests use scikit-learn
as an independent NDCG check. The educational code is not a production evaluator.

The example prints JSON with inputs, configuration, versions, per-query and
macro scores, cutoff and candidate diagnostics, limitations and review status.
`--save` writes a unique dated JSON to the script's ignored `outputs/` folder,
independent of the working directory, without replacing earlier records.
Omit it for console output only. No downloaded data or random draws are used.

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | ID-based per-query metrics and equal-query aggregation with explicit validation |
| [example.py](example.py) | Fixed-membership reordering, K sweep, AP conventions, candidate-universe probe and library agreement |
| [notes.md](notes.md) | Formulas, worked arithmetic, evaluation assumptions, trade-offs and unrun extensions |
| [tests/test_ranking_metrics.py](tests/test_ranking_metrics.py) | Hand arithmetic, library parity, invariants, short lists, invalid inputs and candidate boundaries |
| [interview_questions.md](interview_questions.md) | Metric choice and evaluation-design questions |
| [references.md](references.md) | Primary sources and library conventions |
| [ranking_metrics_visual_lab/README.md](ranking_metrics_visual_lab/README.md) | Eight-section Streamlit lab, Plotly animations/3D landscape, four GIF exporters, selected previews and visual tests |

## Interactive visual companion

The [visual lab](ranking_metrics_visual_lab/README.md) makes the calculations
inspectable through document swaps and grade edits, K curves, AP scan steps,
RR/MRR examples, DCG/IDCG contributions, 56 valid 3D configurations, metric
disagreements and a synthetic two-stage RAG/context-budget simulation.

```bash
cd 02-unsupervised-recommender-systems/44-ranking-metrics/ranking_metrics_visual_lab
streamlit run app.py
python export_gifs.py
pytest -q
```

Run that block from the repository root after installing `.[dev]`. The visual
guide includes a requirements entry point for installation from the laboratory
directory. Its November 6 visual run, seed/configuration and GIF inventory are
separate from the historical study record below. Two GIF previews are selected
for Git; other exports, screenshots and dated JSON records remain ignored.

## Evaluation contract

- Judgments map unique item IDs to finite, nonnegative grades over an explicit
  universe. The example fully judges all six items for each of two queries.
- Binary relevance is **grade > 0**. Grades are illustrative utility labels,
  not a validated product rubric.
- Precision divides by K, including for short or empty result lists. Recall
  divides by all relevant items in the judged universe, including unreturned ones.
- AP@K defaults to division by the total number of relevant items. The explicit
  `capped` alternative divides by the smaller of that count and K.
- NDCG defaults to exponential gain $2^g-1$ and logarithmic discount; `linear`
  gain is also supported. Ideal DCG uses the **whole judged universe**.
- AP and RR are per-query metrics; MAP and MRR are their equal-query means.
  Query sets must match exactly. An unsuccessful query needs an explicit empty
  ranking so it cannot silently disappear from the mean.
- No-positive queries raise by default. `no_relevant="zero"` is an explicit
  scoring convention that retains them in aggregation, not an abstention metric.
- Repeated IDs and unjudged returned IDs raise, even below K. Score ties must
  be resolved upstream; the supplied order is evaluated without tie averaging.

## Executed synthetic check — November 6, 2026

Executed by Codex at **12:46:29 UTC (09:46:29 America/Sao_Paulo)** on Windows,
with Python 3.11.14, NumPy 2.4.6 and scikit-learn 1.9.1. No random seed is
needed. The exact inputs and formulas are in [notes.md](notes.md).

**Hypothesis:** changing order within the same top-five membership changes
rank-sensitive metrics without changing precision/recall; candidate-only
normalization can conceal missing evidence.

| Equal-query mean at K=5 | Initial order | Reordered within top five |
|---|---:|---:|
| Precision | 0.600000 | 0.600000 |
| Recall | 0.750000 | 0.750000 |
| MAP | 0.481250 | 0.750000 |
| MRR | 0.666667 | 1.000000 |
| NDCG, exponential | 0.648661 | 0.911988 |

For the first query, exponential NDCG@5 was **0.825122** and linear NDCG@5
was **0.778331**. Both matched scikit-learn after aligning the gain transform
and supplying distinct predicted scores. The library's direct relevance gain
and tie behavior are documented in its [NDCG API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).

The candidate probe retains IDs `b,c,d,e` from the first query and uses
label-informed oracle ordering `c,d,b,e`, with final K=3. Candidate recall
is **0.50**. Normalizing over candidates gives NDCG@3 = **1.0** and recall =
**1.0**; retaining the six-item universe gives NDCG@3 = **0.349370** and recall
= **0.50**. The oracle is a diagnostic ceiling for that fixed pool, not a
trained reranker. These two evaluations have different denominators.

**Interpretation candidate — author review required:** the constructed cases
illustrate how order and evaluation-universe choices change what a score
means. They do not estimate the benefit of a real reranker. Relevance judgments,
metric conventions and application relevance still require the author's review.

## Takeaways and limits

Fix the query set, relevance unit, judgment universe, K, gain, AP denominator
and no-answer policy before comparing systems. MRR ignores later evidence;
precision/recall ignore ordering within K; NDCG can be perfect at a small cutoff
without complete recall. Offline relevance does not measure answer grounding,
evidence sufficiency, diversity, latency, satisfaction or causal product value.

Two constructed queries provide no statistical reliability or real-world
performance estimate. Incomplete annotations and semantic duplicates need
separate policies; rejecting unknown IDs cannot detect missing labels in the
declared universe. Further studies remain suggestions in [notes.md](notes.md).
