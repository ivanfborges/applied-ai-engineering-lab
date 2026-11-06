# Day 42 — Recommender Systems I

[Module index](../README.md) · [Roadmap](../../ROADMAP.md)

How can shared rating patterns suggest an unseen item, and what should a
recommender return when those patterns are absent? This study connects the
user–item matrix to an executable user-based collaborative filter, an explicit
fallback policy and a small evaluation that hides future ratings before fitting.
It uses **hand-authored synthetic explicit ratings**, not a public benchmark.

Collaborative filtering learns from interactions across users and items;
content-based filtering uses item attributes. The distinction matters for
discovery, sparse histories and cold start. Explicit ratings and implicit events
also require different interpretations: a missing rating is unknown, and an
unclicked item may never have been shown.

## What is implemented

- Pearson correlation on co-rated items, minimum overlap and significance shrinkage.
- Positive-neighbor, mean-adjusted predictions with item-specific neighborhoods.
- Training-only item-mean → user-mean → global-mean fallbacks, with reason codes.
- Unseen-item recommendation with eligibility filtering and deterministic ties.
- One frozen temporal holdout, rating-error baselines and warm/cold segment reporting.

The baseline uses NumPy only. No specialized recommender framework is needed.
Item-based CF, implicit confidence weighting, content/hybrid approaches and
exposure bias are discussed in the notes. The separate visual lab adds a small
masked factorization model and ranking demonstrations as a geometric introduction
to the deeper material in Days 43–44.

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | Educational similarity, fitted user neighborhoods, fallback and candidate policy |
| [example.py](example.py) | Synthetic snapshot, global cutoff, held-out error comparison and recommendation lists |
| [tests/test_recommender.py](tests/test_recommender.py) | Numerical identities, invalid inputs, cold start, selection rules and future-label isolation |
| [notes.md](notes.md) | Formulas, feedback semantics, design choices and dated experiment record |
| [interview_questions.md](interview_questions.md) | Questions about evidence, objectives and operational trade-offs |
| [references.md](references.md) | Primary sources and dataset provenance |
| [visualize_day42.py](visualize_day42.py) | Main entry point for all ten offline visual scenes |
| [VISUAL_LAB.md](VISUAL_LAB.md) | Scene-by-scene guide, assumptions, selected previews and distinct visual-run record |
| [recommender_visual_core.py](recommender_visual_core.py) | Seeded 12×10 data, similarity variants, traces, metrics and simulations |
| [factorization.py](factorization.py) | Educational masked-loss factorization with real optimization checkpoints |
| [recommender_visuals.py](recommender_visuals.py) | Matplotlib/Pillow scenes, standalone Plotly explorer and gallery |
| [tests/test_visual_lab.py](tests/test_visual_lab.py) | Numerical, gradient, evaluation-boundary and temporary-rendering checks |

## Run from the repository root

Use stable Python 3.11+ and the canonical dependencies:

```bash
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/42-recommender-systems-i/from_scratch.py
python 02-unsupervised-recommender-systems/42-recommender-systems-i/example.py
python -m pytest -q 02-unsupervised-recommender-systems/42-recommender-systems-i/tests/test_recommender.py
```

The example prints JSON containing its configuration, versions, per-pair scores,
fallback reasons, segment errors and recommendation lists. It downloads nothing
and writes no artifacts. The similarity demonstration prints `0.331731` with
the default three-common-item threshold and shrinkage of five.

## Explore the visual lab

```bash
python 02-unsupervised-recommender-systems/42-recommender-systems-i/visualize_day42.py --all
python 02-unsupervised-recommender-systems/42-recommender-systems-i/visualize_day42.py --list
python 02-unsupervised-recommender-systems/42-recommender-systems-i/visualize_day42.py --scene 5
python 02-unsupervised-recommender-systems/42-recommender-systems-i/visualize_day42.py --fast
```

Open the generated `outputs/day42/index.html` inside this topic for the gallery,
or `outputs/day42/05_factorization_3d.html` for user/epoch controls and actual 3D
factors. Both work locally with no external assets. The generator creates 14
PNGs and four GIFs; `--animate-matrix` adds a fifth GIF. Fast exports go into a
separate `outputs/day42/fast/` directory. The [visual guide](VISUAL_LAB.md) explains
all ten scenes and includes three inspected public previews; bulky HTML, other
exports and reports stay ignored. No browser opens automatically.

The visual dataset and evaluation protocols differ from the eight-pair study
below. Their observations and interpretation candidates have a separate dated
record in the guide. Author review remains pending for both scopes.

## Observed result and interpretation boundary

On November 6, 2026, the assistant ran the fixed 30-training-rating,
8-future-rating example with Python 3.11.14 and NumPy 2.4.6. Overall MAE was
`0.738605` for the neighborhood-plus-fallback policy and `1.577083` for the
fallback-only baseline. Only four of eight predictions had collaborative
support; all four cold-start predictions used fallback values. The
[full experiment record](notes.md#executed-experiment--november-6-2026)
contains the segment breakdown and limitations.

**Interpretation candidate — author review required:** the constructed matrix
illustrates how shared preferences can improve warm-pair rating predictions
while leaving cold-start predictions entirely dependent on fallback policy.
These eight chosen labels cannot establish general superiority, ranking quality,
user satisfaction or business impact.

## Key takeaways and limits

- Missing values remain `NaN`; no missing rating becomes a zero-star label.
- Small overlap can give extreme Pearson correlations. Thresholds and shrinkage
  are controls, not guarantees of reliable similarity.
- Similarity uses overlap-local means; prediction uses each user's full training
  mean. These are distinct statistics.
- A finite fallback score is not evidence of personalization. A new item's mean
  is unavailable; the recommendation list excludes items with no training ratings.
- Rating prediction and ranking answer different questions. `example.py`'s printed
  lists have no ranking-quality claim. Visual scene 8 separately measures synthetic
  ranking with explicitly defined candidates and relevant labels.
- Dense storage and pairwise user comparison are deliberately small-scale.
  There is no production serving, implicit-feedback learner, exposure correction
  or online experiment. Interpretations remain pending author review.
