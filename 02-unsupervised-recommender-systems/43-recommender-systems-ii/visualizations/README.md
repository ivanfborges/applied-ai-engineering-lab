# Recommender Systems II — Visual Learning Lab

[Study and historical experiment](../README.md) · [Numerical states](visual_core.py) ·
[Generator](visualizations.py) · [Focused tests](tests/test_visual_lab.py)

Observe actual sparse matrices, factorization scores, embedding evolution,
pairwise SGD, source contributions and candidate filtering. Ten scenes connect
these mechanisms without a server, GPU, cloud service or external dataset.

## Install and run

Use Python 3.11+ from the repository root. Dependencies are already declared in
[pyproject.toml](../../../pyproject.toml): NumPy, scikit-learn, Matplotlib, Plotly
and Pillow; pytest is supplied by `dev`. There is no separate requirements file,
FFmpeg, Kaleido, PyTorch, CDN or runtime package installation.

```powershell
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/43-recommender-systems-ii/visualizations/visualizations.py --all --seed 43
```

Run an individual scene:

```powershell
python 02-unsupervised-recommender-systems/43-recommender-systems-ii/visualizations/visualizations.py --module embeddings
python 02-unsupervised-recommender-systems/43-recommender-systems-ii/visualizations/visualizations.py --module bpr
python 02-unsupervised-recommender-systems/43-recommender-systems-ii/visualizations/visualizations.py --module hybrid
```

The other module names are `matrix`, `dot-product`, `objectives`, `candidates`,
`metrics`, `cold-start` and `feedback`. Change fitted dimensionality through
`--factors 6`; use `--epochs 15` or `--frames 12` for a shorter run. Defaults
are **seed 43, three factors, 30 epochs, 24 sampled SGD frames**. Bounds are
2–79 factors, 1–100 epochs and 8–60 sampled frames. Matrix, metrics and onboarding
animations retain their concept-specific finite frame sets. `--frames` also
bounds feedback simulation rounds at 24. Thus it is not a global GIF frame count.

Outputs resolve against the script directory, even when invoked from another
working directory. `--output-dir outputs/my-run` creates a separate artifact set
under this visual folder. Reusing an output directory regenerates its named
visuals; dated JSON run records are retained. Avoid replacing the three selected
previews with exploratory configurations; use a different output directory.

```text
visualizations/
├── README.md
├── visual_core.py
├── visualizations.py
├── tests/
│   └── test_visual_lab.py
└── outputs/
    ├── images/     # PNGs; two selected public previews
    ├── gifs/       # Pillow GIFs; one selected public preview
    ├── html/       # self-contained interactive files; ignored
    └── records/    # dated metrics/configuration/artifact hashes; ignored
```

## What each scene teaches

Output filenames below are relative to `outputs/`. The four HTML files open
directly in a desktop browser; each embeds Plotly and its computed numerical
states. No HTTP server or internet connection is required. Native controls
update the visible calculations; controls do not pretend to retrain a model.

| Module | Question and interaction | Outputs |
|---|---|---|
| `matrix` | How does adding past rounds change sparsity and a low-rank approximation? Observed labels, SVD scores and observed-only residuals animate together. | `gifs/01_matrix_factorization.gif`, `images/01_matrix_factorization.png` |
| `embeddings` | Where do users/items move during training, and which eligible items have the highest full dot product? Select any of 80 users, epoch and K; rotate/zoom/hover. | `html/02_embeddings_3d.html`, `images/02_embeddings_preview.png` |
| `dot-product` | Why can dot, cosine and distance rank items differently? Select a toy 2D user and item; inspect the angle arc, norms, scores and both rankings. | `html/03_dot_product.html` |
| `bpr` | What does a real simultaneous update do to three vectors? Animate sampled steps, exact scores, pair loss and a fixed training pair's score history. | `gifs/04_bpr_training.gif`, `images/04_bpr_loss.png` |
| `objectives` | What changes between individual labels, pair comparisons and a distribution over a list? Compare computed BCE, pairwise softplus and top-one ListNet cross-entropy. | `images/05_ranking_objectives.png` |
| `hybrid` | Which source raises or lowers an item's score? Change user, K and α in 0.1 increments; inspect signed contributions and reordered items. | `html/06_hybrid_ranking.html`, `images/06_hybrid_preview.png` |
| `candidates` | Which relevant targets are lost before ranking? Change user, K and per-source budget 5/10/20/50; inspect real lists and aggregate metrics. | `html/07_candidate_pipeline.html`, `images/07_candidate_recall.png` |
| `metrics` | Why does rank 2 beat rank 8 under NDCG? Move a toy judged positive upward, then compare the five real policies at K=1/3/5/10/20. | `gifs/08_ranking_metrics.gif`, `images/08_model_comparison.png` |
| `cold-start` | What changes when a new user has 0/1/3/6/15 interactions? Fold in their vector against frozen items and include content candidates for a new metadata-only item. | `gifs/09_cold_start.gif`, `images/09_cold_start.png` |
| `feedback` | How can a click-count policy change exposure and future click counts? Compare uniform exposure with a declared popularity policy under stationary hidden click probabilities. | `gifs/10_feedback_loop.gif` |

The Matplotlib embedding thumbnail is computed from the same coordinates as the
HTML; it is a companion view, not a Plotly screenshot. The hybrid thumbnail
shows the same user-0/α=0.5 contributions as its HTML. No external image export
engine is needed.

## Shared experiment and interpretation boundaries

The main scenes reuse the study's deterministic synthetic generator: **80 users,
100 items, eight genres, 16 distinct events per user**, favorite-genre sampling
weight six versus one otherwise. Hidden favorites generate behavior but are
never model inputs. Observed events are positives; missing entries are unknown.
There are **no confirmed negatives or impressions in this main dataset**.
The weaker/stronger relationships come from the sampling weights, not a claim
that genre metadata exhausts real preferences.

All model fitting, profiles, popularity and source retrieval use only global
rounds **0–14**. Round **15** supplies one later target per user, leaving 85
eligible items. The matrix animation refits on prefixes of those past rounds.
Test targets only enter evaluation and post-retrieval labels; they do not alter
the candidate pool. Future positives can be sampled as unknown training items
because their future labels are not available at fitting time.

The policies are popularity, uncentered SVD, BPR, content profile cosine and
fixed-weight hybrid. Fusion uses per-user **eligible-catalog z-scores** before
candidate restriction, mapping standard deviation ≤ $10^{-12}$ to zero.
The educational default is

$$
s_{ui}=0.5z_u(s^{\mathrm{BPR}}_{ui})+0.5z_u(s^{\mathrm{content}}_{ui}).
$$

No visual slider setting is selected as a production optimum. Unlike the
historical study (12 factors, 60 epochs, validation-selected weight), the visual
run uses **three factors, 30 epochs, fixed α=0.5**. Do not merge their numbers or
treat comparisons on an already-inspected test round as fresh model selection.

The following special cases are deliberately separate:

- **Embedding projection:** three factors use their original axes; two factors
  are zero-padded. Above three factors, one PCA fit on combined final user/item
  factors supplies a fixed shared basis for all epochs. Centering and truncation
  can change distances and dot products. Hover/ranking scores always use the
  original full-dimensional factors. Norms affect compatibility; links are not
  assertions of Euclidean nearest-neighbor relationships.
- **BPR animation:** a separate actual **two-factor fit** on the same past-only
  matrix, with learning rate 0.05 and regularization 0.01. Each sampled update
  stores old/new vectors; all gradients use the old vectors. Plotted pair loss
  is $-\log\sigma(s_+-s_?)$, excluding regularization. The epoch curve averages
  this data loss over **all training positive/unknown pairs**. The fixed-pair
  curve follows a different declared probe, so it need not match each displayed
  triplet. Other SGD updates can worsen a particular margin; no per-step or
  held-out monotonic-improvement guarantee is made.
- **Loss illustration:** five fully judged toy candidates have explicit binary
  relevance labels. BCE, mean positive-negative softplus and top-one ListNet CE
  are executable loss calculations; this scene does not fit three additional
  models. NDCG is an evaluation metric, not the displayed listwise loss.
- **Metrics illustration:** a toy list has one known relevant item and nine
  judged irrelevant items. Moving the positive from rank eight to two raises
  binary NDCG@10 from $1/\log_2(9)$ to $1/\log_2(3)$ while Recall@10 stays one.
  The model-comparison chart separately averages over the main 80-user test.
  With one target/user, Recall@K equals Hit Rate@K. Neither unknown catalog items
  nor absent clicks thereby become confirmed irrelevance judgments.
- **Cold start:** remove user 0 from item-model fitting, then replay only their
  past events for a new user's fold-in (only their vector is updated). Append
  item 100 with fixed **genre-0 metadata**, independent of future onboarding
  events, and a zero collaborative vector. Content retrieval is actually
  included in the union. With no history, both signals are zero and the ID
  tie-break is uninformative. Coverage and score variation are not relevance
  accuracy; this separate probe has no cold-start outcome labels. A new item
  need not enter a finite candidate budget, even with content in the union.
- **Exposure:** a separate 12-item simulation draws 120 impressions/policy/round
  and clicks conditional on hidden stationary probabilities. Both policies
  start at ten impressions plus one declared pseudoclick/item. Uniform
  exposure is compared with 10% uniform exploration plus 90% proportional to
  squared cumulative clicks. Repeated policy updates can reinforce early
  differences; the exponent/exploration rate are assumptions, not empirical
  estimates. Observed click counts combine preference, exposure and randomness.

Dense exhaustive retrieval, scalar SGD and full-catalog normalization are
educational CPU calculations, not scalable serving implementations. The single
genre-driven seed favors content; there are no uncertainty estimates, realistic
impression logs, consented behavioral data or online/business outcomes. Passing
tests and attractive figures do not approve scientific interpretations.

## Recorded visual experiment — November 6, 2026

Codex executed all ten modules at **12:24 UTC (09:24 America/Sao_Paulo)**
with the default configuration above. The complete generation took **201.96 s**
on the available Windows CPU environment and produced **17 artifacts** (eight
PNGs, five GIFs and four HTMLs). Later individual regenerations corrected the
HTML title format/margins and the displayed completed-epoch endpoint; their
dated records retain the same numerical model configuration and metrics.

**Hypothesis:** inspect how real embeddings, score fusion and candidate
restriction change the opportunity to retrieve/rank held-out positives.

**Versions:** Python 3.11.14, NumPy 2.4.6, scikit-learn 1.9.1, Matplotlib 3.11.2,
Plotly 6.9.0 and Pillow 12.3.0. BPR uses normal initialization (standard deviation
0.1), shuffled positives, one uniform unknown draw per positive/epoch, learning
rate 0.05 and per-triplet regularization 0.01. SVD uses seven power iterations.

| Full-catalog policy | Recall@10 | NDCG@10 |
|---|---:|---:|
| Popularity | 0.062500 | 0.028825 |
| SVD | 0.187500 | 0.081079 |
| BPR | 0.187500 | 0.083501 |
| Content | 0.325000 | 0.185011 |
| Hybrid, α=0.5 | 0.325000 | 0.177081 |

| Candidates/source | Mean union size | Candidate recall | Final Recall@10 | Final NDCG@10 |
|---:|---:|---:|---:|---:|
| 5 | 9.4625 | 0.337500 | 0.337500 | 0.187227 |
| 10 | 18.2375 | 0.412500 | 0.350000 | 0.185296 |
| 20 | 34.3750 | 0.600000 | 0.325000 | 0.177081 |
| 50 | 68.1875 | 0.862500 | 0.325000 | 0.177081 |

**Interpretation candidates — author review required:** fixed α=0.5 fusion
matched content's hit rate and had lower NDCG in this run. Candidate recall
increased with the source budget; final recall did not increase monotonically.
The smaller candidate pool could remove competitors as well as targets, so
full-catalog NDCG is a reference, not an upper bound on restricted NDCG.
These are one-seed synthetic observations, not model superiority claims.
The separate genre-0 cold-item probe did not guarantee candidate admission;
the exposure simulation reflects its explicitly chosen click-count policy.
Neither probe establishes cold-start quality or production feedback effects.

## Validation and public previews

```powershell
python -m pytest -q 02-unsupervised-recommender-systems/43-recommender-systems-ii/visualizations/tests/test_visual_lab.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

The generator verifies file sizes, PNG/GIF decoding, frame counts and actual GIF
pixel changes. It writes a uniquely dated JSON under `outputs/records/` with
configuration, versions, hypotheses, model/candidate metrics, artifact SHA256
hashes, limitations and **author review required** interpretation status. GIF
frame changes alone do not verify the numerical mechanism: the focused tests
also compare instrumented SGD to the existing optimizer, recompute recorded
losses, alter held-out targets to test fitting isolation, and verify eligibility,
retrieval ceilings, objective arithmetic and shared PCA coordinates. Rendering
tests use temporary directories and never regenerate public previews.

The four HTML files need a real browser check in addition to Python tests.
Check user/epoch/K controls, dot-product item selection, hybrid weight and
candidate budget; rotate/zoom the 3D scene and inspect errors/network requests
with internet access disabled. Browser/GitHub appearance is separate from syntax
and path checks.

Local verification on November 6, 2026: **17 visual tests and 35 existing study
tests passed**, with no skips; syntax parsed 283 public Python files and the
link checker verified 905 targets in 199 Markdown files. Dependency compatibility
passed. All ten generators executed; all 17 artifacts were nonempty, decoded
where applicable and matched their latest recorded hashes. GIF frame counts
were 15 (matrix), 24 (BPR), seven (metrics), five (cold start) and 25 (exposure),
with actual pixel changes in each animation.

Chrome 156.0.8078.80 rendered the four main standalone HTMLs plus a separate
six-factor/two-epoch shared-PCA smoke run with network access disabled.
User/item/epoch/K, weight and budget controls changed the visible calculations;
3D mouse rotation, wheel zoom and a physical weight-slider click worked.
There were zero JavaScript exceptions and zero HTTP(S) requests. Screenshots,
browser checks and artifact audits remain ignored under `outputs/records/`.
The six-factor CLI also ran from outside the repository working directory.
GitHub rendering and production behavior were not validated; arbitrary parameter
combinations were not exhaustively tested. Git emitted only existing LF-to-CRLF
normalization notices for modified tracked text files.

Three compact assets are deliberately unignored and recommended for GitHub:

1. [Observed matrix, scores and masked residuals](outputs/images/01_matrix_factorization.png).
2. [Recorded BPR updates](outputs/gifs/04_bpr_training.gif).
3. [Hybrid component contributions](outputs/images/06_hybrid_preview.png).

All other PNGs, GIFs, HTMLs, browser screenshots and run records stay ignored and
regenerable. Standalone HTMLs embed a several-megabyte Plotly bundle; the epoch
explorer additionally embeds all users' recorded states. These local artifacts
are useful for interaction but are intentionally excluded from public previews.
No commit, push or publication is performed by the generator.

## Implementation references

The [study references](../references.md) cover BPR and SVD. Additional primary
sources checked for this visual phase:

- Cao et al. (2007), [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach/):
  top-one probability distributions and listwise loss. The five-item scene
  computes this loss; it does not reproduce the paper's training experiments.
- Plotly, [standalone HTML export](https://plotly.com/python/interactive-html-export/):
  embedded JavaScript for offline files. The lab uses `include_plotlyjs=True`
  and native controls over computed states.
- Matplotlib, [animation API and Pillow writer](https://matplotlib.org/stable/api/animation_api.html):
  GIF export without an FFmpeg process.
