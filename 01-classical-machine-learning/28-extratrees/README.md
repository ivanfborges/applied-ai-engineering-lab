# ExtraTrees and Tree Ensemble Variants

ExtraTrees explores a useful question: **can weaker individual split choices
produce a better ensemble?** Random Forest searches for strong thresholds
within random feature subsets. ExtraTrees proposes random thresholds and
chooses the best available candidate. Bootstrap sampling is a separate choice.
See the [official ensemble guide](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees).

Additional randomness can trade individual tree strength for diversity. That
makes ExtraTrees a useful nonlinear tabular baseline, including for metadata
classification or quality scoring inside larger AI systems. Its value must be
evaluated against the task's metric, validation boundary, and resource budget.

## What this study makes executable

The library example uses **synthetic binary classification data**, one shared
stratified split, and a four-way comparison: Random Forest and ExtraTrees, each
with and without bootstrap. It measures ensemble accuracy, ROC-AUC, log loss,
average individual-tree accuracy, pairwise hard-label disagreement, and fit
time. Comparing RF and ET with bootstrap fixed separates threshold strategy
from row sampling; comparing bootstrap settings within one family studies row
sampling.

The scratch implementation grows recursive binary trees on the full training
set. It exposes feature proposals, uniform threshold draws, Gini selection,
leaf frequencies, and probability averaging. Its XOR truth-table demo checks
mechanics on training rows and supplies no generalization evidence.

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Split mathematics, variance assumptions, practical limitations, and the executed experiment record |
| [example.py](example.py) | Four controlled scikit-learn configurations on synthetic data |
| [from_scratch.py](from_scratch.py) | Educational finite-data binary ExtraTrees ensemble |
| [tests/test_extratrees.py](tests/test_extratrees.py) | Random candidate selection, recursive constraints, averaging, disagreement, repeatability, and invalid inputs |
| [interview_questions.md](interview_questions.md) | Questions about strength, diversity, evidence, and deployment choices |
| [references.md](references.md) | Original research and official documentation |
| [visualize_extratrees.py](visualize_extratrees.py) | Static, animated, and optional interactive mechanics companion |
| [tests/test_visualize_extratrees.py](tests/test_visualize_extratrees.py) | Numerical identities, undefined correlations, prefix averaging, PNG rendering, and CLI checks |

## Run from the repository root

Dependencies are centralized in [pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/28-extratrees/example.py
python 01-classical-machine-learning/28-extratrees/from_scratch.py
python -m pytest -q 01-classical-machine-learning/28-extratrees/tests
```

The library script prints results and writes its configuration, environment,
and measurements to `outputs/comparison.json` relative to the script.
That generated record is ignored and overwritten on subsequent runs; the
public measured summary lives in the [notes](notes.md#executed-synthetic-comparison).
No external data, notebook, or visual assets are required.

## Key takeaways

- ExtraTrees still selects splits using the target; randomness changes the
  candidates rather than removing supervised learning.
- More disagreement is useful only alongside sufficiently strong trees.
  Disagreement on one held-out set does not measure variance over training sets.
- The executed comparison produced different rankings by metric. Its
  interpretation remains **pending author review**.
- Probability averaging and hard majority voting can yield different classes.
- The scratch model supports labels 0/1 and finite dense arrays only. It omits
  bootstrap, OOB, weights, pruning, missing-value handling, and optimized split
  search. It is an educational implementation, not a library replacement.

## Visual companion

Run the complete companion from the repository root:

    python 01-classical-machine-learning/28-extratrees/visualize_extratrees.py

NumPy, scikit-learn, Matplotlib, and Pillow provide the standard run. Plotly
adds the optional offline 3D surface. Dependencies are already declared in
the shared project configuration. Use `--skip-html`, `--skip-animations`,
`--output-dir PATH`, or `--dpi 80` to adjust generation.

The script creates the following **regenerable artifacts** under the topic's
ignored `outputs/` directory. No generated output is selected as a public
preview or linked as a versioned documentation asset.

| Output | Visual question |
|---|---|
| `01_split_search_vs_random.png` | How does exhaustive midpoint search differ from a small random proposal pool? |
| `02_random_threshold_search.gif` | How does proposal quality vary, and when is best-so-far updated? |
| `03_decision_boundaries.png` | How do DT, RF, and ET partition the same synthetic moons? |
| `04_individual_extratrees.png` | How different are six component ExtraTree partitions? |
| `05_tree_correlation_heatmaps.png` | How do disagreement and hard-prediction correlation compare on the same validation rows? |
| `06_ensemble_variance_surface.png` | Under equal variance/correlation assumptions, why does a variance floor remain? |
| `06_ensemble_variance_surface.html` | Rotate and inspect that analytical surface offline. |
| `07_n_estimators_stability.png` | How do measured AUC and seed-to-seed prediction variation change with nested tree counts? |
| `08_diversity_vs_performance.png` | How does feature access affect individual strength and ensemble ranking quality? |
| `09_sources_of_randomness.png` | What changes with bootstrap fixed, or with threshold strategy fixed? |
| `10_tree_ensemble_mechanics.png` | Which features, thresholds, and rows can differ in each algorithm? |
| `11_ensemble_growth.gif` | How does the actual ensemble boundary evolve across seven prefixes? |

The threshold GIF has 24 frames. Its 1D best-so-far pool is a teaching
illustration: actual ExtraTrees normally draws one threshold per candidate
feature at a node, rather than accumulating many proposals for one feature.
The mechanics schematic shows that rule explicitly. The boundary GIF has
seven frames; it uses actual prefixes of fitted 300-tree ensembles.

`outputs/visual_experiments.json` records hypotheses, configurations, actual
results, interpretation candidates, and limitations. Rerunning overwrites
the generated files. All empirical interpretations remain **pending author
review**.

### Executed visual experiments

Executed by Codex on 2026-10-06. These results belong to this visual lab and
use different configurations from the earlier library example.

The geometry/diversity experiment uses 400 synthetic moons, noise=0.23, a
stratified 70/30 split, seed 42, 80 trees, depth=5, minimum leaf size=3,
max_features=1, and bootstrap=False in both ensembles. The evaluation set
has 120 rows.

| Model | ROC-AUC | Mean tree accuracy | Disagreement | Mean defined prediction correlation |
|---|---:|---:|---:|---:|
| RF | 0.9804 | 0.8818 | 0.1228 | 0.7598 |
| ET | 0.9833 | 0.8131 | 0.2492 | 0.5717 |

**Interpretation candidate -- author review required:** ET had weaker members
and greater hard-label disagreement here, alongside slightly higher ensemble
AUC. Its log loss was worse (0.3175 versus RF's 0.1938), so ranking and
probability quality did not improve together. Three ET trees had constant
hard predictions on validation rows: the correlation mean uses 2,926 defined
pairs out of 3,160; RF uses all 3,160. Undefined cells remain blank.

The feature-access and bootstrap experiments use 1,000 synthetic
make_classification rows with 10 features (4 informative, 2 redundant, 4 noise),
class_sep=1.0, flip_y=0.05, and a stratified 70/30 split, seed 42. They use
80 trees, depth=6, and minimum leaf size=3. The generator's remaining explicit
settings are in the script and JSON record.

At max_features="sqrt", the bootstrap comparison measured:

| Model | Bootstrap | ROC-AUC | Log loss | Disagreement |
|---|---|---:|---:|---:|
| RF | False | 0.9254 | 0.3596 | 0.1680 |
| RF | True | 0.9231 | 0.3663 | 0.2078 |
| ET | False | 0.9152 | 0.4827 | 0.3175 |
| ET | True | 0.9105 | 0.4764 | 0.3225 |

**Interpretation candidate -- author review required:** ET showed more
disagreement and lower AUC at both matched bootstrap settings in this tabular
experiment. This supports inspecting strength and diversity together rather
than selecting the configuration with maximum disagreement.

The stabilization view averages five model seeds (42-46) on one fixed moons
split, using measured prefixes of 1, 2, 5, 10, 20, 50, 100, 200, and 300 trees.
Mean prediction SD across seeds fell from 0.1395 to 0.0099 for RF and from
0.1983 to 0.0100 for ET between the first and last prefixes. No smoothing or
monotonic improvement is imposed.

**Review limits:** these are untuned synthetic experiments. Cross-row
prediction correlations do not estimate error correlation or prediction
variance across new training datasets. The seed experiment isolates
algorithmic randomness with training rows fixed. The 3D variance surface is
analytical, not fitted to those heatmaps. Timings are local single-run
observations, and the feature sweep establishes no generally optimal setting.
Review these boundaries and the teaching simplifications before publishing
selected previews. The offline HTML embeds Plotly and is larger than the PNGs;
keep it regenerable unless a public delivery strategy is chosen.
