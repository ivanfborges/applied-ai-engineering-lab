# Day 34 visual suite: run and measurement record

This companion adds visual experiments to the existing study without rewriting
its theory. Run from the repository root:

```bash
python 01-classical-machine-learning/34-missing-data/visualize_missing_data.py
python -m pytest -q 01-classical-machine-learning/34-missing-data/tests
```

Or change into this topic and run `python visualize_missing_data.py`.
Outputs always resolve relative to the script. Existing named outputs are
overwritten; unrelated files are preserved.

Runtime dependencies are NumPy, scikit-learn, Matplotlib, Pillow, and
threadpoolctl (installed with scikit-learn). Plotly is optional and produces
two self-contained HTML files with its JavaScript embedded. Missing Plotly
prints an installation command and skips only those views. The shared
[dependency configuration](../../pyproject.toml) already supplies these packages;
no separate topic dependency manifest is needed. Execution uses no internet,
GPU, notebook, Streamlit, browser server, or backend. A browser is needed only
to inspect local HTML; browser automation is not a runtime dependency.

[visualize_missing_data.py](visualize_missing_data.py) handles rendering.
[missing_data_visual_core.py](missing_data_visual_core.py) holds testable
data generation, masks, pipelines, and measurements.
[tests/test_visual_missing_data.py](tests/test_visual_missing_data.py) checks
mechanism identities, immutable truth, progressive masks, variance algebra,
training statistics, frozen-model fitting, reproducibility, and optional Plotly.

## Asset inventory

All names below are under `outputs/`. Only three selected previews are
deliberately unignored; other assets are generated locally.

| No. | File | Main question | Policy |
|---:|---|---|---|
| 01 | `01_mcar_mar_mnar.png` | Which regions lose values under each mechanism? | Public preview |
| 02 | `02_missingness_probability.png` | What changes the probability of absence? | Regenerable |
| 03 | `03_missingness_process.gif` | How does a stochastic mask accumulate? | Regenerable |
| 04 | `04_imputation_distribution_distortion.png` | Where does constant filling concentrate a skewed distribution? | Regenerable |
| 05 | `05_imputation_geometry.png` | Where do filled points move relative to hidden truth? | Public preview |
| 06 | `06_correlation_distortion.png` | Does retaining rows preserve covariance and correlation? | Regenerable |
| 07 | `07_missing_indicator.png` | Why can absence expose a latent predictive regime? | Regenerable |
| 08 | `08_imputation_leakage.png` | Which records influence the fitted median? | Regenerable |
| 09 | `09_performance_vs_missingness.png` | What happens when models refit at higher matched missingness? | Regenerable |
| 10 | `10_missingness_shift.png` | What happens when frozen models lose more test features? | Public preview |
| 11 | `11_feature_dropout_robustness.png` | What happens when entire features become unavailable? | Regenerable |
| 12 | `12_missingness_3d.html` | How does selection look after rotating three correlated dimensions? | Regenerable, optional Plotly |
| 13 | `13_imputation_3d.html` | How far do mean/KNN estimates move from true positions? | Regenerable, optional Plotly |

The optional robustness GIF 14 is omitted because it duplicates the stochastic
animation and quantitative curves. The existing `comparison.json` belongs to
the earlier example. The suite separately writes `visual_experiments.json`
with raw scores, realized missing rates, summaries, configuration, versions,
hypotheses, interpretation candidates, and limitations. Both reports stay
ignored. No intermediate review screenshots are stored in this output folder.

## Configuration and information boundaries

The controlled dataset has age-, income-, risk-, and behavior-like features.
Income is lognormal and related to age and behavior. Risk also depends on age
and behavior. The binary outcome is a stochastic Bernoulli draw from the
specified logistic probability in the generator, not a deterministic class.

The mechanism figures use 2,400 complete rows (smaller display subsets), seed
42, and hide income only. MCAR probability is 0.30. MAR probability is
`0.05 + 0.75 sigmoid((age - 45)/5)`, with age always observed. MNAR probability
is `0.05 + 0.75 sigmoid((log(true_income) - 3.65)/0.20)`, evaluated before hiding.
The same uniform draws make masks comparable across mechanisms. Showing hidden
true positions is an explicitly labeled synthetic reference, not recoverable
observed information.

Distribution and geometry figures use same-sample descriptive imputers.
They do not estimate held-out predictive accuracy. Geometry uses 320 correlated
points, seed 43, requested MCAR probability 38%, and seven KNN neighbors.
Distribution diagnostics use MNAR on the skewed income feature.

For prediction, all splits precede fitting. Performance sweeps use 2,000 rows,
stratified 70/30 splitting, and seeds 11, 22, 33, 44, 55. All four workflows
share histogram boosting: 65 iterations, nine leaves, minimum leaf size 20,
learning rate 0.1, L2 penalty 1, no early stopping, and the repeat seed.
KNN uses five donors after observed-training-value standardization.

Requested MCAR rates apply independently to all four feature columns.
Actual fractions are stored in the report. Reusing uniform draws creates
nested masks across rates. At each matched rate, every workflow refits on
training data only. Under serving shift, each workflow fits once at 10% and
reuses the same complete test rows and labels through 60% missingness.
No test condition chooses or tunes a model. Shading is one **sample standard
deviation across five datasets/splits/masks**, not a confidence interval.

Feature dropout trains on complete data, then freezes the training-median
fallback and learner before hiding entire test columns. The indicator
experiment uses a separate unobserved binary regime that influences target
propensity and measurement availability; it never uses realized labels to
construct masks. The intentionally pooled imputation exists only in the
statistic-only leakage figure; it never feeds a predictive experiment.

## Executed experiments and measurements

The hypotheses were that availability changes can alter frozen-model ranking,
imputation changes geometry and distributions, and an indicator can reveal a
latent collection regime. The recorded configurations above precede scoring.
All results below are computed by the suite, not benchmark claims.

### Matched train/test missingness: refit at each rate

| Missing rate | Median | Median + indicator | Scaled KNN | Native NaN |
|---:|---:|---:|---:|---:|
| 0% | 0.7302 +/- 0.0240 | 0.7302 +/- 0.0240 | 0.7302 +/- 0.0240 | 0.7302 +/- 0.0240 |
| 5% | 0.7131 +/- 0.0178 | 0.7132 +/- 0.0167 | 0.7199 +/- 0.0206 | 0.7133 +/- 0.0227 |
| 10% | 0.7086 +/- 0.0151 | 0.7093 +/- 0.0153 | 0.7165 +/- 0.0200 | 0.7118 +/- 0.0173 |
| 20% | 0.6916 +/- 0.0181 | 0.6950 +/- 0.0163 | 0.6934 +/- 0.0237 | 0.6931 +/- 0.0158 |
| 30% | 0.6798 +/- 0.0152 | 0.6795 +/- 0.0173 | 0.6739 +/- 0.0176 | 0.6756 +/- 0.0175 |
| 40% | 0.6714 +/- 0.0181 | 0.6747 +/- 0.0223 | 0.6575 +/- 0.0128 | 0.6692 +/- 0.0258 |
| 50% | 0.6491 +/- 0.0186 | 0.6496 +/- 0.0191 | 0.6318 +/- 0.0278 | 0.6499 +/- 0.0156 |

Entries are mean ROC-AUC +/- one sample SD over the five fixed seeds.

### Serving shift: frozen models trained at 10%

| Missing rate | Median | Median + indicator | Native NaN |
|---:|---:|---:|---:|
| 10% | 0.7086 +/- 0.0151 | 0.7093 +/- 0.0153 | 0.7118 +/- 0.0173 |
| 20% | 0.6934 +/- 0.0144 | 0.6945 +/- 0.0137 | 0.6949 +/- 0.0194 |
| 30% | 0.6777 +/- 0.0183 | 0.6756 +/- 0.0176 | 0.6787 +/- 0.0160 |
| 40% | 0.6670 +/- 0.0156 | 0.6654 +/- 0.0152 | 0.6639 +/- 0.0169 |
| 50% | 0.6510 +/- 0.0158 | 0.6505 +/- 0.0151 | 0.6503 +/- 0.0137 |
| 60% | 0.6255 +/- 0.0114 | 0.6297 +/- 0.0124 | 0.6342 +/- 0.0147 |

Entries are mean ROC-AUC +/- one sample SD over the five fixed seeds.

### Entire-feature dropout

| Unavailable features | Mean AUC | Mean change | SD of paired change |
|---|---:|---:|---:|
| All available | 0.7302 | +0.0000 | 0.0000 |
| Age unavailable | 0.7258 | -0.0044 | 0.0056 |
| Income unavailable | 0.7154 | -0.0149 | 0.0180 |
| Risk unavailable | 0.4849 | -0.2453 | 0.0226 |
| Behavior unavailable | 0.7325 | +0.0023 | 0.0085 |
| Risk + income unavailable | 0.4255 | -0.3048 | 0.0332 |

### Indicator and information-flow examples

Indicator experiment held-out ROC-AUC: **0.7581 without**, **0.8262 with** the indicator (720 test rows, one engineered seed-42 scenario).

Leakage figure: training median **-0.0867** versus pooled median **2.0630**. No predictive metric is claimed for the intentionally leaky example.

Installed versions: python 3.11.0rc2, numpy 2.4.6, sklearn 1.9.0, matplotlib 3.11.0.

## Interpretation candidates: pending author review

- The matched-rate and frozen-model curves show lower mean AUC at the largest
  tested rates than at their starting conditions for this generator. Local
  changes and method rankings need not be monotone or stable.
- In the engineered latent-regime example, adding an indicator improved AUC.
  Its usefulness is tied to that regime; a changed collection policy could
  remove the association.
- Mean/median filling visibly collapses hidden points, while KNN varies locally.
  KNN's high reconstructed correlation also illustrates loss of residual
  uncertainty, not proof that the joint distribution is recovered.
- Full risk dropout caused a substantially larger loss than age dropout here.
  Behavior dropout slightly improved the mean. Those measurements illustrate
  why a failure's effect should be evaluated rather than assumed.
- The leaked median shows information crossing a split boundary; the figure
  makes no claim about a performance advantage.

Review the generator's realism, the meaning of each simulated outage, the
adequacy of ROC-AUC, and whether an operational fallback would meet error-cost
and calibration requirements. This is a small synthetic exercise with five
prespecified repetitions, not deployment validation or general method superiority.
MAR/MNAR are known by construction and generally cannot be conclusively
distinguished from observed real data alone.

## Verification actually performed

The suite executed end to end twice; the complete numerical JSON records
reproduced byte-for-byte. PNG files were decoded and inspected. The GIF contains
26 distinct frames, loops over 7.75 seconds, and pauses at the final state.
Both HTML files rendered in a real browser with networking disabled and no
external requests; their dropdowns and camera controls worked. The Day 34 tests,
repository syntax, publication-aware internal links, and whitespace checks are
reported separately in the implementation handoff.
