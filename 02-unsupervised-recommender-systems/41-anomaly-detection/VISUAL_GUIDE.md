# Day 41 — Offline Anomaly Detection Visual Lab

Seven visual questions connect numerical mechanisms, synthetic labels and alert
decisions. Each HTML includes five short explanations: what is visible, the
internal mechanism, mathematics, controls and a limitation. Interpretation
candidates remain **author review required**.

## Run locally

Use stable Python 3.11+ to match this repository's supported environment.
Dependencies are already declared in [pyproject.toml](../../pyproject.toml):
NumPy, SciPy, scikit-learn, Matplotlib, Plotly and Pillow are used here. No new
dependency file, Streamlit server, cloud account, browser automation package,
FFmpeg or external dataset is needed to export.

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --quick
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --all
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --section isolation_forest
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --quick --section one_class_svm
```

From this topic directory, the equivalent entry point is
`python visual_lab.py --all`. Use `--help` for supported section names and
`--output-dir PATH` for another destination. Invalid combinations and missing
dependencies produce explicit errors. Figures export with Matplotlib's `Agg`
backend and PillowWriter. The generator never launches a browser.

Open `outputs/visual_lab/full/index.html` or `outputs/visual_lab/quick/index.html`
directly in a browser. Every plot HTML embeds Plotly.js and works without a
network connection; embedding the runtime explains the approximately 5–7 MB per
HTML. A modern WebGL-capable browser is needed for the 3D surface. The supplied
index coordinates access to all figures; it does not require a local server.

Asset filenames are regenerable and can be overwritten by subsequent runs in
the same destination. Timestamped `run-*.json` records retain hypotheses,
configuration, versions, observed results, limitations and artifact hashes.
`catalog.json` merges completed sections for the local index. Quick and full
profiles use separate default directories; a custom directory cannot mix profiles.

## Scope and code

- [visual_lab.py](visual_lab.py): CLI, bounded profiles, execution records and index.
- [anomaly_visual_core.py](anomaly_visual_core.py): datasets, actual-tree replay,
  covariance distances, parameter fits and chronological reference updates.
- [anomaly_visuals.py](anomaly_visuals.py): Plotly pages and Pillow GIF renderers.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): numerical, timing,
  reproducibility, metric, export and CLI contracts.

The lab reuses the tested [educational forest](from_scratch.py) and score/metric
helpers in [example.py](example.py). Study scripts and the historical nonvisual
experiment record remain separate from these visual runs. Current curriculum
progress is recorded in the [module index](../README.md).

| Setting | Quick | Full |
|---|---:|---:|
| Normal telemetry train / calibration / test | 240 / 100 / 100 | 600 / 240 / 220 |
| Relationship / extreme test injections | 12 / 8 | 25 / 15 |
| Surface grid | 35 × 35 | 65 × 65 |
| Shared telemetry IsolationForest trees / samples | 50 / 128 | 160 / 128 |
| Normal SVM ring training rows | 120 | 260 |
| Educational ensemble trees | 30 | 80 |
| Robust-statistics GIF frames | 12 | 24 |
| Isolation GIF frames | 23 | 29 |
| Monitoring GIF frames, including required event instants | 23 | 44 |

All data are synthetic. Base seed 141; ring seed 142; chronological stream seed
143. Quick mode retains every section and the same mathematical behavior, with
smaller populations/grids or fewer frames. Its metrics are not comparable to
the full population or the historical seed-41 study as if they were repeated
measurements on the same data.

## 1. Contamination of a statistical baseline

`01_robust_statistics.gif` appends up to ten increasingly extreme latency values
to a fixed normal sample of 100 rows. The scatter, vertical mean/median lines,
one-scale intervals and numeric captions show location and dispersion together.
The second panel tracks standard deviation against normal-consistent MAD.

`01_robust_zscores.html` selects contamination states and displays absolute
classical versus robust standardized deviations. The two panels have different
fixed Y ranges, explicitly labeled; compare numeric scores rather than heights.
The factor $1/\Phi^{-1}(0.75)\approx1.4826$ makes MAD consistent for Gaussian
scale. It does not make data normal or turn a 3.5 heuristic into a guaranteed
error rate. Zero MAD raises a clear error in the core and is tested.

**Try:** compare the final added latencies against the dashed heuristic line.
**Candidate — author review required:** extreme values can inflate their own
classical scale estimate and mask their standardized deviation. Contamination
count and severity both change here; this is not an isolated causal estimate.

## 2. Unusual relationships

`02_relationships.html` contrasts a featurewise median ± three robust scales
rectangle with a squared Mahalanobis-distance contour. Both baselines are fitted
only on normal training observations. Toggle boundaries through the legend.
The star is a separate constructed point with marginal deviations $(1.5,-1.5)$,
excluded from metric populations.

**Try:** inspect the star inside the rectangle but outside the tilted contour.
**Candidate — author review required:** featurewise tests can miss joint
deviations. The contour value 9.21 has a Gaussian population reference; estimated
mean/covariance, non-Gaussian data, multiple regimes or singular covariance
invalidate a simple exact-tail interpretation. Covariance here is ordinary,
normal-only covariance, not an automatically robust estimator.

## 3. Real isolation paths

`03_isolation_tree.gif` replays all 14 actual recursive splits in one educational
tree on 40 rows. Every line ends at its parent-cell boundary. The current cell
and participating observations are highlighted. Green/orange rings track the
typical and isolated points; the right panel shows partial depth, then terminal
leaf corrections. The last phase averages corrected paths over independent
randomized trees generated by the same local RNG protocol.

The original score is $s(x)=2^{-\bar h(x)/c(\psi)}$, with
$c(n)=2H_{n-1}-2(n-1)/n$ for $n\ge2$ and $c(0)=c(1)=0$.
An unsplit multirow leaf adds $c(n)$ to its depth. The forest uses exact harmonic
sums; scikit-learn uses optimized trees and an approximation. Numerical identity
between them is not asserted.

`03_isolation_scores.html` switches actual scikit-learn fits with samples 16/40
and trees 25/100. This particular outlier-detection demonstration includes the
two isolated rows in training; all other normal-only novelty examples retain
separate fitting and evaluation sets. Training never uses anomaly labels.
Library scores are negated `score_samples`; `decision_function` also subtracts
the estimator offset and therefore has a different zero boundary.

**Try:** compare the same location across all four score maps.
**Candidate — author review required:** shorter mean paths identify these
isolated points, but a single tree need not do so. A failure inside a dense
normal region or a failure requiring absent context can still receive a low score.

## 4. RBF support and feature units

`04_one_class_svm.html` precomputes 18 configurations: raw units versus a
training-fitted RobustScaler, gamma 0.03/0.3/3, and nu 0.01/0.05/0.15. Black
zero contours, support-vector circles and scalar fields come from the actual
fitted models. Point hover values are evaluated directly.

**Try:** compare scaled gamma 0.03 against scaled gamma 3 at the center hole;
then hold numeric gamma fixed and switch to raw units. Higher gamma localizes
the RBF kernel. A coarse mesh can miss narrow raw-unit inlier islands; use point
scores and support markers rather than treating every interpolated contour as
an exact boundary. Nu constrains training errors/support fractions in the
formulation, subject to solver tolerance; it does not specify future alert rates.
**Candidate — author review required:** scaling and bandwidth jointly define
the geometry; flexibility alone does not establish useful detection.

## 5. Native-score landscapes

`05_score_landscape.html` switches MAD, Isolation Forest and scaled One-Class
SVM surfaces on the same telemetry population. Observed synthetic rows sit at
their directly evaluated scores. Drag to rotate, wheel to zoom and hover.

All scores are oriented higher = more anomalous. No normalization is used and
the Z axis rescales per model. **Candidate — author review required:** each
method defines a different geometry of unusual behavior. Absolute heights and
color intensities across detector surfaces are not calibrated probabilities or
evidence that one model is more confident.

## 6. Ranking versus action

`06_thresholds.html` links score distributions, TP/FP/FN/TN point categories, a
precision–recall curve and a metrics table through one coordinated selector.
Blue TN, orange TP, purple FP and black-cross FN distinguish decisions from
synthetic truth. Each histogram is normalized within its own group; these bars
do not encode the operational base rate.

For each model, normal-only calibration uses quantile levels 0.95/0.98/0.995 with
`method="higher"`; alerts use strict score > threshold. These controls never
refit the scoring model. AP is constant per score ranking, whereas precision,
recall, FPR and burden depend on the chosen cutoff. The future FPR is not
guaranteed by the empirical calibration setting.

Two additional actual IsolationForest fits differ only in contamination
0.01/0.10, keeping identical seed/tree settings. Their built-in zero decision
boundaries give exactly the same AP but different alerts. They deliberately use
training-derived offsets, a separate protocol from held-out normal calibration.
**Candidate — author review required:** identical ranking can support very
different operational policies; the model and cutoff need separate assessment.

## 7. Chronology, incidents and reference changes

`07_monitoring.gif` progressively reveals a 360-row stream: stable operation,
incident at 180–187, benign mean drift at 240–329 and a changed plateau from 330.
The panels show latency, fixed/adaptive scores and thresholds, and the most
recent 40 available latencies against the historical fitting distribution.
`07_monitoring.html` offers fixed/recalibrated policy selection over the full
timeline. Orange marks are known synthetic incident truth; open purple circles
in HTML are policy alerts. These are different concepts.

Initial fitting uses [0,80), calibration [80,120), evaluation [120,360).
At 300, the approved reference fits [240,272) and calibrates [272,288).
At 330, it fits [270,302) and calibrates [302,318). All intervals are half-open.
No selected window extends past activation; the approval delay is at least 12
observations. Scripted operator approval excludes the earlier known incident
by window selection, without filtering by generated labels. This is an
illustrative incident-gated scenario, not an autonomous labeling mechanism.

**Try:** compare fixed and updated policies after 300 while observing the
remaining benign-shift alerts. **Candidate — author review required:** changing
a reviewed past reference can reduce alerts on accepted workload changes, but
small, already-drifting calibration windows remain unstable. Blind latest-data
retraining can absorb incidents into normality: the new reference estimates then
describe a mixture of healthy and incident behavior, potentially reducing the
contrast of later failures. The GIF's scores before 120 are retrospective
reference diagnostics shown after calibration; operational alerts begin at 120.
This stream demonstrates data
drift; it does not establish concept drift, seasonality handling or root-cause diagnosis.

## Dated visual measurements — November 6, 2026

Codex executed quick and full profiles on Windows x64 with Python 3.11.14,
NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, Matplotlib 3.11.2, Plotly 6.9.0
and Pillow 12.3.0. The initial full record completed at 10:19:08 UTC; subsequent
section exports preserve separate timestamped records. These are new visual
scenarios, not replacements for the historical seed-41 study measurements.

Observed full-profile results, with hypotheses and configuration described above:

- Final contamination: mean 335.349 ms, median 249.268 ms, std 281.420 ms,
  robust scale 21.777 ms.
- Constructed relationship star: marginal deviations ±1.5 and squared
  Mahalanobis distance 19.1553.
- Selected typical/isolated point corrected mean paths over 80 trees:
  8.4303 / 2.3875; original scores 0.4102 / 0.7769.
- Contamination-only IF pair: AP 0.851006 for both; 16 versus 52 alerts.
- Monitoring on [120,360): fixed/adaptive policies generated 82/52 alerts,
  including all eight injected incident rows and 74/44 benign-drift alerts.
  These labels are generator truth, not a measured production incident rate.

Interpretation candidates above remain pending author review. No external data,
resource benchmark, production deployment or automatic gating quality is established.

## Validation and preview policy

```powershell
python -m pytest -q 02-unsupervised-recommender-systems/41-anomaly-detection/tests
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Numerical tests cover real partition membership, bounds, corrected paths,
train-only scaling, score orientation, policy metrics, reproducibility, identical
ranking with different offsets, and future/prefix invariance. Export smoke tests
write to temporary directories. The full and quick generators were also executed;
all GIF frames were decoded and representative frames inspected.

The final local topic suite passed **54 tests**, with no skips or warnings.
Repository syntax parsing passed for **269 Python files**, and internal-link
checks passed for **848 targets in 189 Markdown files**. Dependency compatibility
and `git diff --check` passed. The complete repository test suite was not run.

The browser audit used installed Chrome via CDP, with networking disabled:
seven plots plus index, 62 coordinated control states, pointer-operated slider,
dropdowns and boundary toggle, plus 3D rotation/zoom. It observed zero JavaScript
errors and zero remote requests. Playwright CLI was unavailable through the
restricted npm access; no npm package was installed. Browser automation is
validation-only and is not a generator dependency. CI, browser portability and
GitHub math rendering require separate verification.

All generated HTMLs, GIFs, run records and inspection screenshots remain ignored.
No asset is linked here as a public Git candidate. Recommended small previews
after author review: the real-tree GIF, contamination GIF and monitoring GIF
(approximately 0.31, 0.30 and 1.13 MiB in full mode). Deliberately unignore only
selected previews before adding public Markdown asset links. The seven standalone
HTMLs and index total about 36 MiB; regenerate locally instead of versioning all
embedded runtime copies.
