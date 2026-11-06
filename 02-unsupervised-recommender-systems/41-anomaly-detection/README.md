# Day 41 — Anomaly Detection

When does unusual telemetry deserve an alert? This study separates the model's
definition of unusual behavior from the threshold used to act on it. It compares
robust featurewise deviations, random isolation and a learned nonlinear support
boundary in a normal-only novelty-detection workflow.

| Method | Score used here: higher means more anomalous | Important boundary |
|---|---|---|
| Robust MAD | Maximum absolute robust deviation across features | Does not model feature dependence; zero MAD raises an error |
| Isolation Forest | Negative `score_samples` | Axis-aligned isolation need not capture every unusual relationship |
| RBF One-Class SVM | Negative `decision_function` after training-fitted RobustScaler | Kernel geometry depends on scaling, gamma and the reference population |

An anomaly score is evidence for investigation, not an incident probability or
a root-cause diagnosis. Point, contextual and collective anomalies require
different representations; independently scoring rows does not model time.

## Run from the repository root

Use stable Python 3.11+ and the canonical dependencies in
[pyproject.toml](../../pyproject.toml). No additional extra or dataset download is needed.

```bash
python -m pip install -e ".[dev]"
python 02-unsupervised-recommender-systems/41-anomaly-detection/example.py
python 02-unsupervised-recommender-systems/41-anomaly-detection/from_scratch.py
python -m pytest -q 02-unsupervised-recommender-systems/41-anomaly-detection/tests/test_anomaly_detection.py
```

The example prints metrics and writes a timestamped JSON record under ignored
`outputs/`, including configuration, versions, hypothesis, measurements and
pending interpretation. `--output PATH` selects an explicit destination and
overwrites that destination if it exists. Keep new runs separate from old records.
The scratch script prints a duplicate-only reference invariant, not a benchmark.
The separate visual lab now adds seven offline learning sections; its commands
and evidence are documented below. The study experiment remains a separate run.

## What is implemented

- [example.py](example.py): IID synthetic latency/queue telemetry, normal-only
  fitting and calibration, held-out labels, AP and threshold-policy metrics.
- [from_scratch.py](from_scratch.py): NumPy MAD, quantile calibration and a small
  isolation forest with subsampling, depth limits and terminal-leaf correction.
  It uses exact harmonic sums and local RNG state, and does not promise numeric
  agreement with scikit-learn's optimized trees and path-length approximation.
- [tests/test_anomaly_detection.py](tests/test_anomaly_detection.py): hand-worked
  paths, SciPy MAD agreement, duplicates, invalid inputs, reproducibility, score
  signs, fit boundaries and frozen-score policy behavior.
- [notes.md](notes.md): mathematics, assumptions, monitoring decisions and the
  dated experiment record.
- [interview_questions.md](interview_questions.md): operational and conceptual Q&A.
- [references.md](references.md): primary documentation and the isolation paper.

## Interactive visual lab

The [visual guide](VISUAL_GUIDE.md) explains contamination/masking, correlated
anomalies, actual random isolation paths, RBF gamma/nu and scaling, native-score
3D landscapes, threshold policies and chronological monitoring with past-only
scripted reference updates. Each HTML includes controls and mathematical limits.

```bash
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --quick
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --all
python 02-unsupervised-recommender-systems/41-anomaly-detection/visual_lab.py --section isolation_forest
```

Open the generated `outputs/visual_lab/full/index.html` directly in a browser
(or `quick/index.html` under the same directory). Seven standalone Plotly HTMLs,
three explanatory GIFs and timestamped run records remain ignored and regenerable.
No browser/server is required for export. Dependencies remain centralized.

The [CLI](visual_lab.py), [numerical scenarios](anomaly_visual_core.py),
[renderers](anomaly_visuals.py) and [visual tests](tests/test_visual_lab.py) are
public source candidates. Quick/full visual measurements use seed 141 and differ
from the nonvisual seed-41 experiment below; do not merge their metric populations.
Visual interpretation candidates also require author review.

## Experiment and evidence boundary

The generator uses 800 normal training rows, 200 separate normal calibration
rows, and 300 normal plus 40 injected test rows. Latency and queue depth have a
correlated Gaussian baseline; injections change either their relationship or
both magnitudes. These are artificial continuous features with telemetry names,
not measured requests or a chronological production replay. Test labels never
enter fitting or calibration. Hyperparameters are fixed, not selected on test data.

For each frozen detector, calibrate three normal-reference tail settings
($\alpha=0.05,0.02,0.005$), then evaluate the same held-out test population.
AP evaluates ranking; precision, recall, false-positive rate and alert count
evaluate a decision policy. Raw scores cannot be compared across models.

The [November 6, 2026 record](notes.md#executed-experiment--november-6-2026)
shows why a higher AP need not mean higher recall at a particular calibrated
threshold. All interpretation candidates require **author review**. One seed,
clean normal reference data and a deliberately favorable geometry cannot establish
a general model ranking or operational benefit.

For monitoring, define the unit of observation and actionable incident first,
then validate features, score orientation and alert policy. Temporal deployment
needs chronological evaluation, context/seasonality handling and incident feedback.
