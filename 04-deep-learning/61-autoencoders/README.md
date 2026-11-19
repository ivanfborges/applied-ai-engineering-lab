# Day 61 — Autoencoders

[Deep Learning index](../README.md) · [Roadmap](../../ROADMAP.md)

Can a model learn to reconstruct normal inputs and still miss anomalies?
An autoencoder maps an input through an encoder, a latent representation and
a decoder. Training asks for accurate reconstruction; anomaly detection adds
the separate assumption that relevant abnormal inputs reconstruct poorly.
This study makes that distinction executable on entirely **synthetic data**.

## What the examples establish

The NumPy example exposes both weight gradients in a linear bottleneck model.
Its training reconstruction is compared with the optimal rank-2 SVD/PCA
reconstruction on the same centered matrix. Equivalent reconstructions do
not require identical latent coordinates or encoder weights.

The PyTorch example trains a small nonlinear autoencoder on normal observations
only. PCA and Isolation Forest use the same training data and train-fitted
standardization. Each model receives an independent normal calibration set;
the test labels are used only for evaluation. Two injected anomaly mechanisms
contrast violations of the normal subspace with rare positions inside it.
This is useful for understanding telemetry or quality-screening proposals,
where an unusual observation and an operationally harmful event are different.

| File | Purpose |
|---|---|
| [from_scratch.py](from_scratch.py) | Explicit linear MSE gradients, simultaneous updates and SVD reference |
| [example.py](example.py) | Fixed-budget CPU autoencoder, independently calibrated baselines and subgroup metrics |
| [notes.md](notes.md) | Theory, assumptions, dated execution record, limitations and interview questions |
| [tests/test_autoencoders.py](tests/test_autoencoders.py) | Finite differences/autograd, PCA bound, invalid inputs, calibration and integration |
| [references.md](references.md) | Primary mathematical and library sources |

## Run from the repository root

Use stable Python 3.11+ and the dependencies declared in
[pyproject.toml](../../pyproject.toml). The deep-learning extra provides PyTorch;
the NumPy example does not require it.

```bash
python -m pip install -e ".[dev,deep-learning]"
python 04-deep-learning/61-autoencoders/from_scratch.py
python 04-deep-learning/61-autoencoders/example.py
python scripts/validate_repo.py tests --scope 04-deep-learning/61-autoencoders
```

The example prints a JSON record, including versions and configuration. An
optional local record can be saved with a fresh filename:

```bash
python 04-deep-learning/61-autoencoders/example.py --output 04-deep-learning/61-autoencoders/outputs/my-run.json
```

Existing output files are rejected rather than overwritten. `outputs/` remains
ignored; no generated asset is selected for public documentation. These two
analytical scripts perform no downloads, create no plots and require no credentials.

## Visual and interactive learning lab

The separate [visual guide](VISUAL_GUIDE.md) explains ten executable
demonstrations: encoder/decoder computation, rotating 3D manifold and actual
latent coordinates, feature residuals, real training checkpoints, calibrated
threshold controls, PCA/linear/nonlinear comparison, bottleneck seeds, capacity
and contamination, denoising, and feature scale. It uses the same installation
above, with Matplotlib/Pillow exports and actual CPU PyTorch training.

```bash
python 04-deep-learning/61-autoencoders/visualize_autoencoders.py --quick
python 04-deep-learning/61-autoencoders/visualize_autoencoders.py --all
python 04-deep-learning/61-autoencoders/visualize_autoencoders.py --demo latent --interactive
python 04-deep-learning/61-autoencoders/visualize_autoencoders.py --demo threshold --interactive
```

The [generator](visualize_autoencoders.py), [numerical experiments](visual_core.py),
[figure and widget implementation](visual_figures.py), and
[visual tests](tests/test_visual_autoencoders.py) keep measurement and rendering
inspectable. Exports use Agg without a display; interactive exploration uses
a supported local GUI backend (TkAgg by default). Output goes to a new ignored
run directory, with actual scores, labels, metrics and checkpoint predictions.
Close the GUI window to exit. No generated preview is currently versioned.

The visual protocol has a different architecture, data mixture and budgets
from the analytical example below. Its normal-like anomalies and shifted
legitimate normals deliberately probe observability and calibration failures.
Keep the [dated analytical record](notes.md#executed-study--2026-11-19) separate
from the [visual execution record](VISUAL_GUIDE.md#execution-evidence-and-preview-candidates).

## Read the scores correctly

For standardized inputs, each reconstruction score is the mean squared
residual across features. Isolation Forest uses the negative of `score_samples`
so that **larger means more unusual** for every method. Thresholds are computed
separately at the normal calibration 0.99 quantile with NumPy's `higher`
convention. A strict `score > threshold` triggers an alert; scores are not
probabilities, and the quantile is not a guaranteed future false-positive rate.

The [2026-11-19 execution record](notes.md#executed-study--2026-11-19) separates
observed metrics from interpretation candidates. The test contains 40 injected
anomalies among 400 observations; its precision and average precision do not
describe deployment prevalence. All configurations were fixed in advance,
with no architecture or threshold selection on the test set. This is a
single-seed mechanism exercise, not a general ranking of detectors.

## Key takeaways

- A bottleneck constrains representation; it does not guarantee meaningful,
  independent or identifiable latent coordinates.
- Standardization changes the reconstruction metric. Fit it on training data
  and preserve it together with the model and threshold for inference.
- PCA gives a strong reference when normal structure is linear. A nonlinear
  model must justify its additional complexity on the relevant task.
- Reconstruction, ranking and alert quality answer different questions.
  Report false positives and recall by anomaly mechanism, not just MSE or AUC.
- **Author review required:** automated checks and synthetic observations do
  not establish production usefulness or the author's approval of conclusions.
