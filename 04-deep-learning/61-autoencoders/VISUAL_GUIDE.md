# Autoencoders: a measured visual learning lab

This is a separate visual protocol from [the analytical study](notes.md).
It uses synthetic observations, actual CPU PyTorch fits, independent normal
validation/calibration, real checkpoint predictions and Matplotlib/Pillow.
**Interpretation candidates require author review.** Figures explain model
behavior under stated generators; they do not validate a production detector.

## Installation and execution

Use stable Python 3.11+. Dependencies and version bounds remain in the root
[pyproject.toml](../../pyproject.toml): NumPy, scikit-learn, Matplotlib and Pillow
are shared dependencies, and PyTorch comes from `deep-learning`. No new package
is required in the existing validated environment. No GPU, server, credentials,
downloads, FFmpeg or visualization framework is required.

From the repository root in a newly created/activated virtual environment:

```powershell
python -m pip install -e ".[dev,deep-learning]"
# Runtime-only compatibility entry point, also from the repository root:
python -m pip install -r 04-deep-learning/61-autoencoders/requirements.txt
```

Choose one installation command; the first includes pytest for the tests.
For a Windows CPU-only PyTorch installation, the shared repository setup also
documents the optional official CPU wheel index.

```powershell
$visualLab = '04-deep-learning/61-autoencoders/visualize_autoencoders.py'
python $visualLab --help
python $visualLab --list
python $visualLab --quick
python $visualLab --all
python $visualLab --all --output-dir outputs/day61
python $visualLab --demo latent --interactive
python $visualLab --demo threshold --interactive
# Shorter interactive fits; still actual training:
python $visualLab --demo threshold --quick --interactive --backend TkAgg
python scripts/validate_repo.py tests --scope 04-deep-learning/61-autoencoders
# Optional local GUI verification; uses TkAgg and synthetic widget events:
python 04-deep-learning/61-autoencoders/tests/gui_smoke_autoencoders.py
```

An explicit output directory must be new or empty. Relative explicit paths
are relative to the current working directory. Without that option, exports
go into a new timestamped directory under this topic's ignored
`outputs/day61/`. There is no overwriting, implicit publication or root README
update. All generated GIFs, PNGs, JSON and tensor checkpoints remain ignored.
No generated local asset is linked as though it were a public Git candidate.

Export mode selects `Agg` before importing pyplot. Interactive mode defaults
to `TkAgg`, supports a locally installed GUI backend via `--backend`, and
opens a real Matplotlib window. Tcl/Tk is an OS/Python component, not a pip
dependency. Close the window to exit. A PNG of a slider is only a reference;
the live slider is in the Python window.

## Protocol, data and replay

The numerical implementation is in [visual_core.py](visual_core.py), figures
and widget callbacks in [visual_figures.py](visual_figures.py), orchestration
in [visualize_autoencoders.py](visualize_autoencoders.py), and tests in
[tests/test_visual_autoencoders.py](tests/test_visual_autoencoders.py).
The opt-in [TkAgg smoke script](tests/gui_smoke_autoencoders.py) verifies GUI
canvas redraw and programmatic control events without becoming a mandatory
display-dependent CI test.

Quick mode runs all ten demos with 45 epochs (90 for geometry/comparison),
12 requested frames and two bottleneck initialization seeds. Full mode uses
180 epochs (360 for geometry), 36 frames and three initialization seeds.
Pillow merges identical successive frames into longer holds, so some GIFs
have fewer encoded frames while preserving their displayed states/duration.
No curve or model output is interpolated to invent learning progress.
Full mode is a larger protocol; quick results do not establish its outcomes.

All models use CPU, one PyTorch thread, Adam at 0.01, batches of 128, no weight
decay and the fixed final epoch. Epoch zero is recorded. NumPy and PyTorch
seeds are explicit; PyTorch deterministic algorithms are enabled by the CLI.
The command also limits OMP/MKL/OpenBLAS/NumExpr threads before imports.
These are resource settings, not a total RAM/CPU cap. Reproducibility across
library/hardware versions is not guaranteed.

The main normal generator is a noisy 2D linear subspace inside 12 dimensions:

$$
x=zB_{:,1:2}^{\top}+\epsilon,\quad
z\sim\mathcal{N}(0,I_2),\quad \epsilon\sim\mathcal{N}(0,0.12^2I_{12}),
$$

where $B$ is one seeded orthonormal basis, $z$ is a row of latent factors and
$\epsilon$ is independent feature noise. Training, validation, calibration
and test draw independent rows with shared generating geometry. Scaling is
fit on the designated clean normal training split only.

| Split/group | Quick count | Full count | Role |
|---|---:|---:|---|
| Clean normal train | 512 | 1,600 | Fit scaler and networks |
| Clean normal validation | 128 | 400 | Display loss; no selection or gradient updates |
| Independent normal calibration | 240 | 800 | Predeclared 95%, 97.5%, 99% thresholds |
| Ordinary normal test | 160 | 480 | In-distribution false alerts |
| Off-subspace anomaly test | 24 | 60 | Add $4B_{:,3}$ |
| On-subspace anomaly test | 24 | 60 | Add $6B_{:,1}$ |
| Normal-like anomaly test | 24 | 60 | Same observable generator as ordinary normals |
| Unusual legitimate normal test | 24 | 60 | Add $-4B_{:,2}$; label remains normal |

Quick test prevalence is $72/256=28.125\%$; full is $180/720=25\%$.
The normal-like anomaly label represents an external event absent from the
observed features: **there is no distinguishing signal by construction**.
The unusual normal group is a legitimate regime shift absent from clean
calibration. Its false alerts expose a representativeness failure. This
mixture intentionally stresses the normal-only calibration assumption;
it is not the analytical study's 10%-anomaly test or a deployment prevalence.
Gaussian support is unbounded; injected labels are experimental definitions.

Each new export includes `experiment_record.json` with configuration, versions,
split arrays, labels, predictions, scores, metrics, hypotheses and limitations.
`training_checkpoints.pt` contains actual base-model checkpoint state dictionaries
and probe outputs, saved as tensors/basic types so they load with
`torch.load(path, map_location="cpu", weights_only=True)`.
Only load tensor artifacts produced by a trusted local run.
Checkpoints are never optimized using the test probe: inference records its
reconstruction without gradients. Validation is independent but is not used
for early stopping. Interactive inspection does not authorize retuning on test.

For every detector, higher reconstruction MSE means more unusual. The policy is

$$
s_i=\frac{1}{d}\sum_{j=1}^{d}(x_{ij}-\hat{x}_{ij})^2,\qquad
\tau_q=Q_q^{\mathrm{higher}}(s(X_\mathrm{cal})),\qquad
\hat y_i=\mathbf{1}\{s_i>\tau_q\}.
$$

$d$ is feature count, $s_i$ an uncalibrated score, $q$ a predeclared quantile
and $X_\mathrm{cal}$ independent normal calibration rows. Threshold ties are
not alerts. Precision $=TP/(TP+FP)$, recall $=TP/(TP+FN)$ and
$F_1=2PR/(P+R)$; undefined precision/F1 with no alerts are displayed as zero.
Confusion matrices use true labels on rows and predicted labels on columns,
with normal first. ROC-AUC and AP describe ranking, not a single threshold.

## 01 — Encoder → latent space → decoder

**Concept.** A computation diagram connects 12 input values to 8 hidden
activations, 2 latent coordinates, 8 decoder activations and 12 outputs.
The GIF is `01_encoder_decoder.gif`.

**Mathematical foundation.** $z=f_\theta(x)$ and $\hat x=g_\phi(z)$.
Hidden layers use tanh; latent and output layers are linear. Node numbers
and colors are actual activations of the trained 12→8→2→8→12 model.
Highlighted layers illustrate evaluation order, not an instrumented timeline
of physical information movement. Edges show connectivity, not weight magnitude.

**What to observe.** Compare a normal and an off-subspace sample, numerical
latent coordinates, original/reconstructed bars and their actual MSE. Colors
saturate outside the fixed display range; numerical values remain visible.

**Interpretation.** The two coordinates constrain reconstruction, not every
possible semantic property of the input.

**Failure cases.** A neuron is not necessarily a concept. Diagram highlighting
does not show how long a matrix multiplication takes or prove interpretability.

**Real-world relevance.** A compact telemetry or document-quality representation
must preserve information needed by the intended task.

**Interview insight.** A bottleneck is a capacity constraint, not a guarantee
of disentanglement or useful anomaly features.

## 02 — Curved geometry and actual encoder coordinates

**Concept.** `02_manifold_3d.gif` rotates actual 3D points/reconstructions;
`02_latent_space.png` shows their learned 2D encodings.

**Mathematical foundation.** With independent $u,v\sim U[-1.8,1.8]$, the
clean surface is $(u,v,0.8(u^2+0.6v^2))$ plus isotropic noise SD 0.035.
OOD observations add 2.5 to the third coordinate. Train/validation/calibration
use normal draws; the normal test has 160 rows and OOD has 32. Fit the scaler
on 512 normal train rows in quick mode (1,600 in full), then train a
3→24→2→24→3 tanh AE. The displayed latent coordinates are $f_\theta(x)$,
not the known generating $(u,v)$ variables.

**What to observe.** Original dots, reconstructed crosses, sampled residual
segments, OOD overlap in learned coordinates and the faint *generating* surface.
3D panels use original synthetic units after inverse scaling. MSE uses
standardized units. Interactive sliders rotate both cameras and select a test
sample; Matplotlib also provides native drag-to-rotate for 3D axes.

**Interpretation.** A curved 2D structure can live in 3D. The fitted latent
space parameterizes a learned representation with no required semantic axes.

**Failure cases.** Reconstruction is not guaranteed to be nearest-point
projection onto the true surface. OOD points need not form a separate latent cluster.

**Real-world relevance.** Multisensor correlations can define a nonlinear
normal regime even when every marginal measurement looks plausible.

**Interview insight.** Intrinsic structure, architecture dimension and the
meaning of learned coordinates are different claims.

## 03 — Per-feature reconstruction errors

**Concept.** `03_reconstruction_error.gif` cycles through actual normal,
anomalous and unusual legitimate observations.

**Mathematical foundation.** Each bar on the right is $(x_j-\hat x_j)^2$;
their arithmetic mean is $s(x)$. The largest contribution is orange.

**What to observe.** Original and reconstructed standardized values, score,
the frozen 99% calibration threshold and the resulting decision. A missed
anomaly is added only when one actually exists in that run; selection is
descriptive after evaluation, never a tuning procedure.

**Interpretation.** Large residuals identify mismatched features under this
model and metric, not a causal explanation of an external incident.

**Failure cases.** A large residual does not establish fraud or failure;
normal-like anomalies may receive small scores. Per-feature error is not
feature attribution for a supervised event probability.

**Real-world relevance.** Residual decomposition can help a human investigate
an alert if the feature schema and units are meaningful.

**Interview insight.** Anomaly scoring and event diagnosis require different evidence.

## 04 — Real optimization checkpoints

**Concept.** `04_training_dynamics.gif` links training/validation losses,
a fixed sample's reconstructions and the same samples' evolving latent positions.

**Mathematical foundation.** Backpropagation differentiates the MSE through
both networks; Adam updates weights from those gradients. Displayed epochs
are actual saved state dictionaries and outputs, including initialization.

**What to observe.** The full measured loss path is faint context; stronger
curves reveal the prefix up to the current checkpoint. The fixed test sample
and latent probes never contribute gradients. All axes stay fixed.

**Interpretation.** Training changes a shared reconstruction map. Small latent
rotations/rescalings need not indicate a change in semantic meaning.

**Failure cases.** Smooth loss in one run does not establish monotonic
convergence. Test reconstruction shown for teaching cannot become a stopping rule.

**Real-world relevance.** Independent normal validation can expose poor fitting
or widening generalization gaps before alert calibration.

**Interview insight.** A loss trajectory is optimization evidence; held-out
anomaly metrics answer a different question.

## 05 — Thresholds, confusion matrices and operating points

**Concept.** `05_threshold_interactive.png` is a reference view;
`05_threshold_sweep.gif` cycles through independently calibrated 95%, 97.5%
and 99% policies. Run `--demo threshold --interactive` for real controls.

**Mathematical foundation.** Thresholded predictions use $s(x)>\tau$;
the ROC point is $(FP/(FP+TN),TP/(TP+FN))$ and the PR point is $(R,P)$.
All displayed counts and metrics are recalculated from actual fixed test scores.

**What to observe.** Move the threshold or press a calibration-quantile button.
Watch shaded decision regions, confusion matrix, precision, recall, F1 and
the marked ROC/PR operating points. Ranking AUC/AP stay fixed. Score-axis
compression is stated on the plot and does not change decisions.

**Interpretation.** The three buttons use normal calibration, whereas dragging
the slider is exploratory inspection of test outcomes. It must not be used
to choose the final deployed threshold on that same test.

**Failure cases.** Quantiles do not guarantee future false-positive rates;
rare legitimate regimes, sampling variability and drift can invalidate the
calibration population. Precision/AP depend on prevalence. Scores are not
probabilities. Better reconstruction does not imply better detection.

**Real-world relevance.** Thresholds determine human-review load and missed
cases; an operating policy needs costs, capacity and representative labels.

**Interview insight.** Changing a decision threshold changes operating metrics,
not the detector's ranking on unchanged observations.

## 06 — PCA versus linear and nonlinear autoencoders

**Concept.** `06_pca_vs_autoencoders.gif` cycles through three fitted methods
on both a noisy linear plane and the curved surface.

**Mathematical foundation.** The linear generator uses height $0.7u-0.4v$;
the nonlinear generator is the paraboloid above. Each dataset shares the same
scaler/rows across PCA, a bias-free linear 3→2→3 AE and a tanh 3→24→2→24→3 AE.
Training inputs are centered by StandardScaler. PCA uses full SVD; both AEs
use equal epoch/update budgets but unequal parameter counts.

**What to observe.** Inverse-scaled normal originals/reconstructions in 3D,
actual latent coordinates, and standardized normal held-out MSE. Normal test
rows alone define the displayed reconstruction comparison; OOD scores do not
select the model. All methods' training MSE is also retained in the JSON.

**Interpretation.** At a global optimum, a linear bottleneck with centered
inputs and squared reconstruction loss can attain the PCA training subspace.
Finite optimization need not attain it. The nonlinear advantage is an
experimental question; a linear AE can have slightly lower *test* MSE than
PCA without violating PCA's *training* optimum.

**Failure cases.** Encoder weights need not equal PCA loadings. Comparing
latent axes as semantic factors or claiming a universal nonlinear advantage
from this generator would be unsupported.

**Real-world relevance.** PCA is a useful simple reference before accepting
neural training and maintenance costs.

**Interview insight.** Equivalence of reconstruction subspaces does not imply
identical weights, optimization trajectories or held-out errors.

## 07 — Bottleneck size and initialization variation

**Concept.** `07_bottleneck_comparison.gif` contrasts $k\in\{1,2,4,8\}$.

**Mathematical foundation.** All networks are 12→8→$k$→8→12 with the same
normal training rows, optimizer, epochs and initialization-seed schedule.
Changing $k$ also changes parameter count. The 8-coordinate representation
is still undercomplete relative to 12 input dimensions. The constructor
supports overcomplete models, but the displayed sweep does not test them.

**What to observe.** Normal test MSE, test AP, score distributions and the same
on-subspace anomaly's actual reconstruction. Lines show individual initialization
seeds on one fixed dataset, not a confidence interval over datasets. Distribution
and reconstruction panels display the first initialization seed.

**Interpretation.** Extra representation capacity can help reconstruct normals,
anomalies or both. AP need not improve with normal MSE.

**Failure cases.** This is not a controlled equal-parameter architecture ranking,
a seed-tuning procedure or a reliable production optimum for $k$.

**Real-world relevance.** Choose capacity on a development protocol tied to
the downstream objective rather than reconstruction alone.

**Interview insight.** The representation objective and the detection objective
can reward different capacities.

## 08 — Factorial failure controls

**Concept.** `08_failure_modes.gif` compares low capacity (hidden 4, latent 1)
and higher capacity (hidden 32, latent 8), each clean and contaminated.

**Mathematical foundation.** Approximately 10% of training rows (51/512 quick,
160/1,600 full) receive a shift of 5 along the third right singular vector of
the clean standardized training matrix. This direction is computed from training,
not the test labels. The intervention differs from the raw off-subspace test
shift; it does not guarantee contaminating exactly the same anomaly mechanism.
Preprocessing stays frozen from clean training to isolate the network-training
intervention. Calibration and validation remain clean and independent.

**What to observe.** The matrix shows per-group alert fractions under each
model's own 99% calibration threshold. The scatter shows actual scores and
that threshold. For anomaly groups alert fraction is recall; for normal groups
it is false-positive rate. All displayed model parameters and AP are measured.

**Interpretation.** Capacity affects reconstruction and discrimination; contamination
can teach abnormal patterns, but its effect depends on the shift, architecture
and calibration. This specific intervention may not reduce detection.

**Failure cases.** Do not force a narrative of universal degradation if it is
absent. The normal-like group's external label has no observable signal;
the unusual-normal shift is a deliberate clean-calibration mismatch.

**Real-world relevance.** Vetted training data and representative legitimate
regimes matter as much as network size in alerting systems.

**Interview insight.** A detector can be numerically correct and operationally
wrong because the feature/label relationship or calibration population is wrong.

## 09 — Optional extension: denoising

**Concept.** `09_denoising.gif` compares a standard AE with a denoising AE,
using the same clean training signals and identical corrupted test inputs.

**Mathematical foundation.** Each clean signal has 12 ordered samples of
$a\sin(t+\phi)$, with independent random amplitude and phase. A train-fitted
scaler transforms clean train/validation/test. The denoising objective is
$\mathbb{E}_\eta\|x-g(f(x+\eta))\|^2$, where $\eta\sim N(0,0.5^2 I)$ is
generated anew per training batch in standardized units. The standard AE
uses $\eta=0$. Both are 12→16→2→16→12, use the same initialization seed and
training budget. Evaluation compares actual noisy-input predictions to clean
targets on 64 held-out signals.

**What to observe.** Clean target, noisy observation and each network's output
for the same sample. Displayed MSE averages all held-out signal elements.

**Interpretation.** Robustness is learned for the specific training corruption,
not imposed by smoothing plot coordinates.

**Failure cases.** Better Gaussian denoising does not establish anomaly detection
quality or robustness to missing fields, impulses or distribution shifts.

**Real-world relevance.** Sensor denoising requires a defensible corruption
model and an evaluation target that represents useful clean information.

**Interview insight.** Reconstructing the clean target changes the learning
problem compared with copying the corrupted input.

## 10 — Optional extension: feature scale

**Concept.** `10_feature_scaling.gif` uses per-feature scale multipliers from
1 to 100 on the same raw observations, then compares unscaled and standardized fits.

**Mathematical foundation.** Squared residuals scale as $c_j^2$ for feature
multiplier $c_j$. StandardScaler uses only clean training means/SDs. Both
models have the same architecture, initialization seed and epoch budget, but
the objective and conditioning differ. This is a controlled preprocessing
comparison, not an isolated causal estimate of feature importance.

**What to observe.** Left: each feature's contribution in its model's objective
units. Right: both reconstructions mapped back to the same original units.
Both panels use symmetric-log y axes so zero and large contributions coexist.
The objective units of the two left-hand states must not be equated.

**Interpretation.** Raw MSE can be dominated by large-scale features.
Standardization changes the metric and optimization landscape.

**Failure cases.** Equal standardized weights are not equal business importance;
raw errors across mixed units do not form a universal quality metric.

**Real-world relevance.** Currency, counts, confidence scores and latency in
one document-processing feature vector need an explicit reconstruction metric.

**Interview insight.** Scaling defines what an unsupervised reconstruction
objective considers a large error.

## Execution evidence and preview candidates

### Visual execution — 2026-11-19

**Executed hypothesis/configuration.** Codex ran `--quick` for all ten demos,
dataset seed 42, 45 epochs (90 for geometry/comparison), 12 requested frames,
and bottleneck initialization seeds 42/43 on the same dataset. The command used
the validated stable interpreter, without installing packages:

```powershell
$labPython = '.local/environments/n09-full-2026-11-13/Scripts/python.exe'
$visualLab = '04-deep-learning/61-autoencoders/visualize_autoencoders.py'
& $labPython $visualLab --quick --output-dir 04-deep-learning/61-autoencoders/outputs/day61/quick-2026-11-19-verified
& $labPython 04-deep-learning/61-autoencoders/tests/gui_smoke_autoencoders.py
```

The environment was Python 3.11.14, NumPy 2.4.6, scikit-learn 1.9.1,
PyTorch 2.15.0+cpu, Matplotlib 3.11.2 and Pillow 12.3.0 on Windows.
A first export into `quick-2026-11-19` exposed a colormap initialization warning
and layout issues. `quick-2026-11-19-final` fixed rendering; an extra replay
check then found NumPy arrays in its tensor checkpoint file incompatible with
`weights_only=True`. The command above uses a new directory after converting
checkpoint probe arrays to tensors. Original runs and analytical records
remain preserved; use the `verified` run for safe checkpoint replay.

**Observed base detection.** At the 99% calibration policy, ROC-AUC was
0.792874, AP 0.786635, precision 0.727273, recall 0.666667, F1 0.695652,
with TP/FP/FN/TN = 48/18/24/166. The confusion matrix was
`[[166,18],[24,48]]`. Calibration threshold was approximately 0.3603.

| Calibration policy | Precision | Recall | F1 | False positives |
|---|---:|---:|---:|---:|
| 95% | 0.592593 | 0.666667 | 0.627451 | 33 |
| 97.5% | 0.615385 | 0.666667 | 0.640000 | 30 |
| 99% | 0.727273 | 0.666667 | 0.695652 | 18 |

The base network alerted on all 24 off-subspace and all 24 on-subspace anomalies,
none of the 24 normal-like anomalies, 17/24 unusual legitimate normals and
1/160 ordinary normals. The combined normal test FPR was 18/184 = 9.783%,
despite calibration at 99%; the shifted legitimate group is a deliberate
representativeness failure. Unlike the earlier analytical ReLU AE, this visual
tanh AE flagged the on-subspace shifts. Architecture, population and budgets
all differ; these runs do not isolate the cause of that difference.

**Observed reconstruction comparison.** MSE uses standardized normal test rows:

| Dataset | PCA | Linear AE | Nonlinear AE |
|---|---:|---:|---:|
| Linear plane | 0.000441990 | 0.000495818 | 0.001166938 |
| Curved surface | 0.355140972 | 0.353007363 | 0.101162514 |

On curved *training* rows, PCA MSE was 0.290415063 and linear AE MSE
0.290505534. The slightly lower held-out error of the linear AE does not
contradict PCA's training optimum. The nonlinear model has more parameters;
this is no equal-capacity ranking.

**Observed bottleneck variation.** Individual initialization seeds on fixed data:

| k | Normal test MSE, seed 42 / 43 | AP, seed 42 / 43 |
|---:|---|---|
| 1 | 1.227649 / 1.231473 | 0.733005 / 0.725888 |
| 2 | 0.200419 / 0.196300 | 0.786635 / 0.786575 |
| 4 | 0.176404 / 0.155348 | 0.792771 / 0.795860 |
| 8 | 0.166549 / 0.151690 | 0.788898 / 0.792786 |

Reconstruction improved with k in these fits; AP fell from k=4 to k=8 in both
initialization seeds. Neither seed repetition nor these four settings defines
an optimal deployed bottleneck.

**Observed failure controls.** Low-capacity clean/contaminated AP was
0.740551/0.745125; high-capacity clean/contaminated AP was 0.797961/0.794339.
Every configuration missed the normal-like group and detected all off-subspace
cases. Contamination did **not** consistently reduce discrimination in this
intervention. The hypothesized general contamination-driven miss was not
observed. Unusual-normal alert fractions were 0.875, 0.875, 0.750 and 0.666667
for low-clean, low-contaminated, high-clean and high-contaminated models.

**Observed denoising.** On the same 64 corrupted test signals, clean-target MSE
was 0.121922 for the standard AE and 0.035220 for the denoising AE. This
observed gain applies to the matched Gaussian corruption used here.

**Observed scale contributions.** Unscaled normal-test per-feature MSE ranged
from approximately 0.176 to 1,370.466 in mixed raw squared units. In standardized
objective units it ranged from 0.0946 to 0.3419. These ranges describe different
metrics, not an improvement ratio. The right-hand panel restores both models
to raw units for a common reconstruction comparison. Neither fit establishes
appropriate feature weights for a business decision.

**Interpretation candidates — author review required.** The curved generator
favored nonlinear reconstruction in this run. The bottleneck sweep shows that
better normal reconstruction need not improve AP. Unobservable anomaly labels
and shifted legitimate regimes expose failures beyond numerical correctness.
The factorial contamination result resists a simple universal-degradation
narrative. None of these results establishes real-world detection quality.

**Artifacts and inspection.** All ten GIFs and both PNGs were generated. GIFs
contain 2–12 encoded frames after duplicate-frame coalescing (the network GIF
has 10). Representative states of every demonstration were inspected; dense
threshold, geometry, network and failure views were also checked individually.
The final export produced no warnings. The JSON contains actual predictions
for independent metric recomputation, and tensor checkpoints retain real states.
Every GIF frame was decoded successfully. The 12 final assets were byte-identical
to the individually/contact-sheet-inspected rendering run after the persistence-only
fix. Total GIF/PNG size was 3,203,301 bytes; numerical records are separate local
artifacts, not preview candidates.

**Final checks actually run.** Scoped repository validation passed 30 analytical
tests and 17 visual tests, without skips or warnings. The visual suite includes
real exports of every demo into pytest temporary directories, widget events,
calibration boundaries and safe checkpoint round-trip/replay. The final visual
file took 61.05 seconds; this is check duration, not a performance benchmark.
Separate checks recomputed score arrays for 15 fitted models, reproduced all
45 quantile-policy metric sets with scikit-learn, and replayed 12 disk-loaded
tensor checkpoints with exactly equal encodings/reconstructions.

```powershell
& $labPython scripts/validate_repo.py tests --scope 04-deep-learning/61-autoencoders
& $labPython scripts/validate_repo.py syntax
& $labPython scripts/validate_repo.py links
uv --no-cache pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
git diff --check
```

Syntax passed for 423 public Python files; internal paths passed for 1,507
links in 269 Markdown files; dependency compatibility passed for 78 packages;
diff whitespace checks passed. `--help`, `--list`, and refusal to overwrite a
populated output directory were exercised. Root EN/PT README hashes matched
the start of this visual phase. Public candidate/link checks do not establish
GitHub rendering, human scientific review, CI or full-curriculum test results.

**GUI scope.** Tcl/Tk 8.6.12 was available. An opt-in smoke ran on TkAgg with
actual 3D canvases, slider value changes, synthetic mouse press/release events
on a calibration button, updated confusion/metrics and canvas redraw. This is
programmatic GUI verification, not human mouse/keyboard acceptance. The opt-in
smoke uses 3-epoch models and is not the measured quick protocol above.
QtAgg, other operating systems, manual native 3D drag, and the full 180-epoch
CLI protocol were not tested. Human interpretation remains pending.

**Preview recommendations only.** After review, a small useful set would be
`01_encoder_decoder.gif`, `02_manifold_3d.gif`, `05_threshold_interactive.png`
and `08_failure_modes.gif`: they answer different questions about computation,
geometry, decisions and failure. Sizes are respectively 637,847, 978,384,
106,608 and 103,206 bytes (1,826,045 total, about 1.74 MiB), below the shared
per-file preview targets. All currently remain ignored. Publication would require intentional selection,
Git-candidate checks and documentation links. The static threshold PNG does
not substitute for the actual interactive program. No preview was unignored,
no commit/push was performed, and the root READMEs were left unchanged by this phase.

## Primary implementation references

- [Matplotlib widgets](https://matplotlib.org/stable/api/widgets_api.html): retained Slider/Button objects and change callbacks.
- [Matplotlib backends](https://matplotlib.org/stable/users/explain/figure/backends.html): GUI and headless backend distinction.
- [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html): export without FFmpeg.
- [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html): tensor state dictionaries and `weights_only=True` loading.
- [Topic mathematical references](references.md): reconstruction, PCA and calibration conventions.
