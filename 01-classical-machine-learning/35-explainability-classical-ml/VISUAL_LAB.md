# Interactive explainability learning lab

The [Streamlit app](app.py) makes the measurement process visible: node impurity
reduction, frozen-model shuffling, replacement prediction averaging, individual
response curves, and coalition-based attribution. It is a CPU educational lab
for senior ML interviews, rather than an explanation service.

## Windows and VS Code setup

From the repository root, use Python 3.11+ and the shared environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,explainability]"
cd 01-classical-machine-learning/35-explainability-classical-ml
streamlit run app.py
```

Select the repository's .venv interpreter in VS Code. Reuse an existing
environment. Dependencies remain centralized in
[pyproject.toml](../../pyproject.toml): NumPy, pandas, SciPy, scikit-learn,
Matplotlib, Plotly, Streamlit, Pillow, SHAP and pytest. No GPU, credentials,
external dataset, API, browser image-export engine or extra framework is needed.
Installation needs network access; the installed lab works locally without it.
The shared minimum Streamlit version is 1.60.

Generate exports independently of Streamlit, from the topic directory:

```powershell
python export_gifs.py
python -m pytest -q tests
```

A suggested alternative configuration, **not an executed comparison**:

```powershell
python export_gifs.py --samples 900 --rho 0.985 --interaction 2.5 --fps 6
```

Outputs default to this topic's outputs directory regardless of the current
working directory. The --output-dir argument can select another location.

## Training versus inspection

The sidebar's **Apply dataset and forest** form batches sample size (300–2,000),
population correlation rho (0–0.995), interaction gamma (0–3), tree count
(20–120), maximum depth (2–12), and leaf minimum (1–20). The applied
configuration remains active until submitted again.

Training is cached by the full applied configuration. Feature, observation,
grid position, centering and plot mode do not refit the model. Metric/repeat
changes recompute cached frozen-model importance; SHAP background changes
recompute cached attributions. Lazy tabs compute only the visible tab. All
caches have bounded entry counts.

SHAP backgrounds use 10–300 training rows, limited by available training size.
After applying a smaller dataset, an oversized active background is reduced
to the largest supported choice. Local explanations cover up to 80 held-out
rows; PDP/ICE use up to 100; 3D PDP averages 60.

## Synthetic data and definitions

With independent standard normals s, c, epsilon and noise:

$$
x_{\mathrm{signal}}=s,\quad
x_{\mathrm{proxy}}=\rho s+\sqrt{1-\rho^2}\epsilon,\quad
p=\sigma(1.4s+0.9c+\gamma sc),\quad y=\mathbf{1}(U<p).
$$

rho controls population correlation while preserving unit proxy variance;
gamma controls the generator interaction; sigma is the logistic function;
U is a seeded uniform draw. Sample correlation need not equal rho exactly.
Noise has no direct generator term. A stratified 75/25 split and seed 35
define a fixed RandomForestClassifier. Explanations use **P(y=1)** or an
explicitly named score calculated from it.

This generator differs from [example.py](example.py), which uses an unscaled
signal-plus-noise proxy and another label-sampling sequence. Its earlier
[experiment records](notes.md) remain separate from the visual lab.

## Explore the eight tabs

| Tab | Visual question and controls | Mathematical boundary |
|---|---|---|
| Overview & Dataset | Inspect input geometry, labels, correlation, prevalence, ROC-AUC and sample rows. | Generator probability differs from learned prediction. |
| Tree Importance / MDI | Move a split, play a threshold sweep, inspect Gini/counts/stump and forest importance. Optional random-label control contrasts continuous and binary noise. | MDI accumulates training split credit; many split positions can fit chance. |
| Permutation Importance | Play synchronized original/shuffled panels; change feature, shared-pair shuffle, metric and repeats. | Score decrease is frozen-model reliance; interpolation is illustrative transport. |
| PDP & 3D Surfaces | Inspect replacement inputs, prediction histogram, mean and PDP. Rotate mean/fixed surfaces; inspect slices and weak/strong generator interactions. | Marginal averaging differs from conditional expectation and from fixed nuisance inputs. |
| ICE | Select a row, show more curves and center at an anchor. | The mean identity requires matching rows/grid/output. |
| SHAP | Follow both product-game orders; then select waterfall, beeswarm, dependence or background comparison. | Attribution depends on the reference game, effective background, class and scale. |
| Explainability Pitfalls | Compare correlated reliance, known joint support and an invalid post-outcome field. | An explanation cannot repair label leakage. |
| Summary & Interview Review | Pair each question with its units and limitation; review interview prompts. | Arithmetic fidelity does not establish causal or policy validity. |

Each tab includes defined equations, axis/color descriptions, “What you are
observing,” “Common misinterpretation,” and “Production engineering takeaway.”
Plotly supports hover, zoom and rotation. Each chart's camera exports a
high-resolution PNG. Dedicated 200-dpi Matplotlib downloads cover importance,
ICE/PDP and probability SHAP without Kaleido.

The beeswarm is a deterministic Plotly rendering packed by nearby attribution
bins. Feature-value colors normalize independently per feature; vertical
packing has no numerical meaning. Fixed surfaces hold the selected row's
proxy/noise. Actual sample dots use their own nuisance inputs and need not
lie on that fixed slice. These are prediction/PDP surfaces, not SHAP surfaces.

## Regenerable exports

| Artifact | Calculation made visible |
|---|---|
| permutation_importance.gif | Signal values move to a seeded shuffled assignment; predictions and ROC-AUC update. Only the endpoint is an actual permutation. |
| pdp_ice_construction.gif | Individual curves appear, then the mean of exactly those 30 curves is traced. |
| shap_contributions.gif | Empty/single/full coalitions in both orders, marginal contributions, then averaging to (3,3) for f(2,3)=6. |
| importance_comparison.png, pdp_ice.png, shap_waterfall.png | 200-dpi scientific snapshots of the export configuration. |
| partial_dependence_3d.html | Self-contained rotating/hoverable Plotly surface, with Plotly embedded for offline use. |
| visual_lab_results.json | Hypothesis, configuration, versions, results, frame metadata, stability checks, interpretation candidate and limitations. |

GIF axes are fixed and checked during every frame update. Identical frames
may be coalesced by the encoder, retaining their display duration. Default
GIF sizes are approximately 87–877 KiB; the offline HTML is about 4.9 MB.
A rotating 3D GIF is omitted because interactive rotation already conveys
the surface without adding a rendering dependency.

All exports are **ignored regenerable artifacts**. No preview was unignored
or linked as a public asset. Inspection-frame PNGs in outputs/inspection
are ignored intermediate artifacts.

## Executed export record

**Hypothesis:** visual transport, matching ICE averaging and feature-order
averaging should demonstrate different definitions while satisfying their
numerical identities.

**Configuration:** seed 35; 900 rows; rho 0.95; gamma 1.6; stratified 75/25 split;
60 trees, depth 6, leaf minimum 5, sqrt feature sampling, bootstrap, one worker.
The animation shuffles signal once. The importance comparison uses eight
repeats. GIF PDP averages 30 reference curves on 25 grid values. Forest SHAP
explains 80 held-out rows against 50 training rows sampled with seed 35.
Export rate is six frames per second, with deliberate holds.

**Result:** baseline ROC-AUC 0.839181; animated endpoint ROC-AUC 0.632764;
decrease 0.206417. PDP-minus-mean-ICE error was zero. Product Shapley values
were (3,3), reconstructing six. Forest probability SHAP maximum reconstruction
error was 4.69e-08. The joint signal/proxy scratch mean score decrease was
0.285848 in the separate eight-repeat estimate.

**Interpretation candidate — pending author review:** the checked identities
support the specified arithmetic. Group reliance and apparent heterogeneity
still require interpretation with dependence, support and reference assumptions.

**Limitations:** one synthetic configuration and seed; eight permutation
repeats; one SHAP background; explanation samples smaller than the full test
set. Transport is not a repeated-permutation experiment. No benchmark,
confidence interval, causal effect or stability claim is established.

## Reliability and first experiments

The support diagnostic uses the **known synthetic** distribution
proxy|signal ~ Normal(rho*signal, 1-rho^2). Outside its 95% band does not mean
impossible; inside does not guarantee realistic joint inputs. Replacing a
correlated feature can violate dependence despite a percentile grid.

SHAP explicitly uses interventional marginal replacement, selects class 1
from the fitted classes, checks output shape and verifies the sum within 1e-6.
An explicit Independent masker preserves backgrounds above 100 rows. This
differs from observational conditional SHAP and is not a causal estimate.

The leakage demonstration sets a post-outcome field equal to y in training
and evaluation. Its apparent metrics are invalid for the prediction-time task.
The optional cardinality control uses its own 600-row dataset, 60/40 split,
50 unrestricted-depth trees and eight AUC permutations. Downloadable records
state these configurations and retain pending-review interpretations.

Suggested, **not recorded as completed comparisons**: increase rho and compare
individual/group reliance; remove gamma and inspect centered ICE on probability
and logit scales; switch metrics; keep the observation fixed while changing
SHAP background sizes/seeds; repeat across model seeds and cohorts. Record
hypotheses, configurations, results and limitations before conclusions.

For interviews, state the **question, units, population and assumptions**.
MDI is not held-out reliance; permutation is not retraining; PDP is not
generally conditional expectation; SHAP magnitude is not score loss.
Explanation fidelity does not establish fairness, safety or justification.

