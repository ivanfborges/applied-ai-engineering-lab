# CNN Visual Learning Lab — Day 59

This local CPU laboratory makes array operations, spatial geometry and actual
learning visible. All images are synthetic. There are no dataset downloads,
pretrained models, external APIs or cloud services. The visual experiment is
separate from the earlier [16×16 analytical study](notes.md#executed-study--2026-11-18).

## Run it

Use a stable Python 3.11+ environment. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,deep-learning]"
python 04-deep-learning/59-cnns/example.py --generate-visuals
python -m streamlit run 04-deep-learning/59-cnns/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
python scripts/validate_repo.py tests --scope 04-deep-learning/59-cnns
```

On Linux/macOS activate with `source .venv/bin/activate`. If an environment is
already configured, use it rather than replacing it. The repository's local
workflow selects an existing stable interpreter explicitly; it does not use
or modify the older rollback environment. Dependencies and supported ranges
remain in [pyproject.toml](../../pyproject.toml), not a second requirements list.
For a CPU-only PyTorch installation, see [validation.md](../../docs/validation.md).

The generation command trains both architectures with two seeds and writes a
fresh timestamped collection to the ignored topic `outputs/visual-lab/run-*`.
Use `--seed 59`, `--visual-epochs 20` or `--visual-output-dir PATH` to configure a
new run. A nonempty output directory is rejected. Existing reports and previews
are preserved. `--epochs` remains the separate analytical study's option.
The generator uses one CPU thread, bounded datasets and six short GIFs; no GPU
is required. HTML exports embed Plotly's runtime and work offline with rotation,
zoom and hover. Serve them through a local HTTP server if a browser restricts
local-file WebGL behavior.

To refresh exports from an existing measured run without training again:

```powershell
python 04-deep-learning/59-cnns/example.py --generate-visuals --visual-report PATH/TO/experiment.json
```

The saved report's configuration controls a cached rendering refresh;
`--seed` and `--visual-epochs` configure newly trained runs only.

The app does **not** train on load. It reads the latest generated record, or
shows a request to run the experiment in the sidebar. The explicit button uses
a deterministic, bounded cache; the same configuration can return the same
measured record. The displayed record's seeds/epochs are shown independently
of controls for the next requested run. Model-dependent GIFs are offered only
when their export record matches the displayed experiment. Core mechanism
animations do not depend on training.

`python -m pytest -q 04-deep-learning/59-cnns/tests` is an accurate direct
alternative for this topic. Bare repository-root pytest covers Foundations
only. Export tests use temporary directories and a reduced frame/sample budget;
they do not overwrite selected previews or rerun the full visual experiment.

## Code map

- [visual_core.py](visual_core.py): synthetic scenes, patch products, RGB
  contributions, zero-filled translations and exact structural support masks.
- [experiments.py](experiments.py): actual 32×32 CNN/MLP training, serializable
  checkpoints, paired split/order protocol and stress evaluation.
- [visualizations.py](visualizations.py): Matplotlib/Pillow GIFs and PNGs,
  Plotly 3D/filter charts, immutable run collections and artifact manifests.
- [app.py](app.py): native Streamlit controls and the same figures used by exports.
- [tests/test_visual_lab.py](tests/test_visual_lab.py): numerical oracles,
  framework agreement, checkpoint restoration, real exports and AppTest flows.

The existing [from_scratch.py](from_scratch.py) remains the shared educational
cross-correlation/pooling implementation. No second convolution implementation
is hidden in rendering code.

## Read the visualizations

Every app section includes **what you see, operation, parameter changes,
observations, practical use and limitations**. Image axes use row/column
indices. Signed quantities use blue/red; input/activation magnitudes use a
sequential map. Yellow marks the current patch or potential input support.
Gray output pixels are uncomputed or excluded, never fabricated zeros.
Raster cell labels are rounded to at most two decimals for readability;
calculations, JSON records and interactive hover retain numerical precision.

### 01 — Sliding cross-correlation

`01_convolution_sliding.gif` has 36 actual window positions for an 8×8 step
image, a 3×3 vertical detector and a 6×6 valid output. Six synchronized panels
show the image window, weights, extracted patch, element-wise products,
row/total sums and progressively filled feature map. Positive, negative and
zero weights contribute directly to the displayed sum. The app selects each
position independently.

$$
Y_{i,j}=\sum_{u,v}X_{i+u,j+v}W_{u,v}.
$$

The kernel is not reversed, matching PyTorch cross-correlation. There is no
bias or ReLU in this mechanism. These handcrafted weights explain arithmetic;
they do not represent observed learned semantic detectors.

### 02 — Filters and signed feature maps

`02_filters_comparison.png` compares vertical/horizontal edges, sharpening,
blur, identity and a fixed random filter on the same scene. Each row contains
input, numeric kernel and response. `02_filters_interactive.html` supplies a
native Plotly filter menu, also available in the app. Response scales are shared
across filters; kernel scales are shared separately. Small blur responses can
look muted beside large edge/sharpening responses because units are preserved.
No ReLU or per-panel normalization hides the sign or relative magnitude.

Boundary responses include zero padding 1. Strong edges are contrasts, not
object probabilities. A CNN learns coefficients rather than selecting from
this menu of handcrafted filters.

### 03 — RGB convolution and channel sums in 3D

`03_rgb_convolution_3d.html` shows an input of shape 3×8×8, weights of shape
2×3×3×3, and two 6×6 output maps. The selected output filter spans all three
input channels. Hover identifies `out`, `in`, `u`, `v` and numerical values.
The annotation gives three measured channel contributions plus the selected
bias at the selected output pixel:

$$
Y_{o,i,j}=\sum_{c=0}^{2}\sum_{u,v}X_{c,i+u,j+v}W_{o,c,u,v}+b_o.
$$

The output-filter menu changes all three kernel slices and the contribution
equation together. The app additionally selects output row/column. The other
output map remains visible at lower opacity to explain multiple filters.
Input R/G/B are channel planes; z is **channel index with display spacing**,
not activation magnitude. Horizontal offsets separate input, weights and
outputs; colors encode values. This is a fixed illustrative calculation, not
a trained RGB model.

### 04 — Sampling grids, padding and dimensions

`04_stride_padding.gif` varies stride/padding and sampled windows. The app
controls input size, kernels 3/5/7, stride 1/2/3 and padding 0/1/2/3. All window
centers are marked on the padded grid, along with the real-image boundary and
one selected patch. The output uses a normalized box filter so kernel-size
changes do not simply multiply coefficients by a larger count.

$$
O=\left\lfloor\frac{N+2P-D(K-1)-1}{S}+1\right\rfloor,\qquad D=1.
$$

Invalid dimensions produce an explicit message. More padding changes border
averages; it does not add observed information. More stride changes the
sampling grid and can discard detail. The example handles square kernels,
symmetric zero padding and floor mode; it does not simulate general padding,
dilation or an anti-aliasing filter design.

### 05 — Max versus average pooling

`05_pooling_comparison.gif` visits all sixteen 2×2 windows of an 8×8 map with
stride 2. Input, max output and mean output share one value scale. The selected
four values and both calculations are printed, and unknown output cells stay
gray. A sparse peak and negative values make information loss visible.
Neither operation has weights; no padding is used. Pooling is not guaranteed
to preserve arbitrary translations, locations, small defects or fine text.

### 06 — Receptive-field growth and exact boundary support

`06_receptive_field_growth.gif` follows Input → Conv3 → Pool2 → Conv3 → Conv3.
The nominal receptive-field spans are 1, 3, 4, 8 and 12, with jumps 1, 1, 2, 2
and 2. The app selects a layer and neuron, then traces its structural support
backward through each layer. Convolutions have padding 1 and pooling has none.
Near a border, real-image support is smaller than the nominal span.

$$
r_l=r_{l-1}+(k_l-1)j_{l-1},\qquad j_l=j_{l-1}s_l,\qquad r_0=j_0=1.
$$

This three-convolution geometry example is distinct from the two-convolution
trained classifier. The highlighted mask is **potential structural support**.
Effective sensitivity can be inspected through input gradients, but depends on
the sample, weights, nonlinearities and output being differentiated. No
effective-gradient map or causal interpretation is claimed here. Tests use
positive linear convolutions and average pooling to independently activate
every structural path and check the support mask against autograd.

### 07 — Equivariance, sampling and classifier confidence

`07_translation_equivariance.gif` displays original/shifted noisy bars,
stride-1 edge responses, aligned differences and actual classifier probabilities
for the first vertical bar in the first seed's held-out set. This selection is
defined by the label before inspecting predictions. The app can choose another
test example. The equality check compares $C(Tx)$ with $T(Cx)$ only
where both original and shifted 3×3 windows are interior. Outside that mask,
the difference is gray. Shifts use zero fill, not circular wraparound.

The app also evaluates both trained classifiers at the currently selected
integer shift and compares global means before/after stride-1/stride-2
convolution and max/average pooling. Those handcrafted feature summaries are
separate from the classifiers' logits. Odd shifts do not have an exact integer
output-grid alignment after stride 2. The CNN includes pooling and a global
mean; its finite classifier is not guaranteed invariant. Outer test shifts can
crop the bar, so sensitivity is not solely a pure-position effect there.
Softmax values are probabilities from these models, not calibrated guarantees.
`mechanism_checks.json` records this example's actual probability sweep,
selection rule, operator errors, source experiment and interpretation boundary.

### 08 — Spatial tensors in 3D

`08_cnn_architecture_3d.html` represents the actual trained CNN with symbolic
boxes sized by W, H and C. x includes a layout offset for each layer, y encodes
height and z encodes channels. Outline planes compress the display rather than
drawing all features. Hover gives shape, parameters and receptive-field span.
Boxes encode geometry, not sampled activations.

The local span is 10×10 after the second pooling layer. Global averaging then
aggregates the union of all spatial positions. The two final logit units are
displayed symbolically as 2×1×1; they are class scores, not an image with two
spatial pixels.

### 09 — Learning, kernels and activations

`09_filter_learning.gif` uses actual first-layer weights at epochs 0, 1, 2, 5,
10 and the final epoch (20 by default). There is one fixed signed color scale
over all shown channels/checkpoints. Frames are not interpolated. Checkpoint
spacing is irregular and each receives the same display duration, so playback
speed is not optimizer time.

`09_training_curves.png` shows full-training cross-entropy and validation
accuracy for each model/seed. `09_features_before_after.png` uses one input and
one channel, with initial/selected activations on a shared scale and actual
class probabilities. The dashboard changes saved epoch, channel and example.
Visual similarity to an edge is a hypothesis, not an established semantic role.

### 10 — CNN versus MLP, with a bounded comparison protocol

`10_cnn_vs_mlp.png` includes architectures, parameter counts, clean test
metrics, training/validation curves, translation sensitivity and additional-noise
sensitivity. Each line is one executed seed; no uncertainty band is inferred
from two seeds.

The CNN uses Conv(1→8,3,p1) → ReLU → Pool2 → Conv(8→16,3,p1) → ReLU → Pool2 →
spatial mean → Linear(16→2), totaling **1,282 parameters**. The MLP uses
flatten(1024) → Linear(1024→32) → ReLU → Linear(32→2), totaling **32,866**.

For each seed, the models share synthetic splits, labels, minibatch order,
Adam 0.01, batch size 32 and a fixed 20-epoch budget. They are not parameter-,
arithmetic- or tuning-budget matched. No early stopping, best-seed selection
or test tuning is performed. The shared rate is illustrative; it may suit one
architecture better. These constraints make a reproducible mechanism exercise,
not a benchmark ranking CNNs and MLPs.

### 11 — Inspect real failure cases

`11_feature_maps_and_failures.png` displays actual errors under dx=+8 and
additional Gaussian noise std=1 for the first seed, ranked by confidence. If
there are no errors, uncertain correct predictions are labeled as such. The
gallery includes model, condition, sample, true/predicted class and softmax
value. This deliberately selected view helps diagnose failure modes and is
not a new unbiased performance estimate. It does not establish calibration.

## Reproducibility and evidence

Generated collections include `experiment.json` with hypothesis, configuration,
versions, split seeds, history, actual checkpoints, evaluation results,
interpretation candidate and limitations. `manifest.json` records each artifact's
name, size, hash and raster dimensions/frame count. Timestamped directories
preserve prior executions; generated records remain ignored by default.
Public interpretation remains **author review required**.

The 32×32 visual protocol generates 256/128/128 training/validation/test images
per seed. Balanced bars have equal signal area/intensity (15×3 or 3×15), centers
uniformly drawn from 14..17 on both axes and independent Gaussian base noise
std=0.12. Pixels are not clipped or normalized. Split seeds are
$1000+10s$, $1001+10s$, $1002+10s$ for experiment seed $s$; blocks are distinct
between the two default seeds. Model/shuffle seeds are $2000+s$/$2001+s$.
Extra noise uses seed $1003+10s$ and the same paired perturbations across models
and noise levels. Horizontal shifts are -8..8 in steps of 2 with zero fill.

The earlier analytical study uses 16×16 bars, one convolution, 98 parameters,
other split seeds and 30 epochs. Its dated record and proposed experiments
remain intact. Visual-phase stress comparisons are new measurements under
this separate protocol, not extensions of the old holdout population.

## Executed visual experiment — 2026-11-18

**Hypothesis:** spatial sharing may give a compact CNN different translation
sensitivity from a flattened MLP on the same synthetic images. The executed
protocol is the 32×32, two-seed configuration above: seeds 59/60, 20 epochs,
256/128/128 images per split and 160 updates per model. All four models were
run; no seed or epoch was selected after seeing test results.

The measured report was created at `2026-11-19T01:18:43.390724+00:00`
(2026-11-18 22:18 in America/Sao_Paulo). Environment: Windows, Python 3.11.14,
NumPy 2.4.6, PyTorch 2.15.0+cpu, Matplotlib 3.11.2, Plotly 6.9.0 and Streamlit
1.67.0, one CPU thread. The initial collection and subsequent rendering refresh
retain separate timestamped directories. The refresh reuses this exact report;
it changes presentation, not training, checkpoints or measured results.

| Model | Seed | Initial → final train CE | Final validation accuracy | Clean test CE | Clean test accuracy |
|---|---:|---:|---:|---:|---:|
| CNN | 59 | 0.713264 → 0.000259 | 1.0 | 0.000239 | 1.0 |
| CNN | 60 | 0.694935 → 0.000823 | 1.0 | 0.000899 | 1.0 |
| MLP | 59 | 0.694186 → 0.0 | 1.0 | $9.3132\times10^{-10}$ | 1.0 |
| MLP | 60 | 0.698835 → 0.0 | 1.0 | 0.0 | 1.0 |

Each clean test matrix was `[[64, 0], [0, 64]]` (true rows, predicted columns).
Reported zeros are float32 numerical results, not mathematical certainty or a
claim of calibrated probability 1. Training loss reduction alone does not
establish generalization.

| Stress condition | CNN seed 59 | CNN seed 60 | MLP seed 59 | MLP seed 60 |
|---|---:|---:|---:|---:|
| dx = -8 | 1.0 | 1.0 | 0.5078125 | 0.5 |
| dx = +8 | 1.0 | 1.0 | 0.5078125 | 0.5234375 |
| Additional Gaussian std = 0.5 | 0.7578125 | 0.9765625 | 1.0 | 1.0 |
| Additional Gaussian std = 1.0 | 0.5078125 | 0.875 | 0.953125 | 0.9921875 |

These are accuracies on transformations of the same 128 test images per seed,
not four new independent datasets. The CNN retained 1.0 accuracy at all nine
measured horizontal shifts in both runs. Its confidence need not be unchanged.
The separate stride-1 operator has an exact interior-alignment control, not an
architecture-wide invariance proof.

The final translation illustration uses test sample 4 (the first vertical bar
by label) in seed 59. Its nine interior alignment errors were all 0.0 in the
NumPy control. Actual CNN $P(\mathrm{true})$ changed from 0.9999949 at dx=0 to
0.9990655 at dx=-6 while its predicted class stayed correct. The MLP probability
changed from 1.0 to 0.0002166 for those same inputs. This one example makes the
difference between aligned local responses, unchanged labels and unchanged
probabilities visible; it does not estimate their frequency. The exact values
and selection rule are retained in the generated `mechanism_checks.json`.

**Interpretation candidate — author review required:** the two model families
fit this easy clean task but have different measured weaknesses. The chosen
CNNs tolerated the tested translations better; the chosen MLPs tolerated the
strong added noise better. The CNN noise result also varied substantially by
seed. These observations do not isolate parameter sharing as the only cause,
identify semantic filters, or establish a universal architecture preference.

**Limitations:** only two seeds, one easy generator, narrow training center
support, paired test corruptions, possible cropping at outer shifts, no tuning
or capacity/compute matching, and no representative real-image validation.
The final test stress sweep is diagnostic, not a model-selection protocol.
Learned kernel frames are actual snapshots; the first seed's channel gallery
is a selected illustration rather than a distribution of representations.

## Proposed follow-up work — not executed here

- Use a validation-based, parameter/compute-aware hyperparameter protocol and
  fresh final tests before comparing model quality.
- Separate pure noncropping translations from signal loss, broader position
  supports and acquisition shifts.
- Explore effective input-gradient sensitivity and compare it with structural
  masks, without treating a gradient as a causal attribution.
- Compare anti-aliased downsampling and dense-prediction architectures on
  representative data rather than extrapolating from bars.

The primary [CNN references](notes.md#primary-references-consulted) support the
operator conventions and spatial calculations. Software tests, browser checks,
scientific interpretation review and real-world validation are distinct.

## Executed software verification

Local verification used the stable Python 3.11.14 environment described above:

- 51 analytical/numerical tests and 29 visual/application tests passed, with no
  skips or warnings. This includes real temporary exports, exact RGB agreement,
  structural masks checked against autograd, restored checkpoints and all
  eleven dashboard sections. A regression test requests identical training in
  two AppTest sessions and confirms only one bounded training call.
- All fourteen complete visual exports were generated and their sizes/hashes
  checked against the manifest. GIF states and selected PNGs were visually
  inspected. The rendering refresh preserved the original experiment JSON
  exactly; the selected previews match their generated files byte for byte.
- The running Streamlit agent API verified all sections, invalid geometry,
  RGB channel/pixel controls, border neurons and odd translations. Two real
  sessions also requested the default 20-epoch comparison successfully and
  returned the same downloadable JSON, including the same creation timestamp.
  That UI verification run at `2026-11-19T01:55:39.337183+00:00` repeated the
  exported experiment's numerical results exactly; it is not another seed.
- Chrome 157 verified real filter-menu changes, 3D rotation/zoom, checkpoints,
  the failure gallery and no document-level horizontal overflow at a narrow
  viewport. The dashboard and three standalone HTML controls had no JavaScript
  errors or external HTTP requests. Plotly emitted Canvas2D readback performance
  warnings. The temporary generic HTTP server's favicon request received an
  empty response during standalone checks; it is not a Plotly data dependency.
- `ty check` passed for the six source modules. Repository syntax and internal
  links passed; `uv --no-cache pip check` found 78 compatible installed packages;
  `git diff --check` passed with only Git's CRLF conversion notices.

These are local Windows/Chrome checks, not Linux execution, cross-browser
coverage, GitHub-rendered Markdown inspection, CI, scientific author approval
or production validation. Input-gradient effective receptive-field maps,
parameter-matched tuning and real-image experiments remain unexecuted.

## Selected previews and output policy

The four inspected public Git candidates are:

| Preview | Bytes | Learning question |
|---|---:|---|
| [Sliding convolution](assets/01_convolution_sliding.gif) | 1,933,508 | How do patches, products and sums construct the output? |
| [Receptive-field growth](assets/06_receptive_field_growth.gif) | 150,933 | How does structural input support expand through layers? |
| [Actual filter learning](assets/09_filter_learning.gif) | 418,682 | How do measured kernel coefficients change at real checkpoints? |
| [CNN versus MLP](assets/10_cnn_vs_mlp.png) | 258,511 | How do actual clean and stress results differ under this protocol? |

Total selection: **2,761,634 bytes**. The README inlines convolution and the
comparison (2,192,019 bytes) and links the two complementary GIFs. The convolution
GIF intentionally exceeds the default 1 MiB per-GIF review target: its 36
distinct algorithm states, six readable panels and numerical annotations answer
a question a still cannot. A Pillow lossless optimization trial preserved every
decoded pixel, frame count, duration and loop but saved zero bytes; the original
encoding was retained. No states were dropped and playback time was unchanged.

These files are narrowly unignored, inspected and verified as public Git
candidates before linking. The other exports, fully embedded Plotly runtimes,
checkpoints, experiment/mechanism records, manifests and inspection screenshots
remain regenerable ignored artifacts. No generated HTML is linked as a public
asset merely because it exists locally. The [repository preview policy](../../docs/visual-assets.md)
defines review budgets and what image encoding checks do not verify.
