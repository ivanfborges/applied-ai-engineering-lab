# Day 53 visual learning lab

[Topic overview](README.md) · [Generator](visualize_day53.py) ·
[Numerical helpers](visual_math.py) · [Renderers](visual_render.py) ·
[Tests](tests/test_visual_math.py)

The lab asks how a neural network changes the representation of its input,
and what training actually changes. All observations are synthetic. It runs
locally on CPU with Python 3.11+, without a notebook, web service, GPU, FFmpeg
or ImageMagick. HTML pages embed Plotly and their controls; open them directly
from the file manager after generation.

## Installation and execution

The repository's canonical dependency bounds remain in
[pyproject.toml](../../pyproject.toml). Use a stable isolated interpreter:

```powershell
python -m pip install -e ".[dev]"
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode all --quick
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode all
```

The [requirements entry point](requirements.txt) lists only the five direct
lab dependencies, without duplicating version bounds. For a minimal standalone
environment, run `python -m pip install -r 04-deep-learning/53-perceptron-mlps/requirements.txt`.
The repository editable install above is preferred for version compatibility
and tests. Preserve the neighboring `example.py`, `from_scratch.py` and visual
helpers if moving the lab. No installation is performed by the generator.
Seeds reproduce numerical configurations; fonts and library/platform versions
can change export pixels or bytes. The run record preserves the actual environment.

```powershell
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode activations
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode xor
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode training
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode approximation
python 04-deep-learning/53-perceptron-mlps/visualize_day53.py --mode neuron --seed 42 --output-dir .local/day53-neuron
```

Other modes are `perceptron`, `features`, `forward` and `landscape`. `all`
runs all nine. Default output is `outputs/visual_lab/` beside the generator,
independently of the working directory. `--output-dir` is relative to the
current directory when supplied as a relative path. The script creates it.
Files with the same asset names are replaced on repeat runs; dated JSON run
records are retained. Use different directories to preserve different seeds
or configurations, as in `outputs/quick/` and `outputs/full/`.

| Setting | Quick | Default |
|---|---:|---:|
| Grid points per axis | 41 | 71 |
| Requested activation/feature/checkpoint frames | 20 | 40 |
| NumPy full-batch training updates | 2,000 | 10,000 |
| Export DPI | 90 | 110 |

Training checkpoint indices are geometrically spaced and deduplicated; actual
frame counts can differ from requested counts. Pauses repeat states without
interpolating parameters. GIFs run at 3–6 frames/second. Increasing width in
the constructive approximation is a sequence of actual constructions, not
an optimizer animation. Quick mode retains the same 800-point noisy-XOR
comparison so its population is unchanged; only rendering and the separate
truth-table training workload are reduced.

## Outputs and what to observe

These are **generated filenames**, not links to versioned assets. All exports
remain ignored until a small preview set is intentionally selected.

| # | Output files | Question and viewer action |
|---:|---|---|
| 1 | `01_neuron_diagram.png`, `01_neuron_controls.html` | Follow inputs through weighted summation, bias and activation. In HTML choose a preset, vary both weights and bias, and switch activation. Compare surface slopes, vertical displacement and the input-space zero contour. |
| 2 | `02_perceptron_learning.gif`, `02_perceptron_final.png` | Follow the real mistake-driven updates on 48 separable synthetic points. Read epoch, cumulative updates, current training errors and nonpositive margins. |
| 3 | `03_xor_comparison.png` | Compare fitted logistic and nonlinear MLP score fields on the same noisy-XOR split. Large binary-corner markers remain identifiable. Intersecting class convex hulls explain the affine limitation. |
| 4 | `04_activation_flow.gif`, `04_activation_derivatives.png` | Follow a moving pre-activation through identity, step, sigmoid, tanh, ReLU and exact GELU. Dashed curves are derivatives; shaded tails show sigmoid/tanh saturation. |
| 5 | `05_hidden_representation.gif`, `05_hidden_representation.png`, `05_feature_lifting.html` | Watch the corners move into explicit ReLU features, then inspect a separator there. In 3D rotate the lifted corners and toggle the feature surface/separating plane. |
| 6 | `06_forward_propagation.gif`, `06_forward_propagation.png` | Follow input `[0,1]` through fixed weights and biases. Compare each displayed pre-activation and activation to the preceding layer. |
| 7 | `07_mlp_training.gif`, `07_mlp_training_final.png`, `07_training_checkpoints.npz` | Watch actual NumPy checkpoints change scores, the 0.5 boundary and training loss. Read update/gradient diagnostics rather than inferring wall-clock speed. |
| 8 | `08_loss_slice.html` | Rotate a real two-coordinate loss slice about the trained reference. Read the axis offsets and distinguish this slice from the complete landscape. |
| 9 | `09_relu_approximation.gif`, `09_relu_approximation.png` | Compare a nonlinear target with constructive ReLU interpolants of widths 2, 4, 8 and 16; inspect approximation errors. |

Every invocation also writes a timestamped `run_*.json` with environment,
configuration, measured results, gradient-check error, decoded frame counts,
file sizes and SHA-256 hashes. Checkpoint NPZ files retain the actual parameters
and their update indices plus the complete training loss history.

## Mathematical contracts and limitations

### 1. One neuron

Both HTML surfaces evaluate $z=w_1x_1+w_2x_2+b$ and $a=\phi(z)$ over
the same grid. Presets include $(w_1,w_2,b)=(1,1,0),(2,-1,0),(1,1,-1)$.
Independent sliders demonstrate slope/orientation versus vertical translation.
The third panel draws $z=0$ from those same parameters. A zero vector of
weights yields either a constant nonzero score with no boundary or zero
everywhere with no unique line. A sigmoid's 0.5 threshold coincides with
$z=0$; selecting another activation does not by itself define a classifier.
Input axes remain fixed; the activation's height range follows its scale.

### 2. Perceptron updates

Candidates are sampled uniformly in $[-1,1]^2$ and the first 48 with
$|x_1+0.6x_2-0.1|>0.3$ are retained. Labels follow the sign of that affine
score, so separation is built into the generator. Training starts with zero
parameters and follows the same zero-margin/mistake update convention as the
analytical implementation: if $y(w^Tx+b)\leq0$, add $yx$ to $w$ and $y$ to
$b$. Current training errors use prediction $+1$ iff score is strictly positive.
A tie can therefore count as an update without being a prediction error.
The animation uses real updates, not an interpolated boundary rotation.

### 3. XOR score fields

The fitted comparison uses 800 latent binary corner draws with independent
Gaussian feature noise SD 0.12, seed-controlled generation and a stratified
600/200 split. Each scaler sees only the training split. Models are logistic
regression and a ReLU MLP with hidden widths `(8,8)`, L-BFGS and
`alpha=0.001`. Black contours come from their actual class-1 scores at 0.5.
Small markers are holdout points; large markers are the exact truth table.
The model scores between corners are extensions chosen by the models, with
no intrinsically defined continuous XOR label function. The positive and
negative class segments both contain $(0.5,0.5)$, preventing linear separation
of the exact corners. One synthetic split does not rank model families generally.

### 4. Activations

Identity, sigmoid, tanh and ReLU have their usual mathematical definitions.
Step is $\mathbf{1}[z>0]$. Exact GELU is $z\Phi(z)$ and its derivative is
$\Phi(z)+z\varphi(z)$, using the standard Gaussian CDF and density.
ReLU and step derivatives at zero are undefined, represented numerically by
NaN and visually by open derivative markers plus text. Step's derivative is
zero away from its jump; that does not make it suitable for ordinary
gradient-based training. Panels use their own fixed vertical ranges, so compare
the labeled values rather than apparent line height across panels.

### 5. Two explicit representations

Let $s=x_1+x_2$. The hidden features are
$h_1=\operatorname{ReLU}(s)$ and $h_2=\operatorname{ReLU}(s-1)$.
The separator is $h_1-2h_2=0.5$. Both positive corners map to $(1,0)$,
so their markers coincide at the final hidden representation. The GIF's
transition $(1-t)x+t\,h(x)$ is explicitly a **conceptual morph**; it is not
a sequence of learned layers or training states. Only the endpoint is this
specified hidden representation. Its input-space pullback is piecewise linear.

The separate 3D construction lifts to $h=x_1x_2$. Its separating plane is
$x_1+x_2-2h=0.5$, drawn as $h=(x_1+x_2-0.5)/2$. At the binary corners,
class 1 lies below this plane. These constructions use no training and make
no claim that these exact features are necessarily learned by the fitted MLP.

### 6. Forward computation

Fixed seed-controlled parameters define a `2 → 4 → 4 → 1` network with 37
parameters, tanh hidden layers and sigmoid output. Each layer computes
$z=a_{\mathrm{prev}}W+b$, followed by its activation. Connection colors encode
weight sign; width is $0.4+1.4|w|$. Node colors encode activations, while
the text panel gives all biases and pre-activations for the active layer.
This fixed random network is **untrained**. Its output is not evidence of a
useful classifier. A separate width example compares 17 parameters for
`2 → 4 → 1` with 33 for `2 → 8 → 1`.

### 7. Actual optimization

This is the analytical NumPy `2 → 8 → 1` model, with tanh hidden activations,
sigmoid output and stable mean BCE from logits. Seed, independent fan-in-scaled
normal weights, zero biases, full-batch gradient descent and learning rate
0.5 are recorded. The entire four-point XOR truth table is training data;
there is no holdout. Every saved state contains the actual parameters after
the indicated number of updates. The loss panel includes every intervening
update. Backpropagation computes gradients; gradient descent applies them.
Geometric checkpoint spacing compresses later training; animation timing is
not training time. Convergence is not assumed for a different seed or step size.

### 8. A restricted loss surface

Only $W_2[0,0]$ and $b_2[0]$ vary by offsets in $[-4,4]$ around the final
checkpoint; all other 31 coordinates stay fixed. Heights are the actual
four-point mean BCE. With hidden features fixed, this output-layer slice is
convex. The full network objective need not be convex. The genuine full
trajectory changes many other parameters and is therefore omitted rather
than drawn at inconsistent heights in this slice. Curvature can affect useful
step sizes, but this display is not a learning-rate experiment or a global
optimality certificate.

### 9. Constructive approximation

On $[0,1]$, the target is $f(x)=\sin(2\pi x)+0.3x^2$. For width $m$,
choose equally spaced knots $t_j=j/m$, with slopes
$s_j=[f(t_{j+1})-f(t_j)]/(t_{j+1}-t_j)$. The construction is

$$
g_m(x)=f(0)+s_0\operatorname{ReLU}(x)
+\sum_{j=1}^{m-1}(s_j-s_{j-1})\operatorname{ReLU}(x-t_j).
$$

This uses exactly $m$ ReLU units and an output bias. Its values equal the
linear interpolant at every point on this domain. No optimizer is involved.
RMSE and maximum absolute error are measured on a fixed 1,001-point grid,
not certified uniform-error bounds over all real inputs. The construction
illustrates capacity; it provides no generalization experiment and does not
prove the Universal Approximation Theorem.

## Validation and preview selection

```powershell
python scripts/validate_repo.py tests --scope 04-deep-learning/53-perceptron-mlps
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

Tests check transformations, layer values, every core gradient coordinate,
loss-slice/direct-objective agreement, checkpoint replay, boundary equations
and real temporary-directory PNG/GIF/HTML exports. The generator decodes every
raster frame and checkpoint array, and rejects external script sources in HTML.
These file checks alone do not establish browser interaction or scientific
interpretation; those require separate evidence.

Recommended candidates for a later public selection are the perceptron GIF,
hidden-representation GIF, XOR comparison PNG, training-final PNG and
approximation PNG. They answer distinct questions and fit the normal preview
budgets. The full training GIF is a useful secondary candidate at about 3.15 MiB;
it preserves genuine optimization checkpoints, but exceeds the default 1 MiB
GIF review target. Prefer the final PNG inline and evaluate any secondary GIF
link separately. Keep large standalone HTML and experiment records
regenerable. Follow the [preview policy](../../docs/visual-assets.md) before
unignoring or linking any generated asset. Public integration in the root
README requires the author's explicit approval.

## Executed visual records — 2026-11-18

Codex executed `--mode all --quick --seed 42 --output-dir outputs/quick` and
`--mode all --seed 42 --output-dir outputs/full` (output paths shown relative
to this topic for brevity). Each produced 18 verified artifacts: eight PNGs,
six GIFs, three self-contained HTML pages and one compressed checkpoint file,
plus its own timestamped JSON record. Quick and full are different budgets
of the same seeded process, not independent repetitions. The initial quick
exports precede the label/layout corrections; full exports contain them.
Later focused neuron/feature/landscape exports refine layout without changing formulas
or experiment data; their timestamped records are separate.

Environment: Windows x64, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1, Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0.
No packages were installed or upgraded for these runs.

| Executed question / hypothesis | Configuration | Measured result |
|---|---|---|
| Can the mistake-driven rule separate the constructed data? | Seed 42, 48 points with positive generating margin, learning rate 1, 100-epoch cap | 8 updates; zero-update certification in epoch 3; 0 final training errors |
| Can nonlinear features fit this low-noise XOR comparison? | Same 800-point generator and 600/200 split as documented above | Holdout accuracy 0.575 for logistic regression and 1.000 for ReLU MLP; no fit warnings |
| Does the selected NumPy training procedure fit the exact corners? | Seed 42, four training points, 33 parameters, learning rate 0.5 | Initial BCE 0.7131771922; quick final BCE 0.0018199610 after 2,000 updates; full final BCE 0.0002591694 after 10,000; both final training accuracies 1.0 |
| Are the implemented gradients consistent with the objective? | Central differences, epsilon $10^{-6}$, all 33 coordinates at initialization | Maximum absolute discrepancy $6.58\times10^{-11}$ |

The fixed untrained `2 → 4 → 4 → 1` example computed output 0.45030006 for
`[0,1]`; that is a forward-pass check, not a measured classifier benefit.
The full loss slice's minimum sampled BCE was 0.00023575 versus reference
BCE 0.00025917. The displayed reference is a finite training checkpoint,
not a claim of exact stationarity or global optimality.

The constructive approximation produced these actual errors on the 1,001-point grid:

| ReLU units | Grid RMSE | Grid maximum absolute error |
|---:|---:|---:|
| 2 | 0.70688600 | 1.01875000 |
| 4 | 0.15084041 | 0.21513324 |
| 8 | 0.03927404 | 0.07154824 |
| 16 | 0.00991827 | 0.01913922 |

**Interpretation candidates — author review required:** these runs are consistent
with the intended distinction between an affine boundary and nonlinear features;
the explicit transformations make XOR separable on its four corners, and more
ReLU interpolation knots reduced grid error for this smooth target. The chosen
NumPy optimizer reached a correct truth-table fit for this seed. None of these
observations establishes universal optimizer convergence, calibrated scores,
comparative real-data performance or the Universal Approximation Theorem.

All raster frames were decoded; first/middle/final frames of all six full-mode
GIFs and the static figures were inspected. Chrome 157.0.8103.79 opened all three
HTMLs directly via `file://` in an isolated offline browser at 1440×1000.
Twenty-one browser assertions checked actual plotted neuron grids, boundary
equations, all four selectable activations, keyboard sliders, the degenerate
boundary, mouse-driven 3D camera changes, feature/plane toggles and loss rendering.
Every rendered feature-surface/plane grid value was checked against its equation;
every rendered loss-slice value was checked against stable BCE from the saved model.
The successful run recorded no page/console errors or external HTTP(S) requests.
Three Plotly Canvas2D readback-performance warnings occurred. Browser access
initially required its normal cache-write permission and an isolated local-file
configuration; no Python server or network fallback was used.

The final focused test run passed **57 tests** (24 analytical and 33 visual),
without skips or warnings. Repository syntax parsed 362 Python files, internal
links checked 1,236 targets across 244 files, dependency compatibility passed,
and `git diff --check` passed. No full-curriculum suite or CI run was performed.

These checks do not cover other browsers, mobile layouts, native-GPU performance,
GitHub Markdown rendering, accessibility certification or author review.
