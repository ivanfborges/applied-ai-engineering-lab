# Interactive optimizer laboratory

[Topic overview](README.md) · [Equations and analytical study](notes.md)

The lab asks how gradient history turns into a parameter update. Every path
comes from the shared [NumPy optimizer](from_scratch.py); the renderer does
not invent trajectories or interpolate intermediate optimizer states. These
are toy optimization mechanisms, not neural-network or generalization benchmarks.

## Run locally

Stable Python 3.11+ on Windows; no GPU or network access at runtime. NumPy,
Matplotlib, Pillow and Plotly are already declared in the canonical
[pyproject.toml](../../pyproject.toml). PyTorch is not imported by the visual
workflow or its dedicated tests.

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[dev]"
python 04-deep-learning/56-optimizers/visualizations.py --all
python 04-deep-learning/56-optimizers/visualizations.py --scene trajectories
python 04-deep-learning/56-optimizers/visualizations.py --scene surface3d --show
python 04-deep-learning/56-optimizers/visualizations.py --scene adam
python 04-deep-learning/56-optimizers/visualizations.py --scene adamw
```

Create a new environment only when needed. In the author's existing Windows
workspace, use the stable interpreter documented in the local workflow;
do not replace an existing environment merely to run this lab.

No arguments also generates all scenes. Default outputs go to this topic's
ignored `outputs/` directory. Double-click `02_interactive_loss_surface.html`
to open it directly. Its Plotly runtime is embedded; there is no CDN, local
server, Node installation, JavaScript build or cloud service requirement.
`--show` opens the HTML, or the first PNG for a non-3D scene, in the default
browser. It does not enable live Python recomputation.

For a shorter export or a new configuration:

```powershell
python 04-deep-learning/56-optimizers/visualizations.py --all --quality quick --steps 40 --output 04-deep-learning/56-optimizers/outputs/quick
python 04-deep-learning/56-optimizers/visualizations.py --scene surface3d --objective rosenbrock --steps 120 --log-surface --show
python 04-deep-learning/56-optimizers/visualizations.py --scene trajectories --a 80 --b 1 --angle 55 --noise-std 0.5
python 04-deep-learning/56-optimizers/visualizations.py --scene adamw --weight-decay 0.3 --learning-rate 0.06
python 04-deep-learning/56-optimizers/visualizations.py --scene learning-rate --optimizer momentum
```

Changed options define new runs. Use distinct `--output` directories to retain
alternate exports; dated manifests are never overwritten. Static-only mode
omits GIF encoding with `--static-only` while still generating PNGs and HTML.

## Connected experiments

| Scene / CLI name | Question and exported artifacts |
|---|---|
| 01 / `trajectories` | How do update rules change a common starting point? `01_optimizer_trajectories.png`, `01_optimizer_trajectories.gif` |
| 02 / `surface3d` | How do actual iterates sit on the same objective surface? `02_interactive_loss_surface.html` |
| 03 / `momentum` | Which part of the next direction comes from history? `03_sgd_vs_momentum.gif`, `03_momentum_internal_state.png` |
| 04 / `rmsprop` | How do coordinate histories rescale gradients? `04_rmsprop_adaptive_scaling.gif`, `04_rmsprop_internal_state.png` |
| 05 / `adam` | How do moments and initialization bias affect the update? `05_adam_internal_mechanics.gif`, `05_adam_moments.png`, `05_adam_bias_correction.png` |
| 06 / `adamw` | Does decay enter the moments or bypass them? `06_adam_vs_adamw.gif`, `06_weight_decay_comparison.png` |
| 07 / `learning-rate` | When do actual small/reference/large rates converge or diverge? `07_learning_rate_sensitivity.gif`, `07_learning_rate_curves.png` |
| 08 / `noise` | How does artificial noise interact with optimizer history? `08_gradient_noise.gif`, `08_gradient_noise_curves.png` |

Five optimizer colors are fixed across scenes: SGD blue, Momentum orange,
RMSProp green, Adam red and AdamW purple. Solid/dashed coordinate lines mean
x/y in diagnostic charts. Within one optimizer's rate/noise experiment,
line style distinguishes the configurations; it does not change method identity.

### 01 and 02: geometry and genuine updates

For a quadratic, first compute coordinates relative to the optional center,
then rotate them:

$$
u=x\cos\theta+y\sin\theta,\qquad v=-x\sin\theta+y\cos\theta,
\qquad f(x,y)=\tfrac12(au^2+bv^2).
$$

Default curvature is $(a,b)=(40,1)$, angle is 30 degrees, the center is zero,
and the common start is $(-2,1.5)$. The condition number is 40. The analytic
gradient uses $R^\top\operatorname{diag}(a,b)R\theta$; changing rotation alters
the relationship between curvature directions and coordinate-wise scaling.
RMSProp/Adam are coordinate-dependent, not rotation-invariant Hessian solvers.

Optional Rosenbrock uses $f(x,y)=(1-x)^2+100(y-x^2)^2$, with known minimum
$(1,1)$. Quadratic curvature, rotation and center arguments do not change that
function. A valley can bend without an optimizer having any direct knowledge
of that bend. The default Rosenbrock rates are separate illustrative settings.

The 2D arrows show the previous-to-current displacement in parameter units.
The loss chart evaluates the exact objective even when the supplied gradient
has simulated noise. Adam and AdamW overlap at zero decay; purple may obscure
red. This is expected equality, not a missing implementation.

In the 3D HTML, select one optimizer or compare all, move the step slider,
use Play/Pause, hover the paths, and rotate/zoom/pan with the camera controls.
Hover data retain actual step, parameters and raw loss. A circle marks the
current iterate, while translucent diamonds show precomputed endpoints even
before playback reaches them. Frame sampling keeps the file responsive and
includes the initial and final states. No interpolated iterate is claimed.

The surface mesh is sampled, so its graphical triangles only approximate the
continuous function. Path heights are evaluated analytically from exactly
the same function. Optional `--log-surface` transforms both mesh and path
heights with $\log(1+f)$ and labels that axis; raw losses remain in hover data.
Straight 3D segments join discrete iterates; they do not describe a continuous
optimizer path constrained to the surface between updates.

### 03: Momentum's exact accumulator convention

$$
v_t=\mu v_{t-1}+g_t,\qquad\theta_{t+1}=\theta_t-\eta v_t.
$$

The initial accumulator is zero, with no $(1-\mu)$ factor, dampening or
Nesterov. The vector-composition inset places $\mu v_{t-1}$ at the tip of
$g_t$, whose sum is $v_t$. Those three vectors share a common display scale
at each frame; the black update arrow is unit-scaled to emphasize its
opposite direction. Current numerical magnitudes and the velocity time series
preserve the information that arrow scaling removes. Parameter-space arrows
remain actual displacements.

Increasing momentum retains directions longer and changes their magnitude.
Alternating components can partly cancel while persistent components add,
but momentum can still overshoot or oscillate. This is visible behavior to
inspect, not a promise of improvement at every rate.

### 04: RMSProp's normalization

$$
s_t=\rho s_{t-1}+(1-\rho)g_t^2,\qquad
\Delta\theta_t=-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}.
$$

The synchronized panels show supplied gradients, uncentered second moments,
coordinate scaling factors $\eta/(\sqrt{s_t}+\epsilon)$ and actual updates.
Large history in one coordinate increases its denominator. These are
adaptive gradient scaling factors, not universally valid effective learning
rates, gradient variances or true curvatures. Epsilon is outside the root.
Changing rho changes forgetting; changing epsilon matters most when the
second moment is small. The implementation is uncentered and has no momentum.

### 05: Adam's two histories and initialization correction

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\quad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\quad
\Delta\theta_t=-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

The first moment smooths directions; the second raw moment affects the
denominator. Dotted update curves omit bias correction using the same stored
gradient history. They are a diagnostic counterfactual, not an independent
uncorrected-Adam optimization trajectory.

A separate 20-step constant-gradient experiment uses $g=(0.7,-1.2)$ and
defaults $\beta_1=0.9$, $\beta_2=0.999$. Raw moments start toward zero; their
corrected values equal $g$ and $g^2$ within rounding. This isolates
initialization bias. Correction does not make nonstationary gradients
stationary, and the second raw moment is not gradient variance.

### 06: coupled L2 and decoupled decay

The matched pair uses the same rate, coefficient, start and random draws.
For the quadratic only, its minimum is deliberately shifted to $(1,-0.7)$:
weight decay pulls toward the origin, which is now visibly different from
the loss minimum. The Rosenbrock alternative retains its original function.

Coupled Adam uses $g+\lambda\theta$ inside both moments. It therefore has no
independent additive parameter-decay vector to subtract afterward. AdamW
instead uses the original $g$ and applies

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t
-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

The panels separate AdamW's actual adaptive delta, decay delta and their sum,
alongside coupled Adam's update. Norm curves include a separate zero-decay
control. That control must match up to floating-point tolerance; matching a
nonzero coefficient does not match effective regularization strength.
Neither a smaller norm nor a different path establishes validation quality.
Changing the rate also changes the magnitude of AdamW's decay.

### 07: rate sensitivity and explicit stopping

For the default quadratic and SGD, rates are 0.002, 0.04 and 0.055.
The exact full-gradient contraction bound is $0<\eta<2/40=0.05$.
The large preset crosses that bound. Momentum and adaptive methods have
their own presets; the code does not assign the same rate semantics to all
algorithms. A large adaptive rate need not diverge monotonically.

The loss curve displays actual outcomes. The parameter plot keeps a fixed
view near the start/minimum, so divergent paths may leave the viewport;
updates themselves are never clipped. The focus path retains its actual first
exiting segment, then omits subsequent out-of-view connections; the complete
loss history remains visible. Stop guards are objective loss above $10^{12}$
or parameter norm above $10^6$, plus explicit rejection of
nonfinite arithmetic. Finite guard-crossing points remain recorded. Numerical
failure before a finite objective can be recorded leaves the earlier history
intact and a stated stop reason. The log-loss display floors values at
$10^{-30}$ only for plotting; it does not change optimization.

### 08: synthetic noise, not sampled mini-batches

The dedicated experiment compares exact, sigma=0.5 and sigma=3 Gaussian
perturbations for SGD, Momentum and Adam. It uses identical standardized
random draws for paired comparisons, scaled by each sigma. Each optimizer
still visits different parameters and receives its own analytic gradient.
There is no sampled dataset or actual mini-batch experiment in this scene.
Noise can change a path and retained state; no generalization benefit follows.

## Controls and reproducibility

All Python controls recompute real trajectories before export. The HTML's
controls inspect that finite precomputed experiment; they do not run arbitrary
new Python configurations inside the browser.

| Control | Scope |
|---|---|
| `--objective`, `--a`, `--b`, `--angle`, `--start X Y` | Choose geometry and common initialization; a/b/angle affect quadratics only |
| `--steps`, `--seed` | Requested updates and deterministic NumPy random generator |
| `--optimizer` | Filters 01/02 and selects 07's optimizer; `all` uses SGD for 07 |
| `--learning-rate` | Overrides base rates, including the matched decay pair; the independent rate sweep retains its three presets |
| `--momentum`, `--rho`, `--beta1`, `--beta2`, `--epsilon` | Update-rule hyperparameters; coefficients in [0,1), epsilon positive |
| `--weight-decay` | Targeted scene 06 only; all other scenes use zero decay |
| `--noise-std` | General scenes' gradient perturbation; scene 07 stays exact |
| `--noise-low`, `--noise-high` | Dedicated scene 08's low/high noise levels |
| `--log-surface` | Labeled vertical transform for scene 02 |
| `--output`, `--quality`, `--static-only`, `--show` | Destination, bounded rendering, GIF omission and browser opening |

Default quadratic rates: SGD 0.04, Momentum 0.035, RMSProp/Adam/AdamW 0.09.
Default Rosenbrock rates: SGD 0.001, Momentum 0.0005, RMSProp 0.01,
Adam/AdamW 0.025. The quadratic SGD/Momentum reference rates scale with the
largest curvature. These choices are illustrative and not tuned benchmarks.
Decay defaults to 0.1 in scene 06, all states start at zero, and all simulations
use float64 NumPy. Defaults retain 80 actual updates.

| Quality | Maximum displayed frames | GIF DPI | PNG DPI | Mesh resolution |
|---|---:|---:|---:|---:|
| quick | 18 | 75 | 110 | 51 x 51 |
| standard | 36 | 85 | 150 | 75 x 75 |
| high | 60 | 110 | 180 | 101 x 101 |

GIFs use PillowWriter at 8 fps without FFmpeg. At most 2,000 optimization
steps are accepted, independently of the capped frame count. Frame subsampling
can hide intermediate zigzags in playback; the accumulated trajectory and
loss curve retain every recorded step. Plotly redraws discrete 3D frames,
preserves the camera and keeps the selected method visible across steps.
HTML size includes the bundled Plotly runtime. Browser/WebGL performance is
hardware-dependent; no laptop-speed guarantee or cross-browser validation is
implied.

## Validation

```powershell
python scripts/validate_repo.py tests --scope 04-deep-learning/56-optimizers/tests/test_visual_lab.py
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

The dedicated visual test file requires only `.[dev]`, with no PyTorch skips.
Running all topic tests also includes the pre-existing PyTorch training and
agreement checks, requiring `.[dev,deep-learning]`.
Tests use temporary destinations for actual PNG/GIF/HTML exports, check
finite differences, states, constant-gradient correction, exact loss heights
in every frame, visibility masks and divergence guards. Browser checks remain
separate from numerical and file-format tests.

Source responsibilities: [visual_math.py](visual_math.py) computes objectives
and histories, [visual_render.py](visual_render.py) renders them, and
[visualizations.py](visualizations.py) supplies the CLI and dated manifests.
No second optimizer implementation is hidden in the renderer.

## Executed visual study — 2026-11-18

Codex ran the default `--all` command using the existing stable Python
3.11.14 interpreter on Windows x64, with NumPy 2.4.6, Matplotlib 3.11.2,
Pillow 12.3.0 and Plotly 6.9.0. Plotly.js 3.7.0 is bundled in the exported
HTML. NumPy simulation was float64; no PyTorch training or GPU computation
was part of this visual execution. OMP/OpenBLAS/MKL thread limits were 1.

**Configuration:** the defaults above: quadratic a=40/b=1, angle 30 degrees,
80 requested updates, start (-2,1.5), seed 56, standard quality, no noise
except the dedicated noise scene, and decay 0.1 only in its matched pair.
The constant-gradient check used 20 updates. The full export created eight
PNGs, seven GIFs and one HTML in about 183 seconds locally. That duration
includes plotting and encoding and is not an optimizer benchmark.

The first complete manifest is `visual-run-20261118T181234563927Z.json`.
Later dated exports retain the same numerical settings while correcting
layout/playback and the rate plot's focus-view presentation:
`181904093160Z` (3D), `181934843158Z` (decay), `182029068178Z` (noise),
and `182159430497Z` (rate sensitivity). Those suffixes identify additional
`visual-run-20261118T...json` files in ignored `outputs/`. Earlier manifests
are preserved; their hashes identify their earlier artifacts and source.
An additional Rosenbrock/log-surface run was exported under `outputs/rosenbrock/`
with manifest `visual-run-20261118T182052285118Z.json`. These are separate
visual runs, not replacements for the historical analytical tables.

### Mechanism checks and observed values

**Hypothesis:** the same gradient-to-update definitions generate different
paths under the stated histories/rates, while zero decay removes the
Adam/AdamW distinction and constant-gradient correction removes initialization
factors. The source/manifest retain configurations, paths, losses and stop
reasons for each scene.

| Check | Measured result |
|---|---|
| Reference trajectories | All five completed 80 updates with finite points/losses |
| Same initial objective | 21.9312638786 for every reference optimizer |
| Final reference losses | SGD 0.0038499903; Momentum 0.0006646761; RMSProp 0.0019789915; Adam and AdamW 0.4326248812 |
| Zero-decay paired control | Maximum full-path coordinate difference exactly 0 |
| Constant-gradient correction | Maximum first-moment error $2.22\times10^{-16}$; second-raw-moment error $2.07\times10^{-14}$ |
| SGD rate 0.002 | Final loss 1.9184742172 after 80 updates |
| SGD rate 0.04 | Final loss 0.0038499903 after 80 updates |
| SGD rate 0.055 | Guard crossed at update 68; retained finite loss $1.1322631477\times10^{12}$ |

**Interpretation candidate — author review required:** the zero-decay control
supports the stated equivalence, the constant-gradient experiment isolates
initialization correction, and the SGD sweep illustrates the curvature-based
stability bound on this quadratic. The five final losses are configuration
outcomes, not a tuned ranking. Nonzero decay changes the paired paths and
their update components; it does not establish validation-quality benefits.

**Limitations:** low-dimensional synthetic objectives, fixed illustrative
rates and a single noise seed; no real mini-batch sampling, validation set,
multi-seed quality comparison or neural-network experiment in this phase.
Different rotation, curvature, rates and schedules can change the behavior.

### Automated and browser verification

- The focused runner passed **148 tests** in two isolated files: 102 existing
  analytical/PyTorch cases and 46 visual cases, with no skips or warnings.
  The visual import check also runs in a fresh process and confirms PyTorch
  is not imported by the visual workflow.
- Real PNG/GIF/HTML export tests write to temporary directories. The final
  PNGs were opened; first/middle/last composited frames of all seven full GIFs
  were inspected. Each full GIF has 36 frames and a total decoded playback
  duration of 4,320 ms; Pillow/GIF quantizes the requested 8 fps to 120 ms
  per frame. This playback is not the wall-clock time of optimization.
- Direct-file offline Chrome 157.0.8103.79 at 1440x1000 passed 16 checks over
  the quadratic and Rosenbrock HTMLs: selector, step slider, actual Play/Pause,
  rotate/zoom/pan, camera preservation, selected-method persistence, geometry
  and the labeled log transform. There were no page/console errors and zero
  HTTP(S) requests. Two Canvas2D readback-performance warnings were emitted.
- Browser path-height error was at most $7.11\times10^{-15}$; surface-sample
  error was at most $5.69\times10^{-14}$. The Rosenbrock log-path check differed
  from direct evaluation by at most $2.23\times10^{-16}$. All five final paths
  contained 81 points, including the start.
- The initial browser review found overlapping labels and an asynchronous
  error when native 3D Play/Pause interrupted Plotly's animation queue.
  The final export uses sequential recorded-frame updates and stops after
  an in-flight frame; the final browser run confirmed actual pause behavior.
  Play/Pause depends on the bundled Plotly runtime and should be rechecked
  after dependency changes. Failed attempts and final evidence remain local.
- `uv pip check`, pointed at the stable interpreter with an ignored workspace
  cache, checked 78 installed packages and found them compatible. No packages
  were installed or upgraded. Full-curriculum CI, GitHub math rendering,
  mobile layouts and other browsers were not validated in this phase.

### Selected public previews

| Selected asset | Bytes | Learning question |
|---|---:|---|
| [Trajectory GIF](assets/01_optimizer_trajectories.gif) | 1,144,693 | Which actual direction and loss change follow each sampled update? |
| [Bias-correction PNG](assets/05_adam_bias_correction.png) | 105,333 | What does zero initialization do to a constant signal? |
| [Decay decomposition PNG](assets/06_weight_decay_comparison.png) | 289,947 | Does shrinkage enter the moments or bypass them? |

All three were inspected, copied without re-encoding and deliberately
unignored: 1,539,973 bytes total. The trajectory GIF is a documented exception
to the 1 MiB per-GIF review target: it retains 36 actual sampled states, update
arrows and the synchronized loss history. A still would omit the changing
update direction; no frames or playback duration were removed for the byte
target. The README embeds that GIF and the bias figure, with the decay figure
linked as a secondary view. The roughly 5.8 MB standalone HTML stays local,
alongside other exports, manifests and inspection sheets. The generator
reproduces it with its full offline runtime and licenses.

Interpretations and public-preview publication decisions remain **author
review required**. Numerical and browser checks do not approve them for the
author. No commit, push or deployment was performed.

Implementation references: Matplotlib's
[PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html),
Plotly's [animation controls](https://plotly.com/python/animations/) and
[standalone HTML export](https://plotly.com/python/interactive-html-export/).
