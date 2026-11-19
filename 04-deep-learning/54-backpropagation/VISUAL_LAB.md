# Backpropagation Visual Learning Lab

[Topic overview](README.md) · [Numerical implementation](visual_math.py) ·
[Renderer](visual_render.py) · [CLI](visual_lab.py) · [Tests](tests/test_visual_lab.py)

Eight executable demonstrations connect values, sensitivities and updates.
The models are deliberately small enough to independently verify every
quantity. All data are synthetic. No cloud account, web server, GPU,
ImageMagick, FFmpeg or notebook is required.

## Install and run locally

Use stable Python 3.11+ in an isolated environment. From the repository root:

```powershell
python -m pip install -e ".[dev]"
python 04-deep-learning/54-backpropagation/visual_lab.py --list
python 04-deep-learning/54-backpropagation/visual_lab.py --all
python 04-deep-learning/54-backpropagation/visual_lab.py --only computational_graph
python 04-deep-learning/54-backpropagation/visual_lab.py --only loss_landscape
python 04-deep-learning/54-backpropagation/visual_lab.py --only gradient_flow
```

Alternatively, inside this topic directory:

```powershell
python -m pip install -r requirements.txt
python visual_lab.py --all
```

[requirements.txt](requirements.txt) is a minimal standalone compatibility list;
[pyproject.toml](../../pyproject.toml) owns the repository's dependency ranges.
NumPy, Matplotlib, Plotly and Pillow are already base dependencies. PyTorch is
optional and is not installed automatically. If available, the lab compares
the two-parameter manual gradients with float64 CPU autograd; otherwise the
dated report explicitly says that comparison was not performed.

The standalone list supplies runtime libraries, not pytest or a frozen
numerical environment. Use the root installation above for repository tests;
do not treat the two lists as competing sources of dependency bounds.

The default command with no flags also generates all modules. Output defaults
to a fresh ignored `outputs/visual-lab/run-<UTC timestamp>-<suffix>/` beside
the script, regardless of the current working directory. The command prints
that directory; open its generated `loss_landscape_3d.html` to interact.
Each invocation also creates a distinct `run-<UTC timestamp>.json` with
configuration, versions, numerical observations, interpretation status and
decoded-output hashes. Explicit output directories must be new or empty;
nonempty destinations and existing export files are rejected rather than
overwritten. Repeating a command without `--output` creates a separate run.
The selected `visuals/` previews are historical public artifacts, not the
default generation destination. Review new exports before deliberately
promoting an individual preview; do not bulk-copy a run over that directory.

```powershell
python visual_lab.py --all --quick
python visual_lab.py --only loss_landscape --frames 30 --grid 51 --steps 180 --output outputs/custom-landscape-new
```

Default quality requests 60 frames, a 91×91 surface grid and 180 optimization
updates per rate. `--quick` requests 24 frames and a 41×41 grid while preserving
the dataset and optimization budget. `--frames` accepts 18–120; `--grid`
accepts 21–151. `--steps` changes the numerical experiment, not just display
quality. GIFs sample actual updates rather than inventing interpolated
parameter states. Pillow merges repeated frames while retaining their hold
durations; the computational graph still contains all 14 distinct phases.
Matplotlib uses the headless Agg backend and closes figures after exporting.

## Choose a question

| `--only` | Learning question | Generated files |
|---|---|---|
| `computational_graph` | What is computed forward, backward and during the optimizer step? | `computational_graph.gif` |
| `chain_rule` | Why multiply local derivatives, and how local is a derivative? | `chain_rule.gif`, `local_gradient.png` |
| `loss_landscape` | Does the actual gradient move toward lower loss on this model? | `loss_landscape_3d.html`, `loss_contours.png`, `gradient_descent.gif` |
| `gradient_flow` | What happens to an upstream derivative after repeated scalar factors? | `gradient_flow.gif`, `gradient_magnitudes.png` |
| `activation_gradients` | Where do sigmoid, tanh and ReLU have small local derivatives? | `activation_derivatives.png`, `activation_gradients.gif` |
| `learning_rates` | How does changing only the learning rate affect updates? | `learning_rate_comparison.gif`, `learning_rate_curves.png` |
| `gradient_check` | Why does shrinking epsilon eventually harm a numerical derivative? | `gradient_check.png` |
| `training_engineering` | How do clipping and unequal-microbatch accumulation change gradients? | `training_engineering.png` |

Blue denotes forward values, orange backward sensitivities, green optimizer
updates and red very small or amplifying signals. The learning-rate plots
use separate rate colors with explicit legends; colors do not assert stability.

## Mathematical contracts

The animated single-observation graph has a bias:

$$
z=w_1x+b,\quad h=\tanh(z),\quad \hat y=w_2h,
\quad L=\tfrac12(\hat y-y)^2.
$$

With $r=\hat y-y$, the local factors for $\partial L/\partial w_1$ are
$r$, $w_2$, $1-h^2$ and $x$. Its complete parameter gradient is

$$
\nabla_{(w_1,b,w_2)}L=
\left(rw_2(1-h^2)x,\; rw_2(1-h^2),\; rh\right).
$$

Every backward frame uses the same weights. The green optimizer phase applies
one simultaneous update at learning rate 0.2 and recomputes the forward loss.
The chain-rule animation and local-perturbation PNG use those same initial
parameters. Their model differs from the shared-path scalar example in the
earlier analytical study.

The surface and learning-rate modules share a **two-parameter model without
bias** and exactly the same 41 observations:

$$
x_i\in[-1,1],\quad y_i=\sin(3x_i)+0.2x_i,\quad
h_i=\tanh(w_1x_i),\quad r_i=w_2h_i-y_i,
\quad L=\frac{1}{2N}\sum_i r_i^2.
$$

$$
\frac{\partial L}{\partial w_1}
=\frac1N\sum_i r_iw_2(1-h_i^2)x_i,\qquad
\frac{\partial L}{\partial w_2}=\frac1N\sum_i r_ih_i.
$$

All weights update together using gradients at the previous weights. Trajectory
heights evaluate $L$ at the recorded weights directly, independent of grid
resolution. The contour GIF uses rate 0.1; the HTML and learning-rate
comparison include rates 0.001, 0.01, 0.1 and 1.0. These are experimental
settings, with shared initialization $(0.5,0.8)$ and no validation-based tuning.
The model has the symmetry $L(w_1,w_2)=L(-w_1,-w_2)$; the surface is not a
generic convex bowl.

The depth module uses a controlled scalar chain $u_\ell=c u_{\ell-1}$,
upstream derivative 1 and factors $c=0.5,1,1.5$. After traversing $k$ links,
the magnitude is $c^k$. It is an exact product demonstration, **not measured
gradients from a trained deep network**. Real Jacobian products depend on
directions, inputs and weights and need not change monotonically.

Activation plots shade $|f'(z)|<0.05$ as an explanatory threshold. This is
not a convergence criterion. ReLU uses derivative 0 at zero as an explicit
convention; its classical derivative does not exist there.

The numerical-gradient panel checks both coordinates at $(0.7,1.2)$ with
24 logarithmically spaced epsilon values from $10^{-1}$ to $10^{-12}$.
Central differences trade truncation against cancellation and rounding.
The plotted relative error divides absolute error by
$\max(|g_{\mathrm{analytical}}|,10^{-15})$; zero plotted errors are floored at
$10^{-18}$ and the display states that policy.

Clipping uses the actual initial full-batch gradient $g$, threshold
$\tau=0.6\|g\|_2$, and
$g_{\mathrm{clip}}=g\min(1,\tau/\|g\|_2)$, with a zero-vector guard. The
relative threshold intentionally makes clipping visible; it is not a tuned
training recommendation. Clipping changes the gradient supplied to an
optimizer and does not fix disconnected or vanishing gradients.

Accumulation partitions the 41 rows into microbatches of sizes 13 and 28,
without updating weights between them. If each microbatch returns a mean-loss
gradient $g_k$, the full-batch gradient is
$(13/41)g_1+(28/41)g_2$. Summing mean gradients changes their scale; averaging
them equally also gives the wrong observation weights here. Two accumulation
steps have effective batch size 41 in this example. Equal-size batches of
size $m$ over $K$ accumulation steps would have effective size $Km$.
Batch-dependent layers or updates between batches break the stated equivalence.

## Inspect the offline 3D explorer

Download/open [the selected standalone HTML](visuals/loss_landscape_3d.html)
locally. GitHub shows HTML source rather than running this explorer inline.
Drag to rotate, scroll to zoom, and hover trajectory points to see iteration,
weights and loss. Diamond markers identify initial states and larger circles
final states. Click a legend entry to hide/show a rate; double-click to
isolate it. **Reset 3D camera** restores the starting view and supports the
keyboard. Plotly's JavaScript is embedded; no CDN or server is needed.

## Executed visual phase — 2026-11-18

This is a separate visual-phase record. The earlier analytical run in
[notes.md](notes.md#executed-study--2026-11-18) used a `1 → 8 → 1` network
and 128 training inputs; its losses are not directly comparable to this
two-parameter, 41-input experiment. Historical analytical records are preserved.

Codex ran the default `visual_lab.py --all` on Windows x64 using Python
3.11.14, NumPy 2.4.6, Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0.
Configuration: float64, no randomness, 60 requested frames, 91×91 grid,
41 synthetic observations, initial weights $(0.5,0.8)$, 180 updates per rate.
All 14 exports were nonempty and decoded successfully. Panel layouts
were then refreshed in their individual modules; numerical settings were unchanged.
Distinct dated manifests remain ignored in `visuals/run-*.json`.

**Hypothesis:** the manual derivatives agree with independent numerical
derivatives and generate actual, reproducible loss trajectories.

| Check or experiment | Observed result |
|---|---|
| Manual vs central difference, epsilon $10^{-5}$ at $(0.7,1.2)$ | Maximum absolute error 8.749335e-12 |
| Biased scalar graph | Initial loss 0.003519517; $\nabla_{(w_1,b,w_2)}L=(-0.065644756,-0.082055945,-0.036083612)$ |
| One simultaneous scalar update at rate 0.2 | New loss 0.001491836 |
| Clipping at $0.6\|g\|_2$ | Norm 0.242410099 → 0.145446059 |
| Weighted microbatch means vs full-batch gradient | Maximum absolute difference 0.0 in this float64 run |
| Optional PyTorch parity | Not performed: PyTorch was not installed |

**Hypothesis:** changing only learning rate changes the observed fitting path
and step sizes; no particular stable or unstable outcome is assumed.

| Learning rate | Initial half-MSE | Half-MSE after 180 updates | Loss increases across all updates |
|---:|---:|---:|---:|
| 0.001 | 0.197710271 | 0.187265470 | 0 |
| 0.01 | 0.197710271 | 0.113343518 | 0 |
| 0.1 | 0.197710271 | 0.055086045 | 0 |
| 1.0 | 0.197710271 | 0.024283618 | 0 |

**Interpretation candidates — author review required:** manual gradients agree
at the checked points; the stated batch weighting reproduces the desired
gradient; all four rates decreased loss in this particular short run. This
does not establish an optimal rate or promise stable training elsewhere.
No divergence was observed or artificially introduced. For the controlled
chain, the input magnitudes at depth 16 are exactly $0.5^{16}$, 1 and
$1.5^{16}$; this says nothing about a trained deep model's empirical norms.

**Limitations:** these are tiny synthetic fitting and identity exercises, with
no independent holdout, architecture comparison, optimizer benchmark, mixed
precision, distributed training, native-GPU measurement or production claim.
Successful numerical tests do not approve the interpretations for the author.

Chrome 157.0.8103.79 was exercised through Playwright CLI 0.1.24, headless,
at 1440×1100, with the browser offline and direct `file:` navigation. Real
mouse rotation/zoom, reset, legend hide/show and keyboard reset worked.
Independent JavaScript calculations checked all 8,281 surface cells and all
724 trajectory positions: maximum loss disagreements were 2.22e-15 and
8.33e-17 respectively. Hover data were inspected. There were zero HTTP(S)
requests and zero page/console errors; one Canvas2D readback-performance
warning was observed. This is a desktop Chrome check, not a cross-browser,
mobile, full accessibility, native-GPU or GitHub-rendering audit.

## Preview selection

The topic README embeds only these three inspected GIFs:

- [Computational graph](visuals/computational_graph.gif): 314,039 bytes;
  14 distinct forward/backward/update phases, including bias gradients.
- [Gradient descent](visuals/gradient_descent.gif): 726,805 bytes;
  60 displayed checkpoints on the actual contour field.
- [Gradient flow](visuals/gradient_flow.gif): 359,236 bytes;
  17 distinct backward-depth states for the exact scalar products.

Together they use 1,400,080 bytes (about 1.34 MiB), within the repository's
inline-preview budget. The linked offline HTML is 5,076,467 bytes (about
4.84 MiB), including its runtime. These four files are deliberately unignored
Git candidates. The other ten exports, dated manifests, quick/custom outputs
and inspection screenshots remain ignored. No artifacts were committed.

## Recheck without overwriting previews

```powershell
# Run from the repository root. Real export tests use pytest's temporary directory.
python scripts/validate_repo.py tests --scope 04-deep-learning/54-backpropagation
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
```

The CLI also validates both manual gradient coordinates before rendering and
decodes every generated raster frame afterward. The tests cover every
trajectory's objective values and updates, graph bias gradients, surface
orientation, activation derivatives, clipping, loss reduction and unequal
microbatches. A real export smoke generates all modules at reduced display
resolution in a temporary directory. Browser checks remain separate from
these numerical and raster checks.

The final local topic run passed **60 tests** (27 analytical and 33 visual/CLI,
including real temporary-directory exports), with no skips or warnings.
Repository syntax parsed 369 Python files and internal links checked 1,276
targets across 247 Markdown files. Dependency compatibility and whitespace
checks passed. This was scoped local validation; the full curriculum suite
and Linux CI were not run for this phase.

Implementation references:
[Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)
and [Plotly standalone HTML export](https://plotly.com/python/interactive-html-export/).
