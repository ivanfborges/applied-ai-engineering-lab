# Initialization as control of signal propagation

## Moments, assumptions and fan sizes

For column-vector notation, let $z_l=W_la_{l-1}$ and
$a_l=\phi(z_l)$, with $W_l$ shaped $n_{out}\times n_{in}$.
The implementations use batches of row vectors: `h @ W.T`, matching PyTorch
`Linear.weight`. Passing the transposed layout to a fan-based initializer
without adjusting its interpretation swaps the fan sizes.

At initialization, suppose weights are independent, zero mean, independent
of incoming activations, have common variance $\sigma_w^2$, and biases are zero.
For common incoming second moment $q=\mathbb E[a_i^2]$:

$$
\operatorname{Var}(z_j)=n_{in}\sigma_w^2q.
$$

Cross terms vanish through the weight assumptions; activations need not have
zero mean. The expectation is over the initialization/input ensemble. A finite
network's pooled batch-and-feature measurement is only a diagnostic estimate.
After learning, independence assumptions need not hold.

For linear, centered signals, forward preservation suggests weight variance
$1/n_{in}$; backward preservation suggests $1/n_{out}$. The unit-gain Glorot
compromise is

$$
\sigma_w^2=\frac{2}{n_{in}+n_{out}}.
$$

Normal initialization uses its square root as the standard deviation. The
uniform counterpart uses bounds $\pm\sqrt{6/(n_{in}+n_{out})}$.
A gain $g$ multiplies the standard deviation and therefore multiplies the
variance by $g^2$. Tanh networks can use activation-specific gains, but saturation
still invalidates a simple linear approximation. This study uses gain 1.

For a symmetric pre-activation $Z$, ReLU keeps half the **second moment**:

$$
\mathbb E[\max(0,Z)^2]=\tfrac12\mathbb E[Z^2],\qquad
q_l\approx\tfrac12n_{in}\sigma_w^2q_{l-1}.
$$

Setting the approximate multiplier to one gives He fan-in variance $2/n_{in}$.
ReLU does not halve the centered variance: for $Z\sim\mathcal N(0,v)$,
$\mathbb E[\operatorname{ReLU}(Z)]=\sqrt{v/(2\pi)}$ and its variance is
$v(1/2-1/(2\pi))$. The report keeps mean and centered variance separately.

For equal-width layers, unit-gain Xavier gives an idealized ReLU multiplier of
$1/2$ and He gives 1. The approximation $2^{-20}=9.536743\times10^{-7}$ is a
theoretical relative second moment, not a measured result. With paired matrices
and no normalization/bias, positive homogeneity gives an additional **exact
scale identity**: $q_l^{Xavier}=2^{-l}q_l^{He}$, up to floating-point error.
That identity holds for each paired realization even if both depart from
their ensemble predictions. It is tested without treating it as empirical
support for convergence.

## Backward propagation and what the loss confounds

The activation adjoint for a ReLU block satisfies

$$
g_{l-1}=W_l^{\top}\left(g_l\odot\mathbf 1_{z_l>0}\right).
$$

Under further approximate independence assumptions, the componentwise backward
second-moment multiplier is $n_{out}\sigma_w^2/2$. Kaiming `fan_out` prioritizes
this backward scale; `fan_in` prioritizes the forward scale. They coincide for
square layers here and differ for changing widths. Neither preserves the norm
of every direction. With rectangular matrices, component variance and total
vector norm are also different quantities.

For $L=\frac1{BD}\sum_{b,j}a_{depth,bj}^2$, with batch size $B$ and width $D$,

$$
\frac{\partial L}{\partial a_{depth}}=\frac{2a_{depth}}{BD}.
$$

Thus a smaller output supplies a smaller terminal loss gradient before any
Jacobian multiplication. Comparing only this gradient mixes output scale
with backward propagation. The second diagnostic uses a shared Gaussian
output adjoint $u$, normalized to RMS 1, and computes $J^{\top}u$ directly via
`autograd.grad`. It can be viewed as the derivative of
$S=\sum_{b,j}u_{bj}a_{depth,bj}$, with $u$ held fixed. There is no batch-mean
division in this probe. Its output gradient RMS is exactly 1 within numerical
precision, so the input RMS shows amplification/attenuation of this direction.

One random direction is not an estimate of all singular values or a guarantee
of a well-conditioned Jacobian. Activation gradients, weight gradients and
optimizer updates are distinct. No optimizer is instantiated, and diagnostics
leave parameters and `.grad` buffers unchanged.

## Normalization changes derivatives too

For a selected group $\mathcal G$:

$$
\hat z_i=\frac{z_i-\mu_{\mathcal G}}{\sqrt{v_{\mathcal G}+\epsilon}},\qquad
y_i=\gamma_i\hat z_i+\beta_i.
$$

Here LayerNorm computes population variance across all features of each example,
with no running statistics. It uses the same rule in training and evaluation.
Before ReLU, variance is $v/(v+\epsilon)$ at the initial affine settings, rather
than exactly 1. After ReLU, the second moment can be near $1/2$ under roughly
symmetric normalized values; it need not match the unnormalized He moment.
Constant-feature inputs yield zero normalized values; width 1 is degenerate.

The normalization Jacobian couples features through their mean and variance.
Controlling the forward magnitude does not imply an identity Jacobian or stable
gradients in every direction. Placement before versus after ReLU changes the
model. LayerNorm also adds trainable scale/shift parameters, so this comparison
changes the computation as well as the initialization context.

BatchNorm instead uses batch statistics per feature/channel during ordinary
training, typically tracking running statistics for evaluation. Small batches,
distribution shifts and mode selection matter. This topic explains that
distinction but executes only LayerNorm; the
[Day 57 mechanism tests](../57-regularization/tests/test_regularization.py)
cover BatchNorm conventions separately. Do not interpret this run as a BN/LN
performance comparison.

## Engineering decisions and common mistakes

- Initialize hidden weight vectors differently to break symmetry. Zero biases
  can be fine; making all hidden weights identical prevents feature diversity.
  Selected zero residual projections are an architecture-specific exception.
- Match the full configuration: gain, fan mode, matrix orientation and activation.
  With square layers, Xavier with gain $\sqrt2$ has the same variance as He.
  This study compares Xavier with **gain 1** only.
- A new ReLU hidden head and a final logit projection have different semantics.
  Check initial logits and task loss; do not apply a rectifier rule blindly to
  all pretrained Transformer projections or reinitialize learned representations.
- Record the first non-finite value: input, pre-activation, loss, gradient or
  update. Initialization, learning rate, precision, loss formulation and data
  validity can fail independently. Float64 here reduces numerical range issues;
  it does not validate float32, FP16 or GPU behavior.
- Gradient clipping acts after backpropagation and limits an update's gradient
  input. It cannot restore directions already attenuated. Loss scaling addresses
  representational underflow in applicable precision regimes, not a mathematical
  vanishing derivative.
- Finite activation/gradient RMS can hide rank deficiency, saturation or
  anisotropy. Orthogonal maps and residual paths address additional aspects,
  with architecture-specific trade-offs.

## Executed study — 2026-11-18

**Executor:** Codex. **Interpretation status: author review required.**
These are newly executed analytical diagnostics, with no prior Day 58 record
replaced. No visual phase or supervised experiment was performed.

**Hypothesis:** repeated changes in weight scale compound through ReLU blocks;
normalization controls forward statistics while changing backward sensitivity.

**Configuration:** Windows x64, Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu,
float64, seed 58, batch 128, width 128, depth 20, zero optimizer steps. The
PyTorch script sets one numeric thread and uses weight-generator seed 59.
Inputs and output probes are shared within that script. Xavier gain is 1;
He is normal/fan-in/ReLU. Small/large standard deviations are 0.06/0.18.
LayerNorm is before ReLU, epsilon $10^{-5}$, initial gamma 1 and beta 0.
All inputs are synthetic independent standard Gaussians; there are no labels,
public dataset, training split or holdout metrics.

Commands from the repository root, using the stable interpreter documented
locally (the equivalent portable commands are in the README):

```powershell
python 04-deep-learning/58-weight-initialization/from_scratch.py
python 04-deep-learning/58-weight-initialization/example.py --output outputs/day58/2026-11-18-initialization-baseline.json
```

**Results, rounded from terminal/JSON:**

| PyTorch configuration | $q_{20}/q_0$ | Squared-output loss | Input loss-gradient RMS | Input fixed-adjoint RMS |
|---|---:|---:|---:|---:|
| Small normal | 5.800204e-14 | 5.8342e-14 | 5.362989e-17 | 2.944010e-07 |
| Xavier normal | 3.113106e-07 | 3.1313e-07 | 2.878442e-10 | 6.820474e-04 |
| He normal | 3.264328e-01 | 3.2834e-01 | 3.018266e-04 | 6.984166e-01 |
| He + LayerNorm | 5.018424e-01 | 5.0478e-01 | 3.820037e-05 | 7.923937e-01 |
| Large normal | 7.051694e+05 | 7.0930e+05 | 6.520143e+02 | 1.026513e+03 |

Output fixed-adjoint RMS was 1 for all five configurations. All recorded
measurements were finite. The JSON retains the entire profile, zero fractions
and linear-weight loss-gradient RMS. It remains ignored under `outputs/day58/`,
and is not a linked public asset.

The separately seeded NumPy-generator realization gave $q_0=1.01229941$,
Xavier $q_{20}=1.19404227\times10^{-6}$ and He $q_{20}=1.25204407$.
Relative second moments were $1.17953469\times10^{-6}$ and 1.23683176,
respectively. NumPy and PyTorch seeds do not imply identical random draws;
their absolute endpoints must not be pooled or presented as framework agreement.
Framework agreement is tested using explicitly identical inputs and weights.

**Interpretation candidates — author review required:** this realization is
consistent with under-scaled and over-scaled propagation. He reduces the
systematic attenuation of unit-gain Xavier/ReLU but does not force exact moment
preservation at finite width. LayerNorm controls the final forward scale while
its loss-gradient and probe-gradient behavior differ; the two diagnostics answer
different questions. All these statements concern initialization only.

**Limitations:** one seed, Gaussian inputs, fixed width, no bias or residual paths,
one output direction, float64 CPU arithmetic, no learning or generalization
measurement. A finite large-scale result is amplification, not observed training
divergence. LayerNorm's smaller loss-gradient RMS is not proof of worse
trainability. These numbers establish no general ranking or production benefit.

**Local verification on this date:** scoped tests passed 32 cases in one isolated
file, without skips or warnings. The public runner parsed 396 Python files and
checked 1,387 internal links across 258 Markdown files. `git diff --check` passed.
The stable interpreter had no `pip` module, so `python -m pip check` could not
run successfully. The read-only alternative
`uv pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe --offline --no-cache`
checked 78 installed packages and reported compatibility. No packages were
installed or upgraded. This is local evidence, not a complete curriculum test
run, Linux CI, GitHub math-render review or author scientific approval.

## Proposed follow-ups — not executed

This label preserves the original analytical phase's scope. The separate
[visual laboratory](VISUAL_LAB.md) subsequently implements repeated propagation,
normalization and a supervised task under its own settings. Those runs do not
replace or extend the historical numerical table above, and do not establish
that every proposed ablation below was completed.

1. Repeat paired diagnostics at depths 5/20/50 and at least five seeds; report
   dispersion of moments and fixed-adjoint RMS without selecting a favorable seed.
2. Compare tanh with matched gain and measure saturation fractions. Add changing
   widths to distinguish forward/fan-in from backward/fan-out objectives.
3. Compare BatchNorm and LayerNorm with explicit train/eval modes and batch sizes
   2/32/128. Keep axis choice, affine parameters and placement explicit.
4. Train a known synthetic prediction task with fixed splits, learning rate and
   budgets across seeds. Measure validation loss, non-finite updates and gradient
   norms before making any trainability claim. Clipping is a separate ablation.

## Interview questions

**Derive He initialization without confusing variance and second moment.**
Use zero-mean independent weights and symmetric pre-activations: a linear sum
multiplies the incoming second moment by $n_{in}\sigma_w^2$, and ReLU keeps
half. Set that combined factor to one. ReLU's output mean is positive, so its
centered variance is a different statistic.

**Why can He keep signal scale reasonable while learning still fails?**
Average scale does not constrain every singular value of a product of Jacobians.
Architecture, loss, saturation, data and update rules also affect optimization.
Inspect layer profiles and parameter updates, then test actual validation behavior.

**What does the fixed-adjoint probe add?**
It removes variation in the loss's output derivative from this comparison.
It measures one vector-Jacobian product, not convergence or the full spectrum.

**Does normalization repair bad initialization?**
It changes scale and derivatives but cannot create feature diversity from
identical hidden weights or guarantee well-conditioned gradients. It introduces
axis, epsilon, affine and possibly running-statistics choices that need validation.

**How would you select an initialization for a pretrained model's new head?**
Preserve learned weights, inspect input scales, match hidden-layer activation
and output semantics, then compare paired seeds under a fixed training protocol.
Initial diagnostics are a debugging tool; selection requires task validation.

## Primary references

- [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html):
  initialization, activation/gradient propagation and conditioning motivation.
- [He et al. (2015)](https://arxiv.org/abs/1502.01852): rectifier-aware
  initialization. This study does not reproduce its ImageNet results.
- [Ioffe and Szegedy (2015)](https://proceedings.mlr.press/v37/ioffe15.html):
  original BatchNorm method. Its title's proposed explanation is not tested here.
- [Ba, Kiros and Hinton (2016)](https://arxiv.org/abs/1607.06450): LayerNorm.
- [PyTorch initialization API](https://docs.pytorch.org/docs/2.15/nn.init.html)
  and [LayerNorm API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html):
  version-matched interfaces, fan orientation and normalization conventions.
