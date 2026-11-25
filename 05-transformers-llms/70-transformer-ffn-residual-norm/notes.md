# Transformer feature transformations and signal propagation

## Position-wise does not mean context-free

For row-vector token features $x\in\mathbb{R}^d$:

$$
F(x)=\phi(xW_1+b_1)W_2+b_2,
\qquad W_1\in\mathbb{R}^{d\times h},\quad W_2\in\mathbb{R}^{h\times d}.
$$

The same function acts on every $(b,t)$ position in a $(B,T,d)$ tensor.
Changing one token at the FFN input cannot directly affect another output token.
Its input can already contain context supplied by attention. Parameters are
shared across positions within a layer, but separate layers have separate
parameters. This is the position-wise construction described in the
[original Transformer, section 3.3](https://arxiv.org/html/1706.03762v7).

Removing $\phi$ gives one affine map with weight $W_1W_2$ and bias
$b_1W_2+b_2$. Extra intermediate width alone does not introduce nonlinearity.
The implementation uses

$$
\operatorname{GELU}_{\tanh}(z)=\tfrac12 z\left[1+\tanh\left(
\sqrt{2/\pi}(z+0.044715z^3)\right)\right].
$$

This approximates $z\Phi(z)$; comparing it with exact-erf GELU would conflate
an approximation difference with an implementation error. PyTorch uses
`approximate="tanh"` explicitly here.

## LayerNorm axes, moments and lost information

For one token, with population variance:

$$
\mu=\frac1d\sum_i x_i,\qquad
v=\frac1d\sum_i(x_i-\mu)^2,\qquad
\hat{x}_i=\frac{x_i-\mu}{\sqrt{v+\epsilon}},\qquad
y_i=\gamma_i\hat{x}_i+\beta_i.
$$

Only the last feature axis is reduced. Other tokens, batch members and padding
do not enter these statistics. Padding still needs masking in attention and
losses. PyTorch's [`LayerNorm(d)`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)
uses population variance and the same computation in training and evaluation,
with feature-wise affine parameters by default.

Before affine transformation the mean is zero up to rounding and the variance
is $v/(v+\epsilon)$, not exactly one. Constant vectors map to $\beta$; with
$d=1$ every input does. Arbitrary learned $\gamma$ and $\beta$ can change both
moments. Standardizing does not create a Gaussian distribution.

Adding a scalar offset to every feature leaves the standardized vector
unchanged. Positive rescaling is only approximately invariant with nonzero
$\epsilon$, especially near zero variance. Mean cannot be recovered from the
normalized vector alone. With finite epsilon the scale invariance is not exact;
it would be inaccurate to claim exact loss of all scale information. Fixed
affine parameters cannot restore arbitrary per-token discarded statistics.
An additive residual path offers another route around normalization but does
not make the entire block information-preserving or invertible.

## Backward signal and normalization placement

For $y=x+F(x)$, $J_y=I+J_F$. Backpropagation applies the transpose to the
upstream gradient. The identity contribution can be cancelled by $J_F=-I$;
positive amplification can also compound across depth. Residual learning
facilitates optimization without guaranteeing it; see
[He et al.](https://arxiv.org/abs/1512.03385).

For one sublayer, evaluating Jacobians at the appropriate inputs:

$$
J_{\mathrm{pre}}=I+J_F J_{\mathrm{LN}},\qquad
J_{\mathrm{post}}=J_{\mathrm{LN}}(I+J_F).
$$

Pre-LN leaves an identity term outside normalization. In Post-LN even the
identity contribution passes through the LayerNorm Jacobian. For an upstream
vector $u$, let $g=u\odot\gamma$ and let the bar denote the feature mean. The
implemented input vector-Jacobian product is

$$
J_{\mathrm{LN}}^T u=\frac{1}{\sqrt{v+\epsilon}}
\left(g-\bar g-\hat{x}\,\overline{g\hat{x}}\right).
$$

This includes epsilon and holds separately at each token. Its sum over input
features is zero: perturbations along the all-ones direction have no effect.
The tests compare this expression against coordinate finite differences and
PyTorch autograd, including width-one and constant-vector cases.

[Xiong et al.](https://proceedings.mlr.press/v119/xiong20b.html) analyze
normalization placement, parameter gradients at initialization and warmup under
their assumptions, with task experiments. Their findings motivate investigating
Pre-LN optimization, rather than a universal rule that warmup is unnecessary.
Our state-gradient profiles and random MSE are different measurements and do
not reproduce that study.

A deep Pre-LN residual stream can accumulate magnitude; progressively smaller
relative updates are also possible. A final normalization affects the loss
boundary. Applying it changes the incoming gradient to the last block, even
when every internal forward state is unchanged. Larger gradients are not
automatically more useful gradients.

## Parameter and compute budgets

A dense FFN has $2dh+h+d$ parameters including biases. Conventional full-width
Q/K/V/output attention projections have $4d^2+4d$; two affine LayerNorms add
$4d$. At $h=4d$, the FFN accounts for two thirds of **attention plus FFN weight
matrices**, excluding biases, norms, embeddings and readout. This fraction
changes with gating, grouped-query attention, tied weights or different widths.
For the executed $d=64,h=256$ blocks, there are 49,984 parameters per block and
1,199,616 across 24 blocks. They remain identical across ablations.

For one sequence in dense full-sequence processing, attention costs
$O(Td^2+T^2d)$ and FFN costs $O(Tdh)$. Normalization and residual addition
each cost $O(Td)$. These are scaling expressions, not measured runtime.
Cached autoregressive decoding has a different workload. Real serving depends
on precision, memory traffic, batching, kernels and hardware.

Two extensions are useful for theory but **not implemented or measured in the
analytical scripts** (the separate visual lab implements both):

- RMSNorm uses $\gamma_i x_i/\sqrt{d^{-1}\sum_j x_j^2+\epsilon}$ without mean
  subtraction. It is batch-independent like LayerNorm, but not shift-invariant.
  [Zhang and Sennrich](https://arxiv.org/abs/1910.07467) study its efficiency;
  this lab makes no local speed claim. BatchNorm instead normally uses batch
  statistics during training and running estimates at inference.
- SwiGLU uses $[\operatorname{SiLU}(xW_g)\odot(xW_u)]W_d$. Three weight
  projections cost approximately $3dh$ instead of $2dh$. Matching a dense
  $4d$ FFN gives gated width near $8d/3$, before rounding and biases.
  [Shazeer](https://arxiv.org/abs/2002.05202) evaluates gated variants;
  a larger gate is not evidence of better quality for an arbitrary task.

MoE adds routing and selected experts rather than changing this dense block
into a cheap drop-in replacement. Total weights, active weights, communication
and utilization need separate accounting.

## Execution record — 2026-11-25

**Executor:** Codex, local Windows CPU. **Interpretation status:** author review
required. These are new analytical runs, not results from the source study or
the previous day's experiments.

**Hypothesis:** residual identity paths and normalization placement alter
initialization-time activation/gradient propagation. A common final
normalization should also change loss-boundary gradients without changing
internal states. No convergence or quality hypothesis was tested.

**Configuration:** Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu; float64,
one PyTorch thread; seeds 70/71/72. Each stack has 24 independently initialized
blocks, $d=64$, four heads, $h=256$, tanh GELU and $\epsilon=10^{-5}$.
PyTorch module default initialization is copied into all four conditions within
each seed; corresponding parameters were verified identical. Two batches of
eight synthetic tokens and an independent standard-normal target use a local
CPU generator seeded at `seed + 1000`. The objective is mean MSE over all 1,024
elements. Attention is bidirectional with no masks. There is no positional
encoding, dropout, optimizer, learned task head, dataset or train/test split.

The first invocation uses the raw last-block output. The second applies the
same **non-affine final LayerNorm** to all four conditions. This is a controlled
readout intervention, not a comparison between complete production models.
Both have zero parameter updates.

Commands from the repository root, using the stable interpreter documented
in the repository's local execution workflow:

```powershell
$py = '.local/environments/n09-full-2026-11-13/Scripts/python.exe'
& $py 05-transformers-llms/70-transformer-ffn-residual-norm/from_scratch.py
& $py 05-transformers-llms/70-transformer-ffn-residual-norm/example.py --output 05-transformers-llms/70-transformer-ffn-residual-norm/outputs/2026-11-25_initialization_raw_readout.json
& $py 05-transformers-llms/70-transformer-ffn-residual-norm/example.py --final-norm --output 05-transformers-llms/70-transformer-ffn-residual-norm/outputs/2026-11-25_initialization_final_norm.json
```

The JSON records retain timestamps, configuration, versions and every block's
activation RMS, state-gradient L2 norm and FFN expansion-weight-gradient L2
norm. They stay ignored; the intentional public evidence is the following
text summary. Filenames are provenance labels, not public asset links. Existing
records must be kept; choose new filenames for subsequent executions.

### Raw readout results

Here $g_0$ is the gradient with respect to the input tensor. $g_{24}$ is the
gradient with respect to the raw last-block output, before any readout.
The ratio is $\|g_0\|_2/\|g_{24}\|_2$, not a Jacobian spectral norm.
RMS and gradient norms summarize all tensor elements, not individual tokens.

| Seed | Condition | Output RMS | Mean MSE | Input gradient L2 | Gradient ratio |
|---:|---|---:|---:|---:|---:|
| 70 | Pre-LN + residual | 1.921564 | 4.593947 | 0.510947 | 3.814192 |
| 70 | Post-LN + residual | 0.999995 | 2.011714 | 0.078245 | 0.882656 |
| 70 | Pre-LN, no residual | 0.162403 | 1.016114 | 0.023599 | 0.374579 |
| 70 | Post-LN, no residual | 0.999814 | 1.879957 | 0.105110 | 1.226569 |
| 71 | Pre-LN + residual | 1.903370 | 4.462117 | 0.526943 | 3.991288 |
| 71 | Post-LN + residual | 0.999995 | 1.900560 | 0.080690 | 0.936476 |
| 71 | Pre-LN, no residual | 0.195111 | 1.026906 | 0.015036 | 0.237409 |
| 71 | Post-LN, no residual | 0.999865 | 1.986119 | 0.110585 | 1.255486 |
| 72 | Pre-LN + residual | 1.930206 | 4.721153 | 0.517624 | 3.811628 |
| 72 | Post-LN + residual | 0.999995 | 1.980984 | 0.062400 | 0.709353 |
| 72 | Pre-LN, no residual | 0.244741 | 1.095254 | 0.003373 | 0.051564 |
| 72 | Post-LN, no residual | 0.999916 | 2.099340 | 0.010048 | 0.110956 |

### Shared final normalization results

The raw internal activation RMS values are unchanged from the first run.
The gradient denominator is still measured at that raw last-block state,
so it includes the effect of backpropagating through the added readout norm.

| Seed | Condition | Mean MSE | Input gradient L2 | Gradient ratio |
|---:|---|---:|---:|---:|
| 70 | Pre-LN + residual | 1.947412 | 0.068045 | 2.075549 |
| 70 | Post-LN + residual | 2.011714 | 0.078245 | 1.270191 |
| 70 | Pre-LN, no residual | 1.890903 | 0.117787 | 0.306951 |
| 70 | Post-LN, no residual | 1.880296 | 0.105015 | 1.694472 |
| 71 | Pre-LN + residual | 1.917211 | 0.079650 | 2.404688 |
| 71 | Post-LN + residual | 1.900559 | 0.080690 | 1.311528 |
| 71 | Pre-LN, no residual | 1.998605 | 0.176616 | 0.552541 |
| 71 | Post-LN, no residual | 1.986379 | 0.110731 | 1.802816 |
| 72 | Pre-LN + residual | 2.003593 | 0.073569 | 2.243754 |
| 72 | Post-LN + residual | 1.980983 | 0.062400 | 1.001103 |
| 72 | Pre-LN, no residual | 2.101213 | 0.004850 | 0.019027 |
| 72 | Post-LN, no residual | 2.099505 | 0.010036 | 0.161454 |

### Numerical controls and interpretation candidates

Copied-weight NumPy/PyTorch comparisons at seed 70 gave maximum absolute errors:
LayerNorm forward $8.882\times10^{-16}$, input VJP
$4.441\times10^{-16}$ and FFN $4.441\times10^{-15}$.
The small NumPy demonstration preserved $(2,3,8)$ shapes; pre-affine LayerNorm
variance ranged from 0.999984 to 0.999990, and maximum absolute feature mean
was $6.939\times10^{-17}$.

All measured states and parameter gradients were finite in both three-seed
runs. Forty focused tests passed, including an independent official PyTorch
`TransformerEncoderLayer` forward/backward comparison for both norm orders,
finite differences for LayerNorm, exact residual shapes and repeatability.

**Candidate interpretations — author review required:**

- The Post-LN near-unit output RMS is consistent with the final affine norm
  initialized at scale one and bias zero; it is not evidence of useful features.
- The larger Pre-LN residual RMS is consistent with additive accumulation in
  this configuration. No monotonic growth claim is inferred from endpoints.
- Shared final normalization substantially changes Pre-LN input gradients.
  The raw-readout gradient difference therefore includes a loss-boundary effect.
- Removing residuals does not universally reduce the input gradient: raw
  Post-LN seeds 70 and 71 have larger input gradients after removal. Initialization
  and the loss shape matter; the textbook identity-path argument is not a
  promise about a particular scalar norm.

**Limitations:** three seeds, one width/depth, synthetic random target and no
training. MSE differences can reflect output scale; lower random-target MSE is
not model quality. No learning-rate schedule, mixed precision, semantic task,
runtime benchmark or production environment was tested. The NumPy functions
are educational float64 formulas, not overflow-hardened or optimized kernels
for arbitrary extreme finite inputs. No GitHub math rendering was inspected.

## Follow-up proposals at the analytical handoff

Depth/width sweeps, residual scaling, trained task comparisons with fixed
budgets and separate validation/test sets, RMSNorm, matched-parameter SwiGLU,
mixed precision and latency measurements were unexecuted proposals at that
handoff. Specify hypotheses,
readout, initialization, optimizer, seed budget and evaluation before running
them. This analytical study has no visual exports.

The subsequent [visual laboratory](VISUAL_LAB.md) has a separate execution
protocol and seed family. It implements depth/normalization diagnostics and
signed gating/geometry demonstrations; its new measurements do not replace
the dated analytical results above. Training, mixed precision, quality and
latency comparisons remain unexecuted proposals.
