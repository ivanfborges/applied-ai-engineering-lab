# Activation functions: signals, slopes and coupled outputs

## Why depth needs nonlinearity

Two affine layers give $W_2(W_1x+b_1)+b_2=(W_2W_1)x+W_2b_1+b_2$.
Depth alone adds no nonlinear representation. With hidden activations,
$h_l=f(W_lh_{l-1}+b_l)$, the model can express different mappings, but
representational capacity does not prove optimization success or generalization.

For an elementwise activation, the local Jacobian is diagonal. A network's
gradient involves products of these Jacobians **and weight matrices**. A
small local derivative is one mechanism of attenuation. The analytical study
below isolates local derivatives; the separate [visual lab](VISUAL_LAB.md)
adds controlled chains and small trained networks without isolating every
source of unstable gradients.

## Values and derivatives

Write $s=\sigma(x)$, $u=\tanh(x)$, and let $\Phi,\phi$ be the standard-normal
CDF and density. The first five functions operate independently per element.

| Function | Value | Derivative away from kinks | Main trade-off |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | $s(1-s)$ | Bounded gate or binary output; saturation on both tails |
| Tanh | $\tanh(x)$ | $1-u^2$ | Signed, centered output; still saturates |
| ReLU | $\max(0,x)$ | $0$ for $x<0$, $1$ for $x>0$ | Simple rectification; inactive units can receive no local gradient |
| Leaky ReLU | $x$ for $x>0$, $\alpha x$ otherwise | $1$ for $x>0$, $\alpha$ for $x<0$ | Preserves a negative-side path; slope is a design choice |
| Exact GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ | Smooth, nonmonotonic modulation; no universal quality advantage |

The sigmoid derivative peaks at $1/4$ and the tanh derivative at $1$ at
zero. Their connection is $\tanh(x)=2\sigma(2x)-1$. Mathematical sigmoid
outputs lie in $(0,1)$ and tanh in $(-1,1)$; finite precision can reach the
endpoints. The sigmoid implementation uses $e=\exp(-|x|)$ and computes
its derivative as $e/(1+e)^2$ to avoid both overflow and cancellation from
an output rounded to one. Tanh's derivative can round to zero in the tails.

ReLU and Leaky ReLU are not differentiable at zero for the selected
$0<\alpha<1$. This implementation selects slopes $0$ and $\alpha$ there,
respectively. Central differences at zero return $1/2$ for ReLU, which is
**not** an ordinary derivative or the selected backward convention. Tests
check conventions separately and avoid kinks when comparing derivatives.
An inactive unit on one batch is not evidence that it will remain inactive
on all subsequent training inputs.

Exact GELU uses the standard-normal CDF supplied by
[SciPy `ndtr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ndtr.html).
It is a deterministic function, not a sampled gate. It preserves attenuated
negative values, has a slightly negative global minimum, and its derivative
can be negative or exceed one. The commonly used tanh approximation is

$$
\frac{x}{2}\left[1+\tanh\left(\sqrt{2/\pi}(x+0.044715x^3)\right)\right].
$$

That approximation is not implemented here. The exact definition comes from
[Hendrycks and Gimpel, Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
The example's CDF calculation is library-backed; only the activation and
derivative assembly are manual. Float64 evaluation is not symbolic exactness.

## Softmax couples a vector

For logits $z\in\mathbb{R}^K$ and $T>0$:

$$
p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad
\frac{\partial p_i}{\partial z_j}=\frac{p_i(\delta_{ij}-p_j)}{T}.
$$

Increasing one logit changes competing probabilities. The Jacobian is symmetric,
has nonpositive off-diagonal entries and annihilates the all-ones vector:
$J\mathbf{1}=0$. This reflects invariance to a common logit offset.
At $T=1$, $J=\operatorname{diag}(p)-pp^\top$. Materializing a dense
$K\times K$ matrix is useful for inspection, not large-vocabulary inference.
For an upstream vector $g$, a practical vector-Jacobian product is
$p\odot(g-\langle p,g\rangle)/T$, without constructing the matrix.

The implementation subtracts the maximum **before** dividing by temperature.
Each exponential is then at most one. SciPy's
[softmax documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html)
also describes shifting for stability. Our default axis is `-1`; SciPy's
default `axis=None` normalizes over the entire array, so reference calls
specify the axis. For `(batch, classes)` inputs, rows should usually normalize
independently. Summing to one on the wrong axis can conceal an application bug.

For unequal logits, decreasing $T$ concentrates mass and increasing $T$
flattens it. Positive temperature preserves mathematical ordering and argmax;
ties remain ties. Very small probabilities can underflow to zero, losing
distinctions in float64. A huge offset can also erase logit differences before
the function is called. Shifting cannot recover information already rounded
away. Inputs with unrepresentable scaled differences raise `ValueError`.
NaN, infinities, empty inputs and invalid axes or temperatures are rejected.
There is no masked-attention contract: even partially masked `-inf` rows
are rejected, and fully masked rows need a separate application policy.

## Cross-entropy belongs in logit space

For a fixed target distribution $y$, $y_i\geq0$ and $\sum_i y_i=1$:

$$
L=-\sum_i y_i\log p_i,\qquad
\frac{\partial L}{\partial z_i}=\frac{p_i-y_i}{T}.
$$

The batch implementation uses the **mean**, giving $(P-Y)/(nT)$, and
supports one-hot or fixed soft targets. There are no class weights, label
derivatives or alternative reductions. With $s=(z-\max z)/T$ it evaluates
$\log p=s-\log\sum_i e^{s_i}$ directly. This retains a finite loss for an
incorrect target whose probability underflows to zero. The reference uses
[SciPy `logsumexp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html).
Taking `log(softmax(z))` or clipping probabilities changes the numerical
behavior, and clipping may change the objective and its gradient.

If a framework loss expects logits, pass logits directly. Applying softmax
first would make that loss treat probabilities as its input logits. The same
principle applies to a binary loss that already includes sigmoid internally.
These examples check explicit analytical gradients against central differences;
they do not execute an autograd framework comparison.

## Applied choices and interview questions

**Hidden or output activation?** Hidden nonlinearities change representations
and gradients. Outputs express the task constraints: sigmoid for binary or
coexisting labels, softmax for one exclusive category, and an identity head
for unbounded regression. Separate sigmoid outputs do not enforce exclusivity
or imply that the labels are statistically independent in the data.

**Does ReLU solve vanishing gradients?** Its positive-side slope avoids local
attenuation, but negative units and weight-Jacobian products still matter.
Inspect representative activation distributions, gradient norms, initialization
and update sizes before attributing a training failure to the activation.
Leaky ReLU preserves a local path, not a guarantee of recovery or convergence.

**Does GELU's smoothness make it better?** No task-level ranking follows from
smoothness. Preserve a pretrained architecture's computation unless a change
has a controlled evaluation plan. This study measures no latency or throughput.

**Does temperature improve calibration?** Arbitrary rescaling changes
concentration, not calibration evidence. Fitting temperature requires labeled
validation data and an evaluation protocol. A high next-token probability is
also not a measure of whether a generated claim is true.

**Why check finite differences?** Central differences approximate one input
direction at a time with truncation and floating-point errors. They diagnose
derivation, axis, reduction and scale mistakes on smooth points. They neither
prove correctness for all inputs nor replace reverse-mode differentiation.

## Suggested follow-ups (not executed)

- Compare ReLU, Leaky ReLU and GELU in the same synthetic MLP with fixed splits,
  matched training budgets and multiple initialization seeds. Document whether
  initialization is matched or activation-specific before comparing validation loss.
- Fit a temperature on a validation set and evaluate calibration separately
  on a held-out set. The fixed-logit sweep below fits no temperature.
- Extend the saturation inputs and evaluate gradient products with explicit
  weights; keep local derivatives distinct from whole-network gradient norms.

The analytical run below included no training, calibration fitting, latency
benchmark or visual phase. A subsequent [visual-lab record](VISUAL_LAB.md)
documents a single-seed MLP comparison and controlled gradient products.
Multi-seed comparisons, fitted calibration and latency benchmarks remain
unexecuted; the analytical record is preserved separately.

## Executed study — 2026-11-18

**Executor:** Codex, local Windows CPU execution. **Author review required**
for all interpretation candidates below. This is a deterministic numerical
exercise on synthetic inputs, with no external dataset or random sampling.

**Hypotheses:** (1) analytical activation derivatives and the coupled softmax
Jacobian agree with central differences away from kinks; (2) common shifts
preserve normalization and temperature changes concentration; (3) logit-space
loss remains finite for a confidently incorrect prediction.

**Configuration:** Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1, float64;
pytest 9.2.0 for the separate tests. Both scripts were run from the repository
root. `from_scratch.py` used logits `[2, 1, -1]`, temperatures `[0.5, 1, 2]`,
and central-difference step $10^{-5}$. `example.py` used preactivations
`[-8, -2, -0.5, 0.5, 2, 8]`, Leaky ReLU slope 0.01, and logit rows
`[[2, 1, -1], [-1, 0, 3]]` with targets `[[1, 0, 0], [0, 0, 1]]`.
Derivative checks used `rtol=1e-6, atol=1e-9`; the softmax Jacobian used
`rtol=1e-7, atol=1e-9`. References used SciPy `expit`, `softmax` and `logsumexp`.

**Observed numerical results:** maximum activation-Jacobian absolute error
was $3.79\times10^{-11}$ across the five elementwise functions. The sigmoid
derivative at $x=8$ was 0.000335238; tanh's was $4.50141\times10^{-7}$.
At $x=-2$, ReLU, Leaky ReLU and exact GELU derivatives were respectively
0, 0.01 and -0.0852318. GELU's derivative at $x=2$ was 1.08523.

| Temperature | Softmax Jacobian max absolute error | Mean cross-entropy | Entropy of first row (nats) | Loss-gradient max absolute error |
|---:|---:|---:|---:|---:|
| 0.5 | 1.23e-11 | 0.0659596 | 0.380067 | 1.02e-11 |
| 1.0 | 2.19e-12 | 0.2074481 | 0.713866 | 3.06e-12 |
| 2.0 | 3.69e-12 | 0.4552432 | 0.952808 | 3.29e-12 |

Adding 10,000 to all logits changed probabilities by 0 at printed float64
precision for these integer-valued rows. For logits `[[1000, -1000]]` and
target `[[0, 1]]`, the logit loss was 2000 despite the target probability
underflowing to zero. Both scripts completed their numerical assertions and
printed JSON only; no outputs were selected as public assets.

**Interpretation candidates:** the measured cases support the derived local
gradients, mean reduction, temperature factors and stable loss calculation.
The temperature trend is concentration on fixed predictions, not a trained
model quality comparison. Both rows already favor their target; the smaller
loss at low temperature cannot establish a calibration benefit.

**Limitations:** a small fixed input set, one finite-difference step in the
examples and float64 CPU arithmetic do not cover all scales or precisions.
Tests additionally vary step size, soft targets, tensor axes and invalid
inputs; they are software evidence, not author approval, generalization
evidence or a replacement for maintained numerical libraries.

**Validation scope:** 68 focused tests passed in one isolated file, with no
skips or warnings. Repository syntax and internal-link checks also passed;
these do not establish rendered GitHub math/layout or remote-link availability.
The local stable environment did not contain `pip`, so `python -m pip check`
could not run. No dependencies or environment settings were changed.
