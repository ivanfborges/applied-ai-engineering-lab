# Optimizers: gradient history, normalization and decay

[Topic overview and commands](README.md)

## From gradients to updates

Let $\theta_{t-1}$ denote parameters before optimizer step $t\geq1$.
For mean empirical loss $L(\theta)=N^{-1}\sum_i\ell_i(\theta)$,
a mini-batch gradient is

$$
g_t=\frac{1}{|B_t|}\sum_{i\in B_t}\nabla\ell_i(\theta_{t-1}).
$$

Uniform independently sampled batches give the usual unbiased estimator at
fixed parameters. Random reshuffling without replacement, as used here,
creates dependent steps; the independent-sampling argument does not apply
unchanged conditional on earlier batches. A full-gradient quadratic has no
sampling noise at all. The update mechanisms work in either setting.

The forward pass defines loss, backpropagation computes gradients, and an
optimizer consumes them. Replacing the optimizer does not repair an incorrect
gradient, target definition, validation split or missing training signal.

## Equations and implemented conventions

All vector products, squares, roots and divisions below are coordinate-wise
unless stated otherwise. The learning rate $\eta>0$, numerical stabilizer
$\epsilon>0$, and history coefficients are in $[0,1)$. States start at zero.
For coupled decay, define $d_t=g_t+\lambda\theta_{t-1}$ with
$\lambda\geq0$. This corresponds to adding
$\lambda\|\theta\|_2^2/2$ to the loss before taking its gradient.

### SGD and Momentum

$$
\text{SGD:}\qquad\theta_t=\theta_{t-1}-\eta d_t.
$$

$$
\text{Momentum:}\qquad b_t=\mu b_{t-1}+d_t,\qquad
\theta_t=\theta_{t-1}-\eta b_t.
$$

This Momentum rule is not normalized by $1-\mu$. With constant $d$, its
buffer approaches $d/(1-\mu)$, so increasing momentum also changes the scale
of persistent updates. Alternating directions can partly cancel; persistent
directions can accelerate or overshoot. The code uses no dampening or Nesterov.
PyTorch initializes its buffer to the first gradient; zero initialization
followed by our recurrence gives the same result when dampening is zero.

For plain SGD only, coupled L2 is algebraically equivalent to the shrinkage
$(1-\eta\lambda)\theta_{t-1}-\eta g_t$. Once a penalty enters momentum or
adaptive states, this simple equivalence generally fails.

### RMSProp

$$
s_t=\alpha s_{t-1}+(1-\alpha)d_t^2,\qquad
\theta_t=\theta_{t-1}-\eta\frac{d_t}{\sqrt{s_t}+\epsilon}.
$$

The implementation is uncentered, without momentum or bias correction.
Epsilon is **outside** the square root, following the checked PyTorch rule.
Its placement matters for small gradients. $s_t$ tracks an uncentered second
moment; it is not a variance without subtracting the squared mean, and is not
a Hessian estimate. Rescaling can help unequal gradient magnitudes but does
not guarantee stable steps or remove all effects of feature scaling.

### Adam

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)d_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)d_t^2.
$$

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

For a stationary gradient distribution with mean $q$, zero initialization
gives $E[m_t]=(1-\beta_1^t)q$; the denominator removes that initialization bias.
The analogous argument applies to a stationary second raw moment. Correction
does not make a nonstationary history an unbiased estimate of its current
moment, or make the ratio an unbiased descent direction.
At the first step $\widehat m_1=d_1$ and $\widehat v_1=d_1^2$.

The accumulator update uses $(1-\beta_1)d_t$, unlike the unnormalized
Momentum convention above. A large $\beta_2$ forgets history slowly. There
is no universal guarantee that Adam minimizes validation error better than SGD.

### AdamW

Use $g_t$ in Adam's two moments, excluding the penalty, and update:

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
$$

Decay bypasses coordinate normalization but still depends on learning rate.
Shrinkage alone over $T$ variable-rate steps is
$\theta_T=\prod_{t=1}^T(1-\eta_t\lambda)\theta_0$.
The common contraction interpretation assumes $0\leq\eta_t\lambda<1$.
The educational API allows larger values to expose the equation, even when
they flip signs or expand magnitudes.

With a fresh state and zero supplied gradient, AdamW only shrinks parameters;
coupled Adam generates nonzero moments when parameters and decay are nonzero.
With existing state, even a zero supplied gradient can produce an adaptive
update from residual momentum. A zero tensor gradient is also different from
PyTorch `grad=None`, which can skip a parameter's update and decay entirely.

The code uses `torch.optim.Adam` with its coupled default and
`torch.optim.AdamW` for decoupled decay, always passing `weight_decay`
explicitly. PyTorch 2.15 also exposes `Adam(decoupled_weight_decay=True)`;
the study does not require that newer keyword across the canonical version
range. AdamW's library default decay is 0.01, whereas Adam's is zero.
Comparing defaults without inspecting them can confound an experiment.
See the [official Adam contract](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)
and [AdamW contract](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html).

## State cost and practical choices

| Method | Extra dense arrays per parameter | Question before adoption |
|---|---:|---|
| SGD | 0 | Are scaling and a learning-rate schedule adequate? |
| Momentum | 1 | Are momentum and rate tuned together? |
| RMSProp | 1 | Is gradient-history normalization useful for this task? |
| Adam / AdamW | 2 | Can the optimizer state fit, and what decay policy is appropriate? |

For one billion FP32 trainable scalars, parameters occupy about 4 decimal GB
and two FP32 moment arrays add about 8 GB. This arithmetic excludes gradients,
activations, temporaries, allocator overhead, master copies and sharding; it
is not a measured peak-memory estimate. The NumPy implementation allocates
only the states used by its chosen method, though input/output copies add
temporary memory.

For transformer fine-tuning, start with a validated recipe and examine decay
groups, scheduling and held-out quality. For CNNs, an established SGD/Momentum
recipe can be a relevant comparator. This study executes neither workload.
Training only adapters reduces optimizer state for trainable parameters;
it does not remove base-model or activation memory. Inference-only RAG has
no neural optimizer step; training an embedding model or reranker does.

Clipping modifies gradients before an optimizer consumes them; it does not
replace an appropriate rate or bound every adaptive update. Resuming training
requires optimizer moments and step counters as well as weights, scheduler,
sampling and relevant random states. This minimal implementation does not
provide resumable checkpoints.

## Training controls and comparison limits

The synthetic data generator alternates two balanced labels, draws uniform
angles, assigns radii 0.8 and 1.5, and adds Gaussian radial noise with standard
deviation 0.18. A seeded permutation creates an 80/20 split without
stratification. There is no external dataset or fitted preprocessing.

Defaults: 1,000 samples; data/split seed 56, batch seed 57, model seed 58;
float32 CPU with one PyTorch thread and deterministic algorithms requested;
tanh MLP 2–16–16–2 (354 parameters); mean cross-entropy on raw logits;
60 epochs and batch size 64, retaining the final partial batch.

| Optimizer | Learning rate | Other update settings |
|---|---:|---|
| SGD | 0.1 | No momentum |
| Momentum | 0.03 | $\mu=0.9$, no dampening or Nesterov |
| RMSProp | 0.003 | $\alpha=0.99$, uncentered, no momentum |
| Adam / AdamW | 0.01 | $(\beta_1,\beta_2)=(0.9,0.999)$ |

Adaptive methods use $\epsilon=10^{-8}$. All five run at zero decay; two
additional Adam/AdamW runs use decay 0.01. In that pair, every parameter,
including biases, receives decay as an explicit educational choice.
No clipping, schedule, validation selection or early stopping is applied.
The split, initial parameter fingerprint and epoch permutations are fixed
across configurations. Tests check repeatability within the tested environment;
cross-version or cross-hardware bitwise equivalence is not promised.

Losses are evaluated without an added penalty so reported task losses share
the same definition. Norms cover all parameters or all unregularized loss
gradients. `last_update_norm` covers the final mini-batch update.
`elapsed_seconds` includes the loop's diagnostics and epoch evaluation but
excludes model/optimizer construction. Sequential order, warm caches and
instrumentation prevent interpreting it as an optimizer-speed benchmark.

Equal steps and batches do not equalize effective rates or tuning effort.
The rates are illustrative, chosen before this execution, and were not
searched. Validation data are reported every epoch but never used to select
a model. There is no untouched test set; these observations cannot support a
claim about an optimizer's expected generalization on other data.

## Numerical contract

`Optimizer` owns a nonempty finite real array, casts it to float64 and returns
copies of parameters and state. Each `step` requires exactly matching shape;
invalid gradients and detected update overflow leave parameters, state and
counter unchanged. Hyperparameters must be finite real scalars in their
documented ranges. Use construction to configure a run; recreate the object
to reset state. It is a dense educational API, not a framework replacement.

The quadratic has Hessian $\operatorname{diag}(10,2)$ and minimum at zero.
For zero-decay full-gradient SGD, coordinate error evolves as
$w_{t,i}=(1-\eta h_i)w_{t-1,i}$. Contraction requires $0<\eta<0.2$.
The tests check both a stable rate and divergence beyond that boundary;
this bound is specific to this quadratic and plain SGD.

## Interview questions

**Why distinguish backpropagation from an optimizer?** Backpropagation computes
the loss derivative at current parameters. The optimizer defines how that
derivative and retained state change parameters; changing one does not correct
errors in the other.

**Why does Adam use bias correction?** Zero-initialized exponential averages
underestimate stationary moments early on. The geometric-sum correction
removes that initialization factor, with an update-based counter.

**Is RMSProp a second-order method?** It uses squared first derivatives to
rescale coordinates. These are not Hessian entries, and an uncentered second
moment is not the gradient variance.

**Why is L2 different from AdamW decay?** Coupled L2 changes both moment
estimates and is adaptively normalized. AdamW applies a separate parameter
shrinkage; the learning rate still affects its total magnitude.

**How would you choose under a limited training budget?** Include established
task-specific recipes, tune each rate with a comparable budget, compare
held-out task quality across seeds, and record steps, compute and memory.
Preserve a final test set; do not pick from training loss alone.

**What if training loss improves while validation quality deteriorates?**
Audit split leakage, distribution mismatch, objective alignment and labels;
then investigate overfitting, training duration, scheduling and regularization.
Changing optimizers alone is not a diagnosis.

## Suggested follow-ups (not executed)

- Tune rates independently across multiple seeds with a fixed search budget
  and final untouched test set.
- Vary batch size while distinguishing equal-epoch from equal-step budgets.
- Study stronger quadratic anisotropy and Momentum overshoot analytically.
- Compare parameter-group decay and scheduling on a representative real task.

These proposals have no measured results in this study.

## Primary references

- Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015: moment estimates and bias correction.
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019: coupled L2 versus independent decay; its empirical results are not reproduced here.
- PyTorch 2.15 [SGD](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html), [RMSProp](https://docs.pytorch.org/docs/2.15/generated/torch.optim.RMSprop.html), [Adam](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html) and [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html): checked implementation conventions.

## Executed study — 2026-11-18

Codex executed both default commands from the repository root on Windows x64
using Python 3.11.14, NumPy 2.4.6 and PyTorch 2.15.0+cpu. The training report
completed at `2026-11-18T16:06:38.880420+00:00` (13:06:38 in São Paulo).
This is a new analytical execution record; it does not replace earlier studies.

```powershell
python 04-deep-learning/56-optimizers/from_scratch.py
python 04-deep-learning/56-optimizers/example.py
```

Locally, `python` in these commands was the explicit stable interpreter
`.local/environments/n09-full-2026-11-13/Scripts/python.exe`.
OMP/OpenBLAS/MKL thread limits were 1, and training requested one PyTorch
thread. Complete JSON stdout was retained in two ignored reports named
`quadratic-20261118T1606262235887Z.json` and
`training-20261118T1606262235887Z.json` under this topic's `outputs/`.
Their absence from a public checkout is intentional; the tables below retain
the selected numerical evidence. These analytical commands generated no images
or other previews. The later visual phase has its own
[execution record](VISUAL_LAB.md#executed-visual-study--2026-11-18); its rotated
and shifted objectives do not replace the analytical configurations below.

### Quadratic mechanism check

**Hypothesis:** the five stated rules reduce this simple objective under
their illustrative settings, and zero decay removes the Adam/AdamW distinction.
**Configuration:** float64, initial $(2,-1)$, initial loss 21, 50 full-gradient
steps, no decay in the five-way comparison; rates and results follow.

| Method | Rate | Final unpenalized objective |
|---|---:|---:|
| SGD | 0.06 | 0.00000280716031 |
| Momentum | 0.02 | 0.0232469644 |
| RMSProp | 0.1 | $1.63646129\times10^{-28}$ |
| Adam | 0.1 | 0.0287241313 |
| AdamW | 0.1 | 0.0287241313 |

In the additional matched Adam/AdamW pair at rate 0.1, the final maximum
coordinate difference was exactly zero at decay 0. At decay 0.1 it was
0.0591778540; final parameter norms were 0.0759170500 (Adam) and
0.0240733482 (AdamW).

**Interpretation candidate — author review required:** the zero-decay control
matches the algebraic equivalence, while nonzero decay changes the trajectory.
**Limitation:** this objective has only two coordinates and condition number 5;
rates are untuned and gradient magnitudes contain no stochastic noise. A tiny
final objective here does not rank neural-network optimizers. On this diagonal
quadratic, coupled decay rescales each coordinate gradient by a constant,
which Adam's normalization can largely cancel; its near-equal norm to the
zero-decay run is not evidence that L2 generally has no effect.

### Matched synthetic classification

**Hypothesis:** fixed data, initialization and batches expose different fitted
parameters across the stated optimizer configurations; coupled and decoupled
decay can differ even at an identical numerical coefficient.
**Configuration:** the defaults in
[training controls](#training-controls-and-comparison-limits), 780 updates per
configuration, one seed/split. Initial validation loss was 0.723748267 and
accuracy 0.505. Initial parameter norm was 3.93426728.

| Method | Decay | Final train loss | Final validation loss | Validation accuracy | Final parameter norm | Instrumented seconds |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 0 | 0.103748 | 0.142305 | 0.940 | 8.004902 | 0.651 |
| Momentum | 0 | 0.085750 | 0.140734 | 0.940 | 9.418445 | 0.712 |
| RMSProp | 0 | 0.087903 | 0.131023 | 0.955 | 9.191699 | 0.829 |
| Adam | 0 | 0.074585 | 0.120398 | 0.955 | 10.249730 | 0.921 |
| AdamW | 0 | 0.074585 | 0.120398 | 0.955 | 10.249730 | 0.954 |
| Adam | 0.01 | 0.133984 | 0.155012 | 0.940 | 6.072466 | 0.986 |
| AdamW | 0.01 | 0.076454 | 0.120630 | 0.950 | 9.883888 | 1.063 |

All seven runs shared the initial-parameter SHA256
`647a4c7f15c17f797ce6202462e9bc8b87e9d5aadc37d6ec4903724a8aaeed80`
and batch-order SHA256
`869533b46dc8720dce194396deb20c1457cb36af39631828289c0e98cf6376c7`.
The zero-decay Adam and AdamW final-parameter hashes were identical, not only
their rounded metrics. Their nonzero-decay final hashes differed.

**Interpretation candidate — author review required:** these configurations
reduced validation loss relative to initialization, and the paired runs expose
the implemented decay distinction. The stronger parameter-norm reduction in
coupled Adam did not produce a lower validation loss than the paired AdamW
configuration on this split. This does not establish a generally beneficial
norm, decay coefficient or optimizer.

**Limitation:** one synthetic split with 200 validation observations, no
replicated training seeds, no tuning, no uncertainty estimates and no final
test set. Timing includes diagnostics and evaluation; it is not a speed
benchmark. The coefficients have different meanings, so matching 0.01 is not
matching effective regularization strength. No real-data, large-model,
deployment or scientific author approval is claimed.

### Verification scope

The focused isolated test file passed **102 tests**, with no skips or warnings
on its final run, using pytest 9.2.0. It covers hand calculations, 25-step
PyTorch float64 agreement at two decay values and two epsilon values, moment
states, zero-decay equivalence, invalid inputs, atomic rejection of overflow,
finite differences and the quadratic SGD stability boundary. Small repeated
CPU training checks verify initialization, budgets and repeatable outputs;
they are verification cases rather than additional quality benchmarks.

The initial test run had one pytest parametrization deprecation warning;
converting its iterator to a list removed it in the final run. The stable
environment has no `pip` module, so `python -m pip check` could not execute.
The equivalent environment check succeeded with
`uv --cache-dir .local/cache/uv-day56 pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`:
78 installed packages checked, all compatible. The explicit cache directory
avoided an access-denied error on uv's default user cache. No packages were
installed or upgraded. These are local checks, not a complete curriculum run,
Linux CI, GitHub math-rendering verification or author interpretation review.
