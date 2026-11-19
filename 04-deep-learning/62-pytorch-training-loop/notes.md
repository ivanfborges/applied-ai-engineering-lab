# Training-loop mechanics and evidence

## Data access and the optimization contract

A map-style Dataset defines indexed example access; the example uses
`TensorDataset` because the tensors already fit in memory. DataLoader handles
sampling, collation and batches. Training shuffles without replacement each
epoch, while validation and test use sequential traversal. `drop_last=False`
keeps the final partial batch. `num_workers=0` keeps this small CPU example
simple and avoids multiprocessing setup on Windows.

For streaming or large files, an iterable dataset or lazy indexed reads may be
appropriate. Workers can accelerate loading, but add memory, seeding and data
sharding concerns. This study does not measure input-pipeline throughput.

Training minimizes empirical risk, not a business metric directly:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i),\qquad
g_B=\frac{1}{|B|}\sum_{i\in B}\nabla_\theta\ell_i(\theta).
$$

Uniform sampling makes the batch gradient an unbiased estimator at a fixed
parameter state. With random reshuffling, remaining examples depend on earlier
batches and updates; conditional unbiasedness at every step does not follow
automatically. Weighted sampling and class weighting also change the objective.

For a logit $z$ and binary target $y$, a stable loss is

$$
\ell(z,y)=\max(z,0)-yz+\log(1+e^{-|z|}),\qquad
\frac{\partial\ell}{\partial z}=\sigma(z)-y.
$$

For a linear logit $z_i=x_i^\top w+b$:

$$
\nabla_wL_B=\frac{X_B^\top(\sigma(z_B)-y_B)}{|B|},\qquad
\frac{\partial L_B}{\partial b}=\operatorname{mean}(\sigma(z_B)-y_B).
$$

The NumPy implementation evaluates the loss with `logaddexp` and a
label-dependent logit sign. This avoids overflow and cancellation for extreme
binary logits without clipping the objective. Tests compare gradients against
central finite differences and float64 PyTorch autograd.

## Forward, backward and update are distinct operations

`model(x)` computes logits and records a differentiable graph when enabled.
`BCEWithLogitsLoss` consumes those logits directly. Adding a sigmoid layer
before that loss changes the intended computation. Binary targets here must
be float32 with shape `(N, 1)`, including a singleton final batch; an unrestricted
`squeeze()` can accidentally remove that batch dimension.

`loss.backward()` accumulates derivatives in eligible leaf `.grad` buffers.
It does not change parameter values. `optimizer.step()` consumes gradients;
Adam also updates internal moment estimates. `zero_grad(set_to_none=True)`
clears gradients before each independent batch. A missing gradient and a zero
gradient can produce different optimizer behavior, particularly with decay.
The explicit tests distinguish these responsibilities.

For SGD, $\theta_{t+1}=\theta_t-\eta g_B$. Both weight and bias gradients in
the NumPy example are computed at the same pre-update parameters. Adam in the
PyTorch example uses default betas $(0.9,0.999)$ and epsilon $10^{-8}$, without
weight decay, scheduling or mixed precision.

## Reductions and comparable measurements

For batch mean losses $\bar\ell_b$ and actual counts $n_b$, the epoch aggregate is

$$
\bar\ell=\frac{\sum_b n_b\bar\ell_b}{\sum_b n_b}.
$$

A mean of batch means overweights a short tail batch. The default training
partition has 1,200 examples: 18 batches of 64 and a final batch of 48,
giving 19 optimizer updates per epoch and 1,140 per candidate over 60 epochs.
No accumulation, skipped updates or dropped batches are used.

Training `online_loss` aggregates predictions made at successive parameter
states. `restored_train` evaluates the chosen checkpoint on all training rows
without updates. Compare that frozen measurement to validation loss when
discussing fitting versus generalization; online and validation loss differ
in both population and parameter state.

For gradient accumulation at fixed parameters, microbatch mean losses must be
weighted by $n_b/\sum_b n_b$ before backward. Dividing by the number of
microbatches works only when their sizes are equal. BatchNorm, dropout and
stateful optimizers can still prevent equivalence to a single large-batch step.
The numerical test verifies unequal-size gradient weighting; no accumulation
feature or performance experiment is implemented in the main loop.

## Evaluation, selection and state

`model.eval()` changes layers such as Dropout and BatchNorm. It does not turn
off gradient recording. `torch.inference_mode()` disables recording and further
autograd bookkeeping; tensors created there have restrictions if later used
in differentiable computations. Use `no_grad()` when those tensors need to
re-enter such a computation. The evaluation helper returns Python scalars.

The main candidates have no mode-dependent layers, so their forward values
would be identical in train and eval mode. A separate test includes Dropout
and BatchNorm, verifies inference mode and confirms unchanged parameters,
running buffers and pre-existing gradients. The next training epoch calls
`train()` explicitly. Evaluation never calls backward or an optimizer.

Each epoch is considered for minimum validation BCE; strict improvement means
ties retain the earlier epoch. `deepcopy(state_dict())` preserves a snapshot,
including buffers, instead of live references. The selected epoch is restored
before the model is compared with the other candidate. Architecture ties favor
linear. Test metrics never determine any of these choices.

Only model state is restored: the loop does not resume optimization after
selection. A true resume checkpoint also needs optimizer state, step/epoch,
relevant RNG states and data-order state, plus scheduler/scaler state if used.
No disk checkpoint or serving bundle is provided. The JSON retains training
mean and scale, but is an experiment record rather than a deployable model.

## Executed study — 2026-11-19

Executed by Codex; **author review required** for interpretation candidates.
The two records below are distinct runs. There are no historical Day 62
experiment results being replaced.

### PyTorch: fixed-budget capacity comparison

**Hypothesis fixed before execution:** a ReLU MLP can fit the nonlinear
interaction better than a linear boundary under this fixed budget.

**Configuration:** Windows x64, Python 3.11.14, PyTorch 2.15.0+cpu, CPU float32,
one Torch numerical thread and deterministic algorithms enabled. The data RNG
uses seed 42; features are 1,800 independent two-dimensional standard Gaussian
rows, with labels

$$
y_i=\mathbf{1}\{x_{i1}x_{i2}+0.4x_{i1}-0.2x_{i2}
+0.3\epsilon_i>0\},\qquad \epsilon_i\sim\mathcal{N}(0,1).
$$

Split permutation seed 43 yields 1,200/300/300 training/validation/test cases,
with 659/162/146 positives. Population standard deviations (`unbiased=False`)
and means are fitted on training features only; scales are floored at $10^{-6}$.
Training means are approximately `[-0.003986, 0.028957]`; scales are
`[1.004715, 0.982081]`. Loader seed 44 is reset for each architecture so their
training orders match. Initialization seed 45 is isolated from loader RNGs;
different architectures do not have identical initial functions or weights.

The linear model has 3 parameters; the 2 → 16 → 1 ReLU MLP has 65. Both train
for 60 epochs at batch size 64 and Adam learning rate 0.001, default moments
and no decay. Each receives 1,140 updates. No rate sweep, early stopping or
threshold tuning was performed. The selected checkpoint is the minimum
validation BCE over epochs 1–60; epoch zero is not a candidate.

**Observed results** (BCE rounded to six decimals):

| Candidate | Selected epoch | Frozen train BCE | Validation BCE | Validation accuracy |
|---|---:|---:|---:|---:|
| Linear | 60 | 0.657472 | 0.647401 | 66.00% |
| ReLU MLP | 60 | 0.336355 | 0.314760 | 85.33% |

Both happened to select epoch 60; the snapshot test separately exercises an
earlier optimum followed by degradation. Final online training losses were
0.657909 (linear) and 0.337041 (MLP), distinct from frozen train losses.

Validation selected the MLP. Its **one final test evaluation** produced BCE
**0.412863**, accuracy **77.33%**, TP **116**, FP **38**, TN **116**, FN **30**,
on 300 cases with 146 positives, using `logit >= 0` (probability at least 0.5).
The linear model was not evaluated on test. BCE and accuracy measure different
properties; neither establishes calibration or an appropriate business threshold.

**Interpretation candidate — author review required:** the MLP fitted this
interaction generator better on validation within the specified budget. The
lower test accuracy than validation accuracy also illustrates why a selected
validation score cannot substitute for a separate test estimate.

**Limitations:** a single synthetic split and initialization per architecture,
unequal capacities, a shared illustrative rate and no uncertainty analysis.
Selection across 60 epochs and two models can make validation optimistic;
the test gap alone cannot diagnose its cause. Neither model family is claimed
to have converged or received optimal tuning. No repeated-seed, real-data,
runtime benchmark, accelerator or distribution-shift evidence was collected.

Historical execution from the repository root, using the executor's private
stable interpreter (not a path included in a clone):

```powershell
& ./.local/environments/n09-full-2026-11-13/Scripts/python.exe 04-deep-learning/62-pytorch-training-loop/example.py --output 04-deep-learning/62-pytorch-training-loop/outputs/run-2026-11-19-analytical.json
```

For a new execution, use your activated environment and preserve the dated
record by choosing a fresh filename:

```powershell
python 04-deep-learning/62-pytorch-training-loop/example.py --output 04-deep-learning/62-pytorch-training-loop/outputs/my-new-analytical-run.json
```

The ignored JSON was inspected and contains exact scalar histories and metrics;
it is not a public linked asset. Its UTC execution timestamp is
`2026-11-19T10:41:43.108851+00:00`. No generated preview was selected.

### NumPy: manual optimization on a separate task

**Hypothesis:** manual SGD can reduce training BCE on a noisy linear signal.
Python 3.11.14 and NumPy 2.4.6; `default_rng(42)` generates 400 standard
Gaussian two-feature rows and labels from
$x_1-0.7x_2+0.3\epsilon>0$. The same generator then shuffles each epoch.
Zero initial weights/bias, 30 epochs, batch size 50, learning rate 0.1 and
240 updates; no split or preprocessing is used.

**Observed:** BCE fell from 0.693147 to 0.252313 at frozen final parameters;
final online loss was 0.253803; training accuracy was 92.00%.

**Interpretation candidate — author review required:** analytical gradients
and SGD produced a fitting improvement on these training observations.
**Limitation:** this is resubstitution performance on a different data generator
and optimizer. It supplies no holdout evidence and cannot be ranked against
the MLP's test accuracy.

```powershell
& ./.local/environments/n09-full-2026-11-13/Scripts/python.exe 04-deep-learning/62-pytorch-training-loop/from_scratch.py
```

The stdout record was inspected locally, timestamped
`2026-11-19T10:41:53.221421+00:00`; no NumPy output is published as an asset.

## Interview questions tied to this implementation

**Why can a loop run without learning?** Backward without step leaves weights
unchanged; a detached loss, frozen parameters, saturated activations or an
unsuitable learning rate can also obstruct fitting. Inspect gradients and
parameter deltas before adding capacity.

**Why prefer BCEWithLogitsLoss?** It evaluates sigmoid and binary cross-entropy
in a stable combined formulation. Match output/target shape and dtype; do not
pass probabilities as if they were logits. Multiclass cross-entropy instead
typically uses `(N, C)` logits and integer class indices `(N,)`.

**Does a larger batch make a fair comparison at equal epochs?** Not by itself:
it changes updates per epoch and gradient noise. Equal epochs, updates, examples
and compute budgets are different comparison protocols. No timing claim follows
from the update count.

**When would you choose a custom loop?** For special losses, batches or update
rules that need direct control. For standard tasks, an established trainer can
reduce maintenance. This study deliberately omits distributed execution, mixed
precision, monitoring and fault recovery; those require separate validation.

## Suggested extensions — not executed

- Sweep batch sizes or rates on validation, fixing data and documenting whether
  epochs, optimizer updates or examples processed are held constant.
- Repeat the capacity comparison across prespecified seeds and quantify variability.
- Investigate a 100-row training subset for overfitting under a larger budget.
- Add checkpoint/resume with optimizer and RNG restoration; test continuation
  against an uninterrupted run before claiming reproducible recovery.

No extension result or visual phase is implied by this list. Tests exercise
specific mathematical and orchestration invariants; they do not approve the
author's scientific conclusions.

## Local verification — 2026-11-19

The focused command `python scripts/validate_repo.py tests --scope
04-deep-learning/62-pytorch-training-loop` ran 26 cases in one isolated test
file: all passed, with no skips or warnings. The small integration case also
repeated its three-epoch run and checked identical histories and test metrics
in this environment; it is separate from the 60-epoch experiment above.

Repository checks parsed 426 public Python files and checked 1,520 local link
targets across 272 Markdown files. `git diff --check` passed. The stable
environment has no pip module, so `python -m pip check` could not run;
`uv --cache-dir .local/day62-validation/uv-cache --offline pip check --python
./.local/environments/n09-full-2026-11-13/Scripts/python.exe` instead checked
78 installed packages and reported compatible dependencies. No package was
installed or upgraded. These are local checks, not full-suite, CI, browser,
GitHub math-rendering or scientific-review evidence.
