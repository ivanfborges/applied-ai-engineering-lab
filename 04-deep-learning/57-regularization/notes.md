# Mechanisms, selection and evidence

## What regularization can establish

Empirical risk averages a prediction loss over training observations. Deployment
risk is an expectation over a target distribution that may differ. A validation
split approximates the latter only to the extent that its sampling scheme and
labels represent the intended use. Regularization cannot repair duplicated
documents across splits, unavailable-at-inference features or an invalid label.

An explicit penalty gives one familiar objective:

$$
J(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i)+\lambda\Omega(\theta).
$$

Dropout and stopping rules need not have this additive form. Normalization also
changes optimization geometry; its stochastic behavior can influence
generalization without making it a guaranteed regularizer.

## Dropout: conditional moments, not exact ensemble prediction

For a fixed activation $h$, dropout probability $p<1$ and keep probability
$q=1-p$, sample $m\sim\operatorname{Bernoulli}(q)$:

$$
\tilde h=\frac{mh}{q},\qquad
\mathbb E[\tilde h\mid h]=h,\qquad
\operatorname{Var}(\tilde h\mid h)=\frac{p}{1-p}h^2.
$$

The inference path uses $h$ directly. Nonlinearity means
$\mathbb E[f(\tilde h)]$ generally differs from $f(\mathbb E[\tilde h])$.
The ensemble analogy is useful but does not imply independently trained networks
or exact averaging. Increasing $p$ increases noise and may prevent useful fitting.
This example places dropout after each normalized ReLU activation; moving it
before BatchNorm would change the statistics BatchNorm learns.

## Weight decay: specify the optimizer and parameter set

For $J(w)=L(w)+\lambda\lVert w\rVert_2^2/2$, vanilla SGD gives

$$
w_{t+1}=w_t-\eta(g_t+\lambda w_t)
=(1-\eta\lambda)w_t-\eta g_t,
\qquad g_t=\nabla L(w_t).
$$

This is an update identity under the stated coefficient convention, without
momentum or adaptive preconditioning. Adding L2 to Adam's gradient affects both
moment estimates. AdamW instead uses data-loss moments and a separate shrinkage:

$$
w_{t+1}=(1-\eta_t\lambda)w_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Even with zero data gradient, decay can move a parameter. Over many updates,
shrinkage depends on the product of $(1-\eta_t\lambda)$ factors. Unequal update
budgets therefore change cumulative decay. Here the decay group contains the
three linear weight matrices. Biases and BatchNorm's scale/shift receive zero
decay. This is an explicit experimental choice, not a universal optimal policy.
See [Day 56](../56-optimizers/notes.md) for optimizer-state controls.

## BatchNorm: forward variance and stored variance differ

For one feature over a batch of size $m$:

$$
\mu_B=\frac{1}{m}\sum_i x_i,\qquad
v_B=\frac{1}{m}\sum_i(x_i-\mu_B)^2,\qquad
y_i=\gamma\frac{x_i-\mu_B}{\sqrt{v_B+\epsilon}}+\beta.
$$

Before the affine transform, the batch variance is $v_B/(v_B+\epsilon)$;
it is not exactly one. Constant features remain finite because $\epsilon>0$.
The NumPy function implements only this 2D training forward operation.

With tracked statistics, PyTorch uses $v_B$ for the training forward pass but
the unbiased estimate $m v_B/(m-1)$ when updating running variance. With momentum
$\alpha$, the update is $r_{new}=(1-\alpha)r_{old}+\alpha r_B$. This momentum has
a different role from optimizer momentum. Ordinary evaluation uses running
statistics, not the current batch. With `track_running_stats=False`, evaluation
also depends on the current batch; that mode is outside this example.

Small batches make statistics noisy. A singleton batch is invalid for this
training setup, so the example merges a final singleton with its predecessor
and retains every training observation. BatchNorm also adds affine parameters
and changes parameterization. It is not an isolated "noise switch". LayerNorm
works across specified features within an example and avoids this batch
dependency, which matters in sequence models and variable batch sizes.

## Early stopping: separate three decisions

1. **Checkpoint selection:** retain the smallest observed validation loss.
2. **Termination:** stop after a patience budget without meaningful progress.
3. **Configuration selection:** compare the retained candidates using validation.

For observed epochs $1,\dots,T$, the best checkpoint is

$$
t^*=\arg\min_{1\leq t\leq T} L_{val}(\theta_t).
$$

The implementation copies `model.state_dict()` on every strict new minimum,
keeping the first checkpoint for ties. Patience resets only when the loss falls
more than `min_delta` below the last significant-progress anchor. Smaller
improvements can accumulate until they exceed that threshold; they can also
update the best checkpoint without resetting patience. Thus `min_delta` does
not redefine $t^*$.

A deep copy prevents subsequent updates from mutating the saved checkpoint.
Restoring all state includes running means, running variances and BatchNorm's
batch counter. `eval()` separately disables dropout and selects the inference
normalization path; `no_grad()` disables gradient tracking but does not set that
mode. The snapshot here is for inference selection. Resuming training would
also require optimizer, scheduler, RNG and preprocessing state.

Repeated validation queries induce selection bias. Neither a zero gradient on
validation data nor a separate final checkpoint eliminates it. The held-out test
is evaluated once after configuration selection per invocation. Further tuning
must ignore that test result or obtain a new untouched evaluation set.

## Reading the experiment

The experiment asks whether these mechanisms change fitting and retained
held-out loss on noisy synthetic moons with irrelevant inputs. It does not
assume every mechanism helps. All candidates share the same train-only scaler,
linear initialization and minibatch order. Separate RNG streams govern data,
noise features, batches, model initialization and dropout.

The non-early-stopping candidates retain epoch 120; stopping candidates can
retain earlier states and consume fewer steps. This choice demonstrates the
selection mechanism, but confounds it with training duration. Compare selected
losses together with epochs and steps. Training losses are deterministic
evaluation-mode cross-entropies on training data, not the noisy batch losses
used for differentiation. All losses exclude decay penalties.

## Interview discussion

**Why can dropout raise training loss and still help?** Perturbed activations
make fitting harder and can discourage brittle dependencies. Improvement on
representative held-out data must still be measured; extra noise can underfit.

**Is BatchNorm an overfitting remedy?** It changes normalization and optimization,
with potential regularizing effects from batch variability. Batch size,
distribution shift and inference statistics affect the outcome. It cannot
guarantee improved generalization.

**How would this apply to a document reranker?** Split by realistic queries,
document families and time boundaries, establish a suitable baseline, and
evaluate supported regularizers against ranking quality and cost. Test OCR
noise and domain changes separately. The synthetic classifier here supplies no
measured evidence about those tasks.

**How would you make the comparison stronger?** Predeclare budgets and selection
rules, compare both last and best checkpoints consistently, repeat across
seeds/splits and quantify variability. Record updates as well as epochs, and
keep the final evaluation outside all tuning decisions.

## Proposed experiments — not executed

- Sweep dropout over `0, 0.1, 0.3, 0.5`, keeping other settings fixed; inspect
  whether fitting degrades before assuming more noise helps.
- Sweep decay over `0, 0.001, 0.01, 0.1` with a declared learning-rate policy;
  record weight norms and updates as well as validation loss.
- Compare BatchNorm at batch sizes `8, 32, 64, 128`; account for changed update
  counts per epoch and retain the singleton policy.
- Repeat at least five seeds and use a common checkpoint policy/budget to
  distinguish stochastic variability from configuration effects.

## Executed study — 2026-11-18

Executed by Codex on Windows 10 x64, using stable Python 3.11.14, NumPy 2.4.6,
scikit-learn 1.9.1 and PyTorch 2.15.0+cpu. The comparison completed at
`2026-11-18T19:18:45.178042+00:00` (16:18 in America/Sao_Paulo).
This is a new analytical record, separate from earlier topics' experiments.

**Hypothesis:** activation noise, shrinkage, normalization and checkpoint
selection can change fitting and retained held-out loss; no universal benefit
is assumed.

**Configuration:** the default command below used 1,000 synthetic observations,
600/200/200 stratified partitions with 300/100/100 examples per class, and a
train-only StandardScaler. The architecture was `20-64-64-2`, with two ReLU
hidden layers, AdamW `lr=0.001`, betas `(0.9, 0.999)`, epsilon `1e-8`, no scheduler
or clipping, CPU float32 and one PyTorch thread with deterministic algorithms.
Seeds were 57 for moons/splits, 58 for noise features, 59 for batch order,
60 for linear initialization and 61 for dropout. Maximum duration was 120
epochs, batch size 64, ten updates per epoch. Dropout was 0.3 and decay 0.01
where enabled; biases and normalization affine parameters were exempt from
decay. BatchNorm used epsilon `1e-5`, momentum `0.1` and tracked statistics.
Early stopping used patience 15 and minimum progress `1e-4`.

All candidates had the same initial linear-parameter SHA-256:
`cec1e86258ce47c3198a5b78f831286ed4a2e7f339f4809b11838fdc892cd8df`.
Batch-order SHA-256:
`0f2dca710c0261dcd516555385dd151aeeb54b9262e472f7a24c3b2e9b1b283e`.
Models without BatchNorm had 5,634 trainable parameters; those with it had 5,890.

**Historical commands actually executed**, from the root, with the executor's
private stable interpreter. That environment directory is not included in a
clone; the portable `python` commands in the [README](README.md) use your own
activated environment and print JSON without overwriting these records:

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
& .local/environments/n09-full-2026-11-13/Scripts/python.exe 04-deep-learning/57-regularization/from_scratch.py | Set-Content -Encoding utf8 outputs/day57-mechanisms-2026-11-18.json
& .local/environments/n09-full-2026-11-13/Scripts/python.exe 04-deep-learning/57-regularization/example.py | Set-Content -Encoding utf8 outputs/day57-comparison-2026-11-18.json
```

The ignored JSON files retain the numeric reports, including full histories,
split fingerprints and elapsed times. They are not public linked assets or a
substitute for this intentional, compact record. Re-running the displayed save
commands overwrites those local filenames; use a new dated name for a new study.

**Results:** all losses below are mean cross-entropy in evaluation mode,
excluding decay penalties. Accuracy is a fraction. "Selected" denotes the
retained checkpoint under each configuration's declared policy.

| Configuration | Epochs run | Updates | Selected epoch | Selected train loss | Selected validation loss | Validation accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 120 | 1200 | 120 | 0.002298 | 0.782570 | 0.850 |
| Dropout only | 120 | 1200 | 120 | 0.065064 | 0.390543 | 0.855 |
| Weight decay only | 120 | 1200 | 120 | 0.002396 | 0.771881 | 0.850 |
| BatchNorm only | 120 | 1200 | 120 | 0.002460 | 0.673410 | 0.800 |
| Early stopping only | 30 | 300 | 15 | 0.268684 | 0.267391 | 0.875 |
| Combined | 44 | 440 | 29 | 0.237357 | 0.276015 | 0.875 |

The baseline and early-stopping-only histories were identical through epoch 30;
both reached validation loss `0.267391` at epoch 15. The early-stopping candidate
restored that checkpoint instead of retaining the baseline's epoch 120. The
combined candidate restored epoch 29 after stopping at 44.

Selection by retained-checkpoint validation loss chose **early stopping only**.
Its single final test evaluation gave loss **0.398053**, accuracy **0.830** on
200 observations. Other candidates received no test evaluation. The dropout
candidate's best *observed* validation loss was `0.260806` at epoch 30, but its
declared last-epoch policy retained epoch 120. Consequently, this selection
does not establish that early stopping outperforms dropout under a common
best-checkpoint policy.

The NumPy run used seed 57 and 100,000 unit activations at dropout 0.3. Empirical
mean was `0.997371`, variance `0.430067`, versus conditional targets 1 and
`0.428571`. BatchNorm feature means had maximum absolute magnitude
`5.97e-16`; variances ranged from `0.999998817` to `0.999999036`. With weights
`[2, -1]`, gradient `[0.5, -0.2]`, learning rate 0.1 and decay 0.2, the two
vanilla SGD formulations both produced `[1.91, -0.96]`, with maximum absolute
difference 0 in this run.

**Interpretation candidates — author review required:** the retained baseline's
very low training loss coexisted with larger validation cross-entropy, and
restoring an earlier state changed that outcome here. Dropout reduced final
validation loss relative to the final baseline in this configuration. Combining
all mechanisms did not yield the lowest retained validation loss. The mechanism
controls agree with their mathematical expectations within sampling/numerical
error. None of these statements is an author-approved conclusion.

**Limitations:** one synthetic seed/split; untuned learning rate and strengths;
unequal update counts and checkpoint policies; BatchNorm adds parameters.
No confidence intervals, repeated-seed evidence, real-data transfer, calibration
study or causal attribution to an isolated regularizer is established.
Wall times include diagnostics and copies and are not benchmarks. Test results
were inspected only after selection and were not used to revise configurations.

**Validation:** the scoped runner passed 38 tests with pytest 9.2.0, without
skips or warnings. The smaller repeated four-epoch/100-observation runs inside
the integration test verify determinism and selection boundaries; they are not
additional comparative findings. `python -m pip check` could not execute because
this environment has no pip module. The installed `uv --no-cache pip check
--python .local/environments/n09-full-2026-11-13/Scripts/python.exe` checked 78
packages and found all compatible; no packages were installed or changed.
Public syntax and local-link checks are reported separately at delivery.
No browser/GitHub math rendering, full curriculum suite or CI run is claimed.

## References

Official documentation was checked against the installed PyTorch 2.15 API.

- [PyTorch Dropout](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html)
- [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html)
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- [PyTorch checkpoint copying and evaluation](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Srivastava et al., Dropout (2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Ioffe and Szegedy, Batch Normalization (2015)](https://proceedings.mlr.press/v37/ioffe15.html)
- [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [scikit-learn make_moons synthetic generator](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
