# Backpropagation: local derivatives, shared paths and updates

[Topic overview and commands](README.md)

## Reverse mode and the chain rule

For a scalar objective $L$ and an intermediate vector $u$, write the adjoint
as $\bar u=\nabla_u L$. An operation $v=f(u)$ propagates
$\bar u=J_f(u)^\top\bar v$. This is the column-gradient form of a
vector–Jacobian product: it need not materialize the complete Jacobian.
If several operations consume $u$, add their contributions to $\bar u$.
Processing operations in reverse topological order ensures each operation
receives all downstream contributions before propagating them further.

Reverse mode is well suited to a scalar loss with many parameters. Its
arithmetic cost is typically a small multiple of the forward computation for
ordinary differentiable operations; stored activations can dominate memory.
Forward mode propagates directional derivatives (JVPs) and can be preferable
for few inputs and many outputs. Neither mode is an optimizer. Finite
differences estimate derivatives by repeated evaluations; symbolic
differentiation manipulates expressions.

For $z=wx+b$, $L=\tfrac12(z-y)^2$:

$$
\frac{\partial L}{\partial w}=(z-y)x,\qquad
\frac{\partial L}{\partial b}=z-y.
$$

At $x=3,w=2,b=-1,y=0$, loss is 12.5 and the gradients are 15 and 5.
A simultaneous update with learning rate 0.01 yields $w=1.85,b=-1.05$;
the new loss is 10.125. This is an analytical illustration, not a training run.

## A shared node requires addition

The executable scalar trace evaluates
$z=wx+b$, $a=\tanh(z)$, $\hat y=a+z$ and
$L=\tfrac12(\hat y-y)^2$. With $r=\hat y-y$:

$$
\bar z=\underbrace{r(1-a^2)}_{\text{path through tanh}}
       +\underbrace{r}_{\text{direct path}},\qquad
\bar w=\bar z x,\quad \bar b=\bar z,\quad \bar x=\bar z w.
$$

Overwriting either contribution computes the gradient of a different graph.
This addition across paths is distinct from accumulation across batches or
across separate backward calls. The trace is hand-written for these operations;
it is not an automatic differentiation engine.

## Matrix backward pass

The regressor has one hidden layer and a linear scalar output:

| Quantity | Shape |
|---|---|
| $X$, $Y$ | $(n,d)$, $(n,1)$ |
| $W_1$, $b_1$ | $(d,h)$, $(1,h)$ |
| $Z_1=XW_1+b_1$, $H=\tanh(Z_1)$ | $(n,h)$ |
| $W_2$, $b_2$ | $(h,1)$, $(1,1)$ |
| $\hat Y=HW_2+b_2$ | $(n,1)$ |

For $L=\frac{1}{2n}\sum_i(\hat y_i-y_i)^2$:

$$
\Delta_2=\frac{\hat Y-Y}{n},\qquad
\nabla_{W_2}L=H^\top\Delta_2,\qquad
\nabla_{b_2}L=\sum_i(\Delta_2)_{i,:}.
$$

$$
\Delta_1=(\Delta_2W_2^\top)\odot(1-H^2),\qquad
\nabla_{W_1}L=X^\top\Delta_1,\qquad
\nabla_{b_1}L=\sum_i(\Delta_1)_{i,:}.
$$

Biases broadcast in the forward pass, so their gradients sum over the batch.
The $1/n$ factor occurs exactly once. Because there is exactly one output per
row, `0.5 * mean(residual**2)` implements this objective. Multiple outputs
would require an explicit decision about averaging over output coordinates.
Flat targets are rejected: subtracting an $(n,)$ vector from $(n,1)$
predictions would silently create an $(n,n)$ residual matrix.

`loss_and_gradients` reads parameters and returns fresh gradients. Only
`sgd_step` computes $\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)$, returning
new arrays. Updating $W_2$ before computing $\Delta_1$ would combine
incompatible parameter states. The function name describes the update rule;
the executed study uses the entire batch, with no stochastic sampling.

## Reduction and accumulation

For disjoint microbatches $B_k$ of sizes $n_k$, whose union is the full batch
of size $N$, let $L_k$ be each microbatch's mean loss. At frozen parameters:

$$
L=\sum_k\frac{n_k}{N}L_k,\qquad
\nabla L=\sum_k\frac{n_k}{N}\nabla L_k.
$$

An unweighted average of microbatch mean gradients agrees only for equal
sizes or by coincidence. Updating parameters between microbatches changes
the evaluation points and no longer reproduces a single full-batch step.
Batch-dependent operations such as batch normalization can also invalidate
equivalence. The present model has no such operations.

In PyTorch, backward calls accumulate into leaf gradient buffers, which
normally need clearing before a new optimization step. Shared graph paths
are combined within each backward pass. Tensor detachment can disconnect a
desired path; freezing a parameter need not prevent propagation through its
operation to earlier trainable tensors. These are framework concepts, not
executed PyTorch demonstrations in this study.

## Verification and practical limits

Central differences approximate each coordinate by

$$
g_j^{\mathrm{FD}}=
\frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon}.
$$

The script checks every weight and bias, using 50 forward evaluations for 25
parameters. Tests additionally use a non-symmetric two-feature problem and
nonzero biases: symmetric inputs can conceal bias-gradient errors. They use
float64, perturbations $10^{-4},10^{-5},10^{-6}$, and combined tolerances
`rtol=2e-5, atol=1e-8`. Tiny gradients need an absolute tolerance; relative
error alone is unstable near zero. Large perturbations introduce truncation
error; tiny perturbations introduce cancellation. Agreement is evidence for
the tested computation and points, not a proof for every input or a check of
whether the chosen objective represents the intended task.

The numerical code requires finite real inputs, compatible shapes and positive
finite step sizes. Overflow or invalid arithmetic raises rather than silently
continuing a corrupted update. Float64 arithmetic still has rounding error.

Deep chains propagate products of Jacobians. Contraction or amplification in
relevant directions explains vanishing or exploding gradients; individual
activation derivatives alone do not determine the outcome. Saturating tanh
can weaken signals; ReLU has a derivative convention at zero. Ordinary
backpropagation does not differentiate arbitrary hard selections such as
`argmax` or external API calls.

Residual connections, initialization and normalization affect gradient flow.
Clipping bounds a gradient norm but does not restore missing gradients or
ensure convergence. Activation checkpointing trades recomputation for memory.
In adapter fine-tuning, fewer trainable parameters reduce parameter-gradient
and optimizer-state requirements, while needed activation paths still cost
memory. These considerations motivate diagnosing neural components of
embedding, ranking or language systems; they do not make hard retrieval or
agent orchestration automatically differentiable.

## Executed study — 2026-11-18

Executed locally by Codex on Windows x64 with Python 3.11.14, NumPy 2.4.6
and pytest 9.2.0. This is a new Day 54 record, separate from Day 53 runs.
Commands are in the [README](README.md#inspect-and-run); the actual local
interpreter was `.local/environments/n09-full-2026-11-13/Scripts/python.exe`.
No packages were installed. No external data, GPU or framework was used.

**Hypothesis A:** the explicit chain-rule backward pass agrees with central
differences for all parameters of this smooth small network.

Configuration: nine evenly spaced inputs on $[-1,1]$;
$y=\sin(3x)+0.2x$; `1 → 8 → 1` tanh/linear regressor; float64;
`default_rng(7)`; weights independently drawn from $N(0,0.3^2)$; zero biases;
half-MSE; $\varepsilon=10^{-5}$. Parameter count is $8+8+8+1=25$.

| Parameter block | Maximum absolute analytical/finite-difference disagreement |
|---|---:|
| $W_1$ | 5.117851e-12 |
| $b_1$ | 2.775564e-12 |
| $W_2$ | 2.201406e-12 |
| $b_2$ | 2.775544e-12 |

The scalar trace at $x=3,w=0.2,b=-0.1,y=0.4$ produced contributions
0.442075764 and 0.562117157 to $\bar z$, summing to 1.004192921;
$\bar w=3.012578764$.

**Interpretation candidate — author review required:** these observations
support the implemented derivatives at this configuration. **Limitation:**
the symmetric batch has near-zero bias gradients; the separate asymmetric
automated tests provide additional coverage. No framework parity was tested.

**Hypothesis B:** full-batch gradient descent reduces the fitting objective
with this fixed configuration and supplies measurable hidden-layer gradients.

Configuration: 128 evenly spaced inputs on $[-1,1]$, same synthetic function,
architecture, initialization and precision; 2,000 updates at learning rate
0.1. No regularization, noise, data split or tuning search. Each checkpoint
evaluates loss and gradients after the indicated number of completed updates,
before any next update. Norms are Euclidean/Frobenius norms of each block.

| Completed updates | Half-MSE | $\|\nabla W_1\|$ | $\|\nabla W_2\|$ |
|---:|---:|---:|---:|
| 0 | 0.294870961 | 0.168568055 | 0.221379964 |
| 1 | 0.287154249 | 0.168183796 | 0.219496560 |
| 100 | 0.070613807 | 0.017774866 | 0.006648899 |
| 1,000 | 0.023833284 | 0.018602127 | 0.014931029 |
| 2,000 | 0.004222148 | 0.001550998 | 0.003454435 |

Total parameter displacement norm was 3.595248704. Bias-gradient norms at
these checkpoints were between approximately $7\times10^{-18}$ and
$8\times10^{-17}$. Symmetric inputs, an odd target and tanh, and zero initial
biases explain cancellation here; tiny bias gradients alone do not establish
vanishing gradients across a deep network.

**Interpretation candidate — author review required:** the fixed update rule
improved this training objective while the hidden weights received a nonzero
signal. **Limitation:** one seed, shallow model and no holdout; checkpoint
values do not establish monotonicity at unreported steps, real-data accuracy,
optimizer superiority or a memory/performance benchmark.

**Hypothesis C:** sample-weighted accumulation at frozen parameters reproduces
the full-batch gradient for unequal microbatches.

Configuration: the same 128-point batch at initialization, split into its
first 37 and remaining 91 rows. Weight each mean gradient by $37/128$ or
$91/128$. Maximum absolute coordinate difference from the full-batch
gradient was **3.854339e-17**.

**Interpretation candidate — author review required:** the result agrees with
linearity of differentiation for the specified mean loss. **Limitation:**
this is a numerical identity check for a model without batch-dependent
operations; it does not measure distributed training or memory savings.

Both commands printed JSON only; no generated output or visual preview was
selected for versioning. Automated checks and author interpretation review
remain separate evidence.

## Suggested extensions — not executed as studies

- Repeat training with rates 0.001, 0.01 and 1.0, keeping other settings fixed.
  Record failures and complete trajectories before comparing stability.
- Add depth and compare per-layer norms across seeds, with stated initialization.
- Change tanh to sigmoid and inspect saturation alongside derivative scales.
- Compare every gradient with framework autodiff at identical parameters,
  dtype and loss reduction, before comparing training trajectories.
- Add a separate holdout and justify the target population before making any
  generalization claim.

The finite-difference tests already cover three perturbations. The large-step
unit test is a controlled counterexample to guaranteed descent, not a
learning-rate sweep or optimizer benchmark.

## Interview checks

**Why does reverse mode help for a scalar loss?** It reuses intermediate
adjoints to obtain all parameter derivatives in one backward traversal;
coordinate finite differences need two loss evaluations per parameter.

**Can correct gradients increase the loss?** Yes. A gradient is a local
sensitivity, and a finite step can overshoot. Check gradient correctness and
step size as separate questions.

**Why sum bias gradients but average the loss?** The forward bias is shared
across rows, so its local backward operation sums row contributions. Those
contributions already include the objective's $1/n$ factor.

**What is the first check for an unexpected zero gradient?** Verify graph
connectivity, parameter state, input/target symmetry, activation saturation,
loss reduction and numeric scale. A single norm does not identify the cause.

**Does backpropagation optimize a whole RAG system?** Only supported
differentiable paths. Hard selection, external services and discrete workflow
decisions need other methods or explicitly justified gradient estimators.

## References

- [Stanford CS231n: backpropagation and gradient flow](https://cs231n.github.io/optimization-2/)
  explains local derivatives, branch accumulation and vectorized operations.
- [Stanford CS231n: gradient checks and parameter updates](https://cs231n.github.io/neural-networks-3/)
  discusses central differences, floating-point scale and optimization checks.
- [PyTorch: autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)
  documents the framework graph and gradient-buffer semantics discussed above;
  PyTorch is a reading reference, not a dependency or executed comparison here.

References consulted on 2026-11-18. See also
[Day 5: calculus](../../00-foundations/05-calculus-for-ml/) and
[Day 53: perceptron and MLPs](../53-perceptron-mlps/).

## Separate visual phase

The [visual lab guide](VISUAL_LAB.md) records the subsequently executed
eight-module visual phase. It uses a distinct biased scalar graph and a
two-parameter, 41-input model. The analytical run above and its original
output policy are preserved as historical evidence; visual exports and
interpretation candidates have their own dated record and preview selection.
