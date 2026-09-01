# Technical Notes

## Information and logarithm units

For an event with probability `p(x)`, self-information is

$$
I(x) = -\log p(x).
$$

Likely events reveal little; rare events reveal more. Base 2 gives bits and the
natural logarithm gives nats. The code uses natural logarithms because that is
the common convention for machine-learning losses.

For a discrete distribution `P = (p_1, ..., p_K)`, Shannon entropy is

$$
H(P) = -\sum_i p_i \log p_i = \mathbb{E}_{X\sim P}[-\log p(X)].
$$

Entropy is zero for a point mass and reaches `log K` at the uniform
distribution over `K` outcomes. This statement assumes a fixed finite support.

## Cross-entropy and KL divergence

If data follow `P` while predictions use `Q`, cross-entropy is

$$
H(P,Q) = -\sum_i p_i \log q_i.
$$

KL divergence is

$$
D_{KL}(P\|Q) = \sum_i p_i \log\frac{p_i}{q_i}.
$$

Expanding the logarithm gives

$$
\begin{aligned}
D_{KL}(P\|Q)
&= \sum_i p_i\log p_i - \sum_i p_i\log q_i \\
&= -H(P) + H(P,Q),
\end{aligned}
$$

and therefore

$$
H(P,Q) = H(P) + D_{KL}(P\|Q).
$$

Gibbs' inequality implies `D_KL(P||Q) >= 0`, with equality when the
distributions agree almost everywhere. KL is asymmetric and does not satisfy
the requirements of a metric.

### Zero-probability conventions

The discrete formulas use these limiting conventions:

- `0 log 0 = 0` in entropy;
- a term with `p_i = 0` contributes zero to cross-entropy and KL;
- if `p_i > 0` and `q_i = 0`, both `H(P,Q)` and `D_KL(P||Q)` are infinite.

Clipping `q_i` to an epsilon is sometimes convenient in probability-based
software, but it replaces infinity with an epsilon-dependent finite value.
Stable loss functions normally avoid this issue by working from finite logits
with log-sum-exp rather than by clipping already-normalized probabilities.

## Classification, NLL, and maximum likelihood

For a one-hot target whose observed class is `y`, categorical cross-entropy is

$$
-\sum_i y_i\log q_i = -\log q_y.
$$

For independent observations `(x_n, y_n)`, conditional maximum likelihood
maximizes

$$
\prod_n q_\theta(y_n\mid x_n).
$$

Taking a negative logarithm turns the product into the summed categorical NLL.
Thus categorical cross-entropy equals NLL for the specified categorical model.
When the empirical target distribution is fixed, minimizing it also minimizes
`D_KL(P||Q_theta)` because `H(P)` does not depend on `theta`.

This equivalence is conditional on the model and target construction. Class
weights, label smoothing, sampling changes, and auxiliary losses modify the
effective objective or target distribution.

## Softmax, gradients, and stability

For logits `z`, softmax is

$$
q_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Subtracting any constant from all logits leaves the probabilities unchanged.
Implementations subtract `max(z)` and compute log-softmax as

$$
\log q_i = z_i - \operatorname{logsumexp}(z).
$$

This avoids overflow from values such as `exp(1000)`. For target distribution
`y`, differentiating softmax cross-entropy yields

$$
\frac{\partial L}{\partial z_j} = q_j - y_j.
$$

Framework categorical-loss functions generally expect raw logits so they can
fuse these stable operations. Applying softmax before a logits-based loss is a
common interface error.

## Choosing the correct loss structure

- Binary classification models one Bernoulli output and uses binary
  cross-entropy.
- Multiclass classification models one categorical outcome and normally uses
  softmax categorical cross-entropy.
- Multilabel classification models several non-exclusive Bernoulli outcomes
  and normally uses independent sigmoid/BCE terms.
- Soft targets represent distributions rather than class indices. They arise
  in distillation, label smoothing, probabilistic labels, and some forms of
  annotation uncertainty.

Label smoothing may regularize a classifier and discourage extreme logits, but
its effect on calibration is not universally beneficial. The smoothing rule,
model, data, and evaluation procedure matter.

## KL direction

`D_KL(P||Q)` weights discrepancies by samples from `P`; missing regions where
`P` has mass is costly. `D_KL(Q||P)` weights them by `Q`. In common approximate
inference examples, forward KL is described as mass-covering and reverse KL as
mode-seeking. Those phrases are tendencies within particular distribution
families and optimization setups, not universal laws.

The direction should follow the question. Supervised cross-entropy naturally
induces empirical `P` to model `Q` forward KL. Distillation and variational
objectives may choose a direction based on which distribution can be sampled
or evaluated and on which errors should be emphasized.

## Autoregressive language models

An autoregressive sequence model factorizes

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

Its mean token loss over the set `M` of valid target positions is

$$
L = -\frac{1}{|M|}\sum_{t\in M}\log p_\theta(x_t\mid x_{<t}).
$$

Important implementation decisions include:

- shifting inputs and targets so a position cannot see its own answer;
- causal attention that prevents future-token leakage;
- excluding padding, prompt-only, or context-only targets as intended;
- aggregating over valid tokens rather than averaging sequence means when a
  dataset-level token objective is desired;
- defining how document boundaries and context windows are handled.

With natural-log mean NLL, perplexity is `exp(L)`. It is sensitive to the
tokenizer, corpus, normalization, context length, stride, masking, and boundary
policy. A lower value under incompatible evaluation conditions is not evidence
of a better language model or product.

## Trade-offs and limitations

- Accuracy and cross-entropy answer different questions. Identical class
  decisions can have different probability quality.
- Cross-entropy is a proper probabilistic objective, but finite models and
  optimization can still produce miscalibrated predictions. Reliability
  analysis and scores such as Brier loss provide complementary evidence.
- Aggregate loss can hide minority-class or rare-token failure. Report
  suitable slices and task metrics.
- Log loss may not align with ranking quality, asymmetric business costs,
  safety, latency, or generation usefulness.
- Data leakage can make any loss misleading.
- The from-scratch NumPy code operates in standard floating point and omits
  automatic differentiation, devices, mixed precision, distributed reduction,
  and framework-specific weighting semantics.

## Proposed follow-up experiments

These are suggestions and have not been run as part of this topic:

1. Sweep the probability assigned to the true class and plot `-log(q_y)`.
2. Hold `P` fixed while varying `Q`; verify that cross-entropy and forward KL
   differ by the constant `H(P)`.
3. Vary softmax temperature and measure entropy, confidence, and true-class
   NLL.
4. Compare one-hot and smoothed targets through their logit gradients.
5. Compare token-weighted and sequence-weighted losses on deliberately
   unequal sequence lengths.

Any empirical interpretation would need its hypothesis, configuration,
observed result, interpretation candidate, and limitations recorded after
execution.
