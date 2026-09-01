# Senior Interview Questions

## 1. How do entropy, cross-entropy, and KL divergence differ?

Entropy `H(P)` is the expected surprise intrinsic to `P`. Cross-entropy
`H(P,Q)` is the expected negative log-probability assigned by `Q` to events
drawn from `P`. KL divergence is the excess:

$$
D_{KL}(P\|Q) = H(P,Q) - H(P).
$$

With a fixed target distribution, only the mismatch term changes as the model
changes.

## 2. Why is cross-entropy used for classification instead of accuracy?

Accuracy depends on a discrete argmax and supplies no useful local gradient for
most parameter changes. It also treats a barely correct and highly confident
correct prediction identically. Cross-entropy is differentiable with respect
to logits, rewards probability assigned to the observed class, and is the NLL
of a categorical output model. It is a training objective, not a replacement
for task-specific evaluation.

## 3. How are cross-entropy and maximum likelihood connected?

For categorical observations, maximizing the conditional likelihood is
equivalent to minimizing the sum of `-log q_theta(y_n|x_n)`. With one-hot
targets, that sum is categorical cross-entropy. The equivalence depends on the
categorical probabilistic model and on how weighting or smoothing changes the
objective.

## 4. Why is KL divergence not a distance?

It is generally asymmetric: `D_KL(P||Q) != D_KL(Q||P)`. It can also be
infinite and does not satisfy the triangle inequality. It should be called a
divergence.

## 5. What happens when `q_i = 0` but `p_i > 0`?

Forward KL and cross-entropy are infinite because `Q` declared an event that
can occur under `P` impossible. An educational implementation should either
preserve this extended-real result or disclose that clipping changes it.
Finite logits passed through mathematical softmax remain strictly positive,
while stable log-softmax avoids materializing problematic probabilities.

## 6. Why should a framework cross-entropy function receive logits?

Logits allow the function to combine log-softmax and NLL through the
log-sum-exp trick. Passing probabilities to an API that expects logits changes
the calculation and can lose numerical stability. The exact contract must be
checked because not every library uses the same input convention.

## 7. Derive the softmax cross-entropy gradient.

For `q = softmax(z)` and target distribution `y`, differentiating
`-sum_i y_i log q_i` gives

$$
\frac{\partial L}{\partial z_j} = q_j - y_j.
$$

For a one-hot target, the true-class logit receives a negative gradient when
its probability is too small, while other logits receive positive gradients
proportional to their predicted probabilities.

## 8. When do you use BCE versus categorical cross-entropy?

BCE models a Bernoulli outcome. A binary task uses one Bernoulli probability;
a multilabel task normally uses one independent Bernoulli term per label.
Categorical cross-entropy models one mutually exclusive outcome across classes
using softmax. Treating a multilabel task as multiclass incorrectly forces the
labels to compete for total probability one.

## 9. Does low cross-entropy guarantee good calibration?

No. Expected log loss is a proper scoring objective, but a finite fitted model
can still be miscalibrated because of misspecification, regularization,
distribution shift, finite data, or optimization behavior. Calibration should
be assessed separately with held-out reliability analysis and complementary
scores.

## 10. How does cross-entropy appear in LLM training?

At each valid position, an autoregressive model predicts a categorical
distribution over vocabulary tokens and incurs `-log p_theta(x_t|x_<t)` for
the observed next token. Correct causal shifting, attention, masking, and
token-level aggregation are part of the objective; otherwise low loss can be
an artifact of leakage or padding.

## 11. When is perplexity comparable across language models?

Only when the evaluation corpus, tokenizer or token accounting, text
normalization, context treatment, stride, boundary policy, and masking are
compatible. Perplexity is exponentiated mean token NLL, not a universal measure
of generation quality, safety, latency, or usefulness.

## 12. Why does KL direction matter in approximation or distillation?

The left-hand distribution defines the expectation and therefore which regions
receive weight. Forward KL strongly penalizes assigning too little probability
where `P` has mass; reverse KL weights discrepancies where `Q` places mass.
The popular mass-covering/mode-seeking shorthand is context-dependent, so the
choice should follow sampling access, density access, and the cost of each
kind of mismatch.

## Interview-ready summary

Entropy is uncertainty within a distribution. Cross-entropy evaluates how a
model distribution predicts data from a target distribution, and KL divergence
is the additional cost caused by their mismatch. The identity
`H(P,Q) = H(P) + D_KL(P||Q)` explains why fixed-target cross-entropy training
minimizes forward KL. With one-hot labels the loss is the categorical NLL, so
it is also maximum-likelihood training. In an LLM the same loss is repeated at
valid next-token positions, where stable logits, causal shifting, masking, and
token-aware aggregation are essential.
