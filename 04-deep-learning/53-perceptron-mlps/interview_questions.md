# Interview questions

**Why does a single perceptron fail on XOR?**

Its boundary is affine in the supplied features, and opposite XOR corners
cannot be separated by one such boundary. Expanding the features can change
this limitation; an MLP can learn nonlinear hidden features. See the
[contradiction and constructive example](notes.md#why-xor-needs-a-different-representation).

**What does bias add?**

It shifts a unit's response independently of its input weights. Without it,
the zero-score hyperplane passes through the origin. Bias is a parameter,
not a constant prediction error or the statistical notion of estimator bias.

**Does stacking identity layers reproduce logistic regression exactly?**

The final logit remains affine and the threshold boundary has the same form.
Factorization, optimization and regularization can differ, so the learned
scores need not match. In this study, equal test accuracies coexist with
slightly different log losses.

**How many parameters are in `2 → 8 → 1`?**

There are $2\cdot8+8=24$ in the hidden layer and $8\cdot1+1=9$ in the
output, for 33 total. A trainable bias belongs to each non-input neuron.

**Why is the output activation task dependent?**

A binary logit plus sigmoid supports a Bernoulli objective, softmax supports
mutually exclusive classes, and an unconstrained regression output can remain
linear. Hidden activation choices serve representation and optimization;
they need not match the output. A probability-shaped output still needs calibration testing.

**How do backpropagation and gradient descent differ?**

Backpropagation applies the chain rule to compute gradients. Gradient descent
uses those gradients to update parameters. The handwritten example keeps these
operations separate and checks derivatives by central differences.

**Does universal approximation mean eight neurons can learn any function?**

No. The theorem permits width and parameters to depend on the continuous target,
compact domain and desired error, with suitable activations and biases. It
does not specify a practical architecture or guarantee that finite-data
optimization will find it. See the [primary reference](references.md).

**What does fitting all four XOR corners establish?**

It checks that the selected training procedure found a correct truth-table
fit. There is no independent test set in that demonstration. The separate
noisy-XOR holdout offers a different, still synthetic, evaluation population.
Neither experiment establishes production reliability or seed robustness.

**When would an MLP be a reasonable engineering choice?**

When learned dense representations, feature interactions or joint differentiable
training offer value under the actual data and serving constraints. Compare
against simple models on representative validation data. Check label quality,
preprocessing consistency, calibration, latency and drift instead of treating
the XOR result as a model-selection rule.
