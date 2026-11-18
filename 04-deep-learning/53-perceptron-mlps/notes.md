# From a threshold to learned representations

## A neuron, a boundary and a feature

For an input $x\in\mathbb{R}^d$, a neuron computes

$$
z=w^Tx+b,\qquad a=\phi(z).
$$

The weight vector sets the direction of the response; the bias shifts its
offset. Omitting the bias forces the zero-score hyperplane through the origin.
A hidden neuron's activation becomes an input feature for the next layer.
Those features are learned jointly, so interpreting one weight in isolation
usually misses interactions and alternative parameterizations.

The classical perceptron uses a hard threshold. Here predictions are $+1$
when $z>0$ and $-1$ otherwise, while training updates **every zero-margin or
misclassified example** with $y\in\{-1,+1\}$:

$$
y(w^Tx+b)\leq0
\quad\Longrightarrow\quad
w\leftarrow w+\eta yx,\qquad b\leftarrow b+\eta y.
$$

An epoch with no updates certifies strict positive training margins under
this convention. For bounded, strictly linearly separable data, the usual
perceptron argument bounds the number of updates. Incorporate the bias by
augmenting $x$ with a constant feature. The epoch cap in code prevents an
infinite loop on nonseparable data; reaching it alone proves no impossibility.
This rule is different from optimizing sigmoid cross-entropy.

## Why XOR needs a different representation

The synthetic XOR labels at $(0,0),(0,1),(1,0),(1,1)$ are $0,1,1,0$.
Suppose an affine score gives strictly positive scores to both positive
corners and nonpositive scores to the negative corners. The positive pair
requires $w_2+b>0$ and $w_1+b>0$, so $w_1+w_2+2b>0$. The negative pair
requires $b\leq0$ and $w_1+w_2+b\leq0$, so $w_1+w_2+2b\leq0$.
The contradiction establishes nonseparability in these original features.

Feature engineering can change that. Adding $x_1x_2$, for example, makes
XOR accessible to an affine classifier in the expanded feature space. A hidden
layer offers a way to learn nonlinear features instead of fixing them manually.

Let $s=x_1+x_2$ and construct two hidden features:

$$
h_1=\operatorname{ReLU}(s),\quad
h_2=\operatorname{ReLU}(s-1),\quad f=h_1-2h_2.
$$

At binary inputs, $f$ is exactly $0,1,1,0$. Classify with
$\mathbf{1}[f>0.5]$. This is a fixed `2 → 2 → 1` network with ReLU hidden
units and a linear output. It demonstrates existence without optimization.
Outside the binary domain, $f$ is a piecewise linear score, not a Bernoulli
probability. The public helper deliberately accepts only binary inputs.

## Layers, shapes and parameter counts

Code uses **row batches**, unlike the column-vector convention in many texts.
For $X\in\mathbb{R}^{N\times d}$ and a hidden width $h$:

$$
H=\tanh(XW_1+b_1),\quad z=HW_2+b_2,\quad p=\sigma(z).
$$

| Quantity | Shape | Meaning |
|---|---|---|
| $W_1$ | $d\times h$ | Input-to-hidden weights |
| $b_1$ | $h$ | Broadcast across rows |
| $H$ | $N\times h$ | Learned hidden representation |
| $W_2$ | $h\times1$ | Hidden-to-logit weights |
| $b_2$ | $1$ | Output offset |
| $z,p,y$ | $N$ in code | One logit, score and label per example |

A dense layer has $(d_{\mathrm{in}}+1)d_{\mathrm{out}}$ trainable parameters
with bias. Thus `2 → 8 → 1` has $3\cdot8+9\cdot1=33$, while the
library `2 → 8 → 8 → 1` network has $24+72+9=105$. The input features
are not a trainable dense layer.

With identity activations, two affine layers collapse:

$$
(XA+a)B+b=X(AB)+(aB+b).
$$

More layers can change parameterization, regularization and optimization,
but the logit remains affine in $X$. A sigmoid output makes the score nonlinear
while its 0.5 decision boundary remains affine. Consequently an identity MLP
and logistic regression need not have identical fitted coefficients or scores;
they have the same kind of boundary, with different objectives/parameterizations.

## Activations and output semantics

| Activation | Role and trade-off |
|---|---|
| Step | Hard class decisions; ordinary backpropagation cannot use it to learn useful gradients |
| Sigmoid | Binary output score in $(0,1)$ mathematically; hidden saturation can shrink gradients |
| Tanh | Signed, bounded hidden features; derivative $1-h^2$ is simple, but saturation remains |
| ReLU | Piecewise linear hidden features; negative pre-activations have zero local gradient |
| GELU | Smooth gating used in many modern networks; unnecessary for this small demonstration |

The comparison uses ReLU; the handwritten gradient example uses tanh to keep
the derivative transparent. Binary classification uses one logit and sigmoid;
mutually exclusive multiclass classification typically uses multiple logits
and softmax; unconstrained regression can use a linear output. Those other
tasks are explanations, not implemented experiments here. A sigmoid score is
not evidence of empirical calibration.

## Minimal optimization machinery

For binary labels $y_i\in\{0,1\}$, mean BCE can be evaluated from logits:

$$
\mathcal{L}=\frac1N\sum_i
\log\left(1+\exp((1-2y_i)z_i)\right).
$$

`np.logaddexp(0, signed_logit)` evaluates this stably. The sigmoid uses
$\exp[-\operatorname{logaddexp}(0,-z)]$ to avoid exponential overflow.
Floating-point saturation to 0 or 1 is still possible. No probability clipping
is inserted into the objective, so analytical gradients match that objective.

Writing the output derivative as a column:

$$
D_2=\frac{p-y}{N},\qquad
\nabla_{W_2}\mathcal{L}=H^TD_2,\qquad
\nabla_{b_2}\mathcal{L}=\sum_i D_{2,i},
$$

$$
D_1=(D_2W_2^T)\odot(1-H^2),\qquad
\nabla_{W_1}\mathcal{L}=X^TD_1,\qquad
\nabla_{b_1}\mathcal{L}=\sum_iD_{1,i}.
$$

Backpropagation computes these sensitivities; full-batch gradient descent
updates each parameter with $\theta\leftarrow\theta-\eta\nabla_\theta\mathcal{L}$.
Gradients are all computed using the same current parameters before any update.
Random independent fan-in-scaled weights break hidden-unit symmetry; biases
start at zero. This is a small tanh teaching example, with no minibatching,
regularization, early stopping, optimizer state or automatic differentiation.
Day 54 develops backpropagation in more depth.

Tests compare every parameter derivative against central differences with
$\epsilon=10^{-6}$ on a small nonsaturated batch. They also verify that
duplicating a batch leaves its **mean** loss and gradients unchanged and check
finite BCE/gradients for logits of $\pm1000$. These cases do not establish
stability for arbitrary floating-point magnitudes or all network architectures.

## Universal approximation: an existence statement

For a compact domain $K\subset\mathbb{R}^d$, a continuous target $f$, a
continuous nonpolynomial activation such as tanh or ReLU, and any
$\epsilon>0$, one-hidden-layer networks with biases and linear output can
approximate $f$ uniformly: there exists a finite width and parameters such
that $\sup_{x\in K}|f(x)-g(x)|<\epsilon$.
See the [Leshno et al. primary paper](references.md) for the broader conditions.

The intuition is that many shifted, rescaled nonlinear features can combine
into increasingly detailed shapes. A chosen width of eight is not universal
for all targets and tolerances. The theorem supplies neither a practical
width nor a training algorithm, data requirement, extrapolation guarantee or
deployment budget. Approximation of continuous functions is distinct from
uniform approximation of a discontinuous class indicator. XOR on four points
illustrates nonlinear representation; it is not an empirical proof of the theorem.

## Executed study — 2026-11-18

Executed locally by Codex on Windows x64, using Python 3.11.14, NumPy 2.4.6,
SciPy 1.17.1 and scikit-learn 1.9.1. Numerical thread counts for the example
run were set to one for OpenBLAS, OpenMP and MKL. No runtime benchmark was
performed. Values below are rounded from script output, not copied from the
private source's proposed experiments. This is the initial analytical record.

### A. Nonlinearity control on noisy synthetic XOR

**Hypothesis:** nonlinear hidden features can learn this XOR generator,
whereas additional identity layers retain an affine logit.

**Configuration:** `example.py`, seed 42; 800 independent random binary
corner draws; labels are XOR of the latent corners **before** independent
Gaussian feature noise of SD 0.12. A stratified 25% holdout uses split seed 42:
600 training and 200 test observations, with test class counts 101/99.
Every pipeline fits `StandardScaler` only on training data.

Logistic regression uses `C=1`, `max_iter=1000`. Both MLPs use hidden widths
`(8,8)`, L-BFGS, `alpha=0.001`, `tol=0.0001`, `max_iter=1000`,
`max_fun=15000` and `random_state=42`; their activation is identity or ReLU.
Other estimator settings use scikit-learn 1.9.1 defaults. The class threshold
is 0.5; no hyperparameter search or post-holdout selection was performed.
The official [MLPClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
describes these settings, including the identity activation and full-batch L-BFGS.

**Result:**

| Model | Train accuracy | Test accuracy | Test balanced accuracy | Test log loss | Iterations |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.555 | 0.575 | 0.574607 | 0.69207910 | 2 |
| Identity MLP | 0.555 | 0.575 | 0.574607 | 0.69207282 | 5 |
| ReLU MLP | 1.000 | 1.000 | 1.000000 | 0.00003094 | 11 |

No fit warnings were recorded for any model. Iteration counts are optimizer
diagnostics, not comparable computational-cost measurements.

**Interpretation candidate — author review required:** the result is consistent
with the nonlinear network learning useful features for this generator and the
identity control retaining its affine boundary limitation. Equal accuracies
do not make the identity MLP and logistic regression identical estimators.

**Limitation:** low-noise corner clusters are an intentionally easy constructed
problem. One seed, holdout and architecture comparison do not isolate every
optimization/regularization effect or establish performance on real data.
Zero observed holdout errors do not imply zero population error or calibrated
scores. This run provides no seed-sensitivity or confidence-interval study.

### B. Representation versus fitting on exact truth tables

**Hypothesis:** a perceptron can separate AND, a fixed two-ReLU network can
represent XOR, and a trained tanh MLP can fit its four exact corners.

**Configuration:** `from_scratch.py`; points ordered `(0,0),(0,1),(1,0),(1,1)`.
Perceptron: zero weights/bias, learning rate 1, fixed order, 100-epoch cap,
zero-margin updates, signed labels. NumPy MLP: `2 → 8 → 1`, seed 42,
tanh hidden layer, sigmoid output, fan-in-scaled independent normal weights,
zero biases, unregularized mean BCE, 10,000 full-batch updates, learning rate
0.5. All four points are used for training; there is no holdout.

**Result:** AND reached a zero-update epoch at epoch 9 after 18 updates,
with weights `[3,2]`, bias `-4` and training accuracy 1.0. XOR reached the
100-epoch cap with 400 updates, final weights `[0,0]`, bias 0 and final
training accuracy 0.5. Zero-margin updates count as updates even when the
prediction tie rule happens to give the correct class.

The fixed ReLU construction returned `[0,1,1,0]`. NumPy MLP mean training
BCE decreased from **0.7131771922** to **0.0002591694**. Class-1 scores were
approximately `[0.00010013, 0.99973258, 0.99972560, 0.00039457]`, giving
predictions `[0,1,1,0]` and training accuracy 1.0.

**Interpretation candidate — author review required:** the construction
establishes representability on the specified domain; this optimizer run
found a successful fit for the chosen seed. The perceptron result agrees
with the geometric XOR contradiction above.

**Limitation:** training on every truth-table point measures fitting, not
generalization. Failure at an epoch cap alone would not prove a capacity
limit; the separate geometric argument does. Successful training on four
points proves neither universal approximation nor convergence from all seeds.

## Practical decisions and common mistakes

An MLP can combine document embeddings and quality indicators into a prediction
head, but the label contract, leakage controls and evaluation population must
be established first. This is a potential application, not an implemented
document-processing system. Linear baselines and tree models deserve comparison
on representative validation data; XOR does not rank model families for tabular work.

Feature scaling affects optimization; fitting a scaler before splitting leaks
information. A larger architecture changes capacity and cost, but can overfit
or be harder to train. Sigmoid outputs require separate calibration checks.
Hidden units are distributed features, not guaranteed semantic detectors.
Serving requires consistent preprocessing and feature schemas, labeled
monitoring and resource measurements that this local exercise does not supply.

## Suggested extensions — not executed

- Repeat initialization and data seeds, keeping the final evaluation protocol fixed.
- Increase noise from 0.12 to 0.30; inspect overlap without assuming a numerical outcome.
- Compare widths 2, 4, 8 and 32 using training/validation data for selection.
- Compare tanh and ReLU under disclosed initialization and optimization settings.
- Add the explicit interaction $x_1x_2$ to a linear baseline to separate feature
  engineering from learned representation.

These remain proposals. Any new execution should append a dated configuration,
results, interpretation candidate and limitation rather than overwrite this record.

## Local verification — 2026-11-18

Both documented scripts executed in the environment above. The focused command
`python scripts/validate_repo.py tests --scope 04-deep-learning/53-perceptron-mlps`
passed **24 tests**, with no skips or warnings, using pytest 9.2.0. Repository
syntax and internal-link checks, installed-dependency compatibility and
`git diff --check` also passed. These are local checks, not a full-curriculum
test run, CI execution, GitHub math-rendering inspection or author approval
of the interpretation candidates.
