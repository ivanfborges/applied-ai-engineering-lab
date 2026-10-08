# Explainability: definitions before interpretation

Let f(x) be a scalar output of a **fixed fitted model**. For this example it
is P(y=1), rather than a predicted label or an unspecified raw margin.
A global importance statistic, a replacement response, and a local attribution
describe different properties of f. Intrinsic interpretability concerns model
structure; post-hoc inspection adds a defined measurement after training.

## Impurity-based feature importance

For a classification node t, with class proportions p(k,t),

$$
G(t)=1-\sum_k p(k,t)^2.
$$

A split's reduction is

$$
\Delta G(t)=G(t)-\frac{n_L}{n_t}G(L)-\frac{n_R}{n_t}G(R).
$$

Feature j accumulates (n_t / n_root) times this reduction at nodes that use j.
Weighted sample counts replace counts when weights are used. Tree importances
are normalized and aggregated for a forest; they need not equal held-out
performance gains or the performance change after removing a feature.

Continuous and high-cardinality inputs offer more split candidates. Training
overfit and this selection bias can produce nonzero MDI for irrelevant noise.
Correlated inputs can share or compete for split credit. See the official
[permutation importance discussion](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Permutation: reliance under a perturbation

For a higher-is-better score s and evaluation dataset D,

$$
\widehat{PI}_j=s(f,D)-\frac{1}{R}\sum_{r=1}^R s(f,D_j^{\pi_r}).
$$

The educational helper always subtracts the shuffled score from the baseline.
A negated loss such as negative MSE therefore gives the **increase in loss**.
An AUC decrease of 0.1 is a score difference, not a 10% accuracy decline.
Negative estimates are retained: finite samples, chance, or harmful model
reliance can make the shuffled score larger.

No model is refitted. A retraining ablation changes the fitted function and
allows other features to substitute, so it answers another question.
Changing the metric, reference population, model, or shuffle procedure can
change the ranking. Labels must remain aligned with the original rows.

For a group G, one row permutation is applied to every column in G. This
preserves relationships *within* the group while breaking its alignment with
the remaining inputs and target. Independently shuffling each group member
would destroy its internal dependence and define a different perturbation.
Groups may overlap when comparing a pair to its individual members.

Correlated features do not guarantee that both individual importances are low:
the forest may rely heavily on just one. Joint importance can be larger than
either individual value, but it is not generally their sum. Cross-group
dependence can still make a group shuffle unrealistic.

Both implementations use population standard deviation across repeats
(ddof=0). It describes shuffle variability conditional on the frozen dataset,
rather than a confidence interval for population importance. Our fresh
default_rng shuffles differ from scikit-learn's random stream. The tests compare
Monte Carlo estimates within sampling error and check exact known invariants;
they do not assert identical draws.

## PDP and ICE: what is averaged?

For one feature j and fixed reference rows,

$$
ICE_i(z)=f(z,x_{i,-j}),\qquad
\widehat{PD}_j(z)=\frac{1}{n}\sum_i ICE_i(z).
$$

This averages over the **marginal** empirical distribution of the other
features. It is generally different from E[f(X) | X_j=z], which changes
the distribution of the remaining inputs with z. See the official
[PDP/ICE definitions](https://scikit-learn.org/stable/modules/partial_dependence.html).

An ICE curve retains one reference row's response. Its name does not mean
it estimates a statistical conditional expectation. Centering at a grid
anchor removes vertical offsets and can emphasize response heterogeneity.

Nonparallel probability curves alone do not identify interactions on a
log-odds scale: a sigmoid can bend the response of an additive logit model.
The synthetic generator here actually contains an interaction, but that
does not validate every apparent interaction in the fitted forest.

The direct helper replaces one column, predicts on a copy, and retains an
n-by-m array. Averaging those exact curves gives its PDP. The example uses
scikit-learn's brute method, explicitly selects probabilities, and compares
the same rows and grid. Using an average over all rows with ICE on a subsample
would break this exact identity.

A 5th–95th percentile grid limits univariate extremes; it does **not** ensure
joint support. Replacing signal while retaining its proxy is deliberately
problematic here. Inspect the joint distribution before treating a response
as behavior on realistic inputs. ALE uses local differences and can reduce
some extrapolation problems, but sparse regions and causal identification
remain separate concerns. ALE is discussed only; it is not implemented.

## SHAP: define the coalition game

For features F of size p and coalition value v,

$$
\phi_j=\sum_{S\subseteq F\setminus\{j\}}
\frac{|S|!(p-|S|-1)!}{p!}
\left[v(S\cup\{j\})-v(S)\right].
$$

Each weight represents how frequently coalition S precedes j in all feature
orders. For a consistent scalar game,

$$
f(x)=v(\varnothing)+\sum_j\phi_j.
$$

The [original SHAP paper](https://arxiv.org/abs/1705.07874) connects Shapley
allocation to additive feature explanations. For f(x1,x2)=x1*x2 at (2,3),
a fixed zero reference gives empty and singleton coalition values of zero
and a full value of six. Each feature receives three. This allocates an
interaction within that specified game; it does not estimate two independent
causal effects.

A conditional game uses E[f(X) | X_S=x_S] and can allocate credit to an unused
feature carrying information about used ones. Marginal replacement uses
E[f(x_S, X_not_S)] over a background distribution and may evaluate implausible
combinations. Tree path dependence is a tree-specific approximation based on
training path counts; it should not be described as the true observational
conditional distribution.

The example explicitly sets interventional TreeSHAP, a training background,
and probability output. Its multi-output classifier explanation is checked
for rows-by-features-by-classes shape. The requested class is located from
classes_, rather than assuming that a hard-coded column always means positive.
See [TreeExplainer's API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

A probability SHAP contribution of +0.20 contributes 20 percentage points
in an additive decomposition relative to the baseline. A log-odds contribution
of +0.20 has different units. Neither is a predicted intervention benefit.
Mean absolute SHAP aggregates attribution magnitude; it is not an AUC loss.

The helper checks baseline equality with the background's mean prediction
and verifies reconstruction within 1e-6. Backgrounds larger than 100 rows
are explicitly rejected to avoid silently changing the intended reference
through masker subsampling. Supporting larger backgrounds requires explicitly
configuring and checking the effective masker; it is outside this helper's
educational scope. Background sensitivity and retraining stability remain
unmeasured.

## Operational use and common mistakes

Start with valid evaluation and feature availability. A highly attributed
post-decision field may expose leakage. Then choose a method for the actual
question: performance reliance, response behavior, or individual attribution.
Version the model, output class, reference distribution, metric, and inspection
population so future comparisons are meaningful.

For selective explanations at high prediction volume, sample global diagnostics
offline and explain individual cases where needed. Evaluate stability across
cohorts and model versions before building a business narrative. Do not infer
policy correctness, calibration, causality, or permissible feature use from
a successful numerical identity.

The numerical helpers accept finite numeric arrays and scalar prediction
functions. They omit sample weights, categorical masking, pipelines with
feature names, conditional shuffles, and parallel execution. Their cost is
one baseline plus R evaluations per group; direct response curves need one
prediction batch per grid point and retain n*m values. They are educational
implementations rather than production library replacements.

## Executed forest experiment

Interpretations in this forest record and the linear control below received an
AI-assisted technical review on October 8, 2026, delegated by the author; see
the [curation record](../../docs/CURATION.md#delegated-technical-review--october-8-2026).
No new execution was performed. The separate interactive lab and exports are
outside the reviewed scope.

**Hypothesis.** Correlated signal may receive different allocations under MDI,
metric-based permutation, and SHAP. Joint permutation may reveal reliance not
captured by either individual perturbation.

**Configuration.** Executed with seed 35, 1,800 synthetic rows, stratified
75/25 train/test split (1,350/450), and no hyperparameter search.
signal and context are independent standard normals; signal_proxy is
signal plus independent Normal(0, 0.15^2) noise; noise is independent.
The outcome is Bernoulli with probability
sigmoid(1.4*signal + 0.9*context + 1.6*signal*context).
The forest uses 120 trees, leaf minimum five, sqrt feature sampling, bootstrap,
unrestricted depth, and one worker. Permutation uses held-out ROC-AUC and ten
repeats. PDP/ICE use the first 150 held-out rows, a 21-value signal grid over
the 5th–95th percentiles, and positive-class probability. SHAP uses 100 randomly
sampled training rows and the first 40 held-out rows. No diagnostic changes
the fitted model or selects features.

Environment: Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0, SHAP 0.50.0.
The ignored JSON report records versions, selected row indices and raw curves.
These results describe that environment; compatible versions may differ.

**Results.** Held-out ROC-AUC was **0.815720** and signal/proxy correlation
was **0.988423** (rounded; use the report for full precision).

| Feature | MDI | Library AUC decrease | Repeat SD | Scratch AUC decrease | Repeat SD | Mean absolute probability SHAP (40 rows) |
|---|---:|---:|---:|---:|---:|---:|
| signal | 0.296671 | 0.172104 | 0.019224 | 0.182565 | 0.017182 | 0.140110 |
| signal_proxy | 0.227843 | 0.018422 | 0.006772 | 0.019186 | 0.006711 | 0.053604 |
| context | 0.365781 | 0.199472 | 0.012572 | 0.194142 | 0.022253 | 0.089615 |
| noise | 0.109705 | 0.004710 | 0.004371 | 0.008450 | 0.002427 | 0.022009 |

Scratch joint signal/proxy permutation decreased AUC by **0.256499**, with
repeat SD **0.020833**. This exceeds either individual estimate in this run.
The noise feature had nonzero MDI despite lacking a direct generator term;
a single run does not establish the source or significance of that credit.

Direct PDP/ICE matched the library within absolute tolerance 1e-12.
The mean absolute signal/proxy gap in the response reference was **0.121509**;
after signal replacement across the grid it was **1.156943**. This diagnostic
describes altered feature relationships rather than a formal support test.

SHAP's background mean probability was **0.464369**. The first explained row's
contributions were signal -0.251356, proxy -0.071259, context -0.054181 and
noise +0.009238, reconstructing probability **0.096812** with the baseline.
The maximum reconstruction error across 40 rows was **5.48e-08**.

**Technical interpretations — delegated review.**

- Joint permutation had a larger recorded AUC decrease than either individual
  estimate. Under this frozen model and perturbation protocol, the dependent
  pair matters as a group. This is consistent with reliance on their shared
  signal, not proof of a unique mechanism, additive importance or causal effect.
- MDI, probability SHAP magnitude and AUC decrease measure different quantities;
  their magnitudes and rankings are not interchangeable. The enlarged
  replacement gap shows disruption of the signal/proxy relationship and motivates
  inspection of implausible PDP combinations; it is not a formal support test.
- SHAP reconstruction and matched PDP/ICE calculations support class/output
  bookkeeping and agreement for the specified rows, grid and reference. They do
  not validate causality, calibration, explanation stability or deployment fitness.

**Limitations.** One seed and population; modest evaluation and explanation
samples; ten permutations; no background or model stability sweep. Differences
between library and scratch estimates include different shuffle streams.
All four features are continuous, so this run does not isolate cardinality bias.
The gap diagnostic does not identify every out-of-support replacement.
The SHAP ranking covers 40 rows, not all held-out data. None of these results
establish causality, population significance, calibration, or deployment fitness.

## Executed linear control

**Hypothesis.** A model fitted to y=2*x0 should have zero reliance on unused x1
and PDP values -2, 0, 2 at grid -1, 0, 1.

**Configuration.** Scratch script; independent standard-normal numeric inputs;
100 training and 60 test rows; NumPy generator seed 35; LinearRegression;
ten permutations scored with negative MSE.

**Result.** Mean MSE increase was **7.654766** for x0 and numerically zero
for x1. PDP predictions were **-2, 0, 2** to printed precision.

**Technical interpretation — delegated review.** To the reported numerical
precision, this control is consistent with the specified function, the
negative-MSE score convention and zero reliance on the unused feature. It is
a known-function implementation check, not validation on complex real inputs.

**Limitation.** A noiseless known-function check is not an empirical estimate
of explainability reliability on complex or dependent real inputs.

## Suggested experiments — not run

Change proxy noise to 0.01 and 1.0; refit and compare all three importance
definitions. Remove the interaction and compare centered ICE in both probability
and logit space. Compare ROC-AUC, average precision and negative log loss.
Repeat SHAP with backgrounds of 25 and 100 rows and different sample seeds,
keeping explained rows fixed. Repeat training and inspection across seeds and
cohorts. Each proposed run needs its own hypothesis, configuration, result,
interpretation candidate and limitation before any conclusion is recorded.
