# Missing data: assumptions, estimates, and availability

## The value and the process that hides it

Let `Z` contain the full variables relevant to an analysis and let `R_j=1`
mean feature `j` is observed; `M_j=1-R_j` marks absence. The notation
`Z_obs, Z_mis` refers to the observed and missing components for a particular
pattern. Definitions concern the conditional distribution of that pattern:

$$
\begin{aligned}
\text{MCAR: }&P(R\mid Z_{obs},Z_{mis})=P(R),\\
\text{MAR: }&P(R\mid Z_{obs},Z_{mis})=P(R\mid Z_{obs}),\\
\text{MNAR: }&P(R\mid Z_{obs},Z_{mis})\text{ still depends on }Z_{mis}.
\end{aligned}
$$

MAR is a conditional independence statement. If income is missing more often
for self-employed people and employment status is observed, a MAR assumption
can be plausible. If high income itself raises nonresponse probability after
conditioning on all available variables, the mechanism is MNAR. An omitted
or unavailable driver can change which assumption is defensible.

Associations between missingness and observed variables can contradict MCAR.
Failure to reject an MCAR test does not prove it, and observed records generally
cannot distinguish MAR from MNAR. In the example, mechanism labels are known by
construction because the masks are generated from complete synthetic data.
That privileged knowledge is absent in ordinary applications.
See [Rubin's original formulation](https://doi.org/10.1093/biomet/63.3.581).

Under MCAR, retained complete cases are representative of the full population
in the usual independent sampling setting, but estimates lose precision.
Even 5% independently missing cells across 20 features leave only
`0.95**20 ≈ 36%` complete rows. Under MAR, deleting rows can change the
population represented; validity depends on the estimand and model, so deletion
is not categorically invalid or safe. Dropping columns can remove essential
predictive signal or adjustment variables.

For likelihood inference, ignoring the missingness mechanism usually also
requires distinct parameters for the data model and missingness model, alongside
an appropriate MAR assumption. MAR alone is not permission to use arbitrary
imputation and ordinary complete-data standard errors.

## What deterministic filling preserves

For a single column with `n_obs` observed entries among `n` rows, fill each
missing entry with the **same observed-column mean** `xbar_obs`. The completed
mean equals `xbar_obs` exactly. With population-style variance (`ddof=0`),

$$
v_{filled}=\frac{n_{obs}}{n}v_{obs}.
$$

With sample variance, when `n_obs > 1`,

$$
s^2_{filled}=\frac{n_{obs}-1}{n-1}s^2_{obs}.
$$

These are algebraic identities relative to the observed sample, not evidence
that its mean is unbiased for the full population. If evaluation data are filled
with a **training** mean or median, the filled evaluation mean need not equal
its observed-only mean and those variance identities do not apply directly.

Under ideal MCAR, with population mean filling and observation probability
`q`, `X*=mu+R(X-mu)` gives
`Var(X*)=q Var(X)` and, when `Y` remains observed,
`Cov(X*,Y)=q Cov(X,Y)`. Those statements require missingness independent of
`(X,Y)`; arbitrary MAR/MNAR filling can distort covariance in other ways.
It is too strong to claim that every missing-data strategy always reduces
every correlation.

Even conditional-mean filling `E[X_j | X_-j]` removes residual uncertainty.
The law of total variance makes the omitted term explicit:

$$
Var(X_j)=Var(E[X_j\mid X_{-j}])+E[Var(X_j\mid X_{-j})].
$$

Good predictive performance after filling is a different question from recovery
of a distribution, a scientific coefficient, or its uncertainty.

## Choose a strategy for the objective

| Strategy | Useful property | Assumption or cost to inspect |
|---|---|---|
| Constant / explicit unknown category | Simple and semantically visible | Zero must not conflate a legitimate value and unavailable data; distinguish not applicable from failed extraction |
| Mean / median | Cheap, deterministic baseline | Ignores conditional relationships; mean is sensitive to outliers, median is robust in location but does not restore a distribution |
| Mode | Simple categorical filling | Inflates the dominant category; an explicit missing category has different semantics |
| KNN | Uses local observed-feature geometry | Requires meaningful units and neighbors; donor searches cost time/memory and overlap may be sparse |
| Iterative conditional modeling | Uses multivariate relationships | Conditional models, convergence, and missingness assumptions matter; one deterministic result is not multiple imputation |
| Multiple imputation | Represents missing-value uncertainty for inference | Requires plausible stochastic draws and valid analysis/pooling models |
| Native tree routing | Learns a branch for NaN | Estimator-specific behavior; does not identify MNAR or guarantee robustness under shift |

KNN estimates a missing feature from training neighbors **observed on that
feature**; usable neighbors can differ by feature. In scikit-learn's
NaN-aware Euclidean distance, if `O_ab` is the coordinates observed in both rows,

$$
d(a,b)=\sqrt{\frac{p}{|O_{ab}|}\sum_{j\in O_{ab}}(a_j-b_j)^2}.
$$

No shared coordinates means an undefined distance. The implementation can fall
back to a training feature mean when no usable distances exist. Scaling must
precede distance computation and fit only observed training values; filling
before computing geometry changes which evidence counts as observed.
[Official KNN documentation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
describes donor and indicator behavior.

The [scaling documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
confirms that NaNs are excluded at fit and preserved at transform. The example
uses `StandardScaler → KNNImputer → RandomForestClassifier`. The forest can
operate on standardized coordinates; an inverse transform is unnecessary here.

For proper multiple imputation, let `Q_k` be a scalar estimate and `U_k`
its estimated within-imputation variance across `m>=2` completed datasets:

$$
\bar Q=\frac1m\sum_k Q_k,\quad
\bar U=\frac1m\sum_k U_k,\quad
B=\frac1{m-1}\sum_k(Q_k-\bar Q)^2,\quad
T=\bar U+(1+1/m)B.
$$

The between-imputation term reflects additional uncertainty. Repeating a
deterministic median imputer produces no such variability. Confidence intervals
also require suitable degrees of freedom and analysis assumptions; pooling
arbitrary classifier scores with these equations is not justified.
[MICE](https://www.jstatsoft.org/article/view/v045i03) develops chained-equation
imputation and pooling; this topic explains the principle but does not implement
an inferential multiple-imputation workflow.

An inferential imputation model may need the observed outcome to preserve
associations. A deployed predictive preprocessing pipeline cannot depend on an
outcome unavailable at prediction time. These are different statistical tasks.

## Indicators and schema boundaries

A predictive model can use `f(X_filled, M)`, but adding indicators is a
hypothesis to validate. It may learn a collection policy that later changes.
For example, selectively extracting amounts from high-quality documents can
make absence informative until an OCR outage makes it widespread.

Both the educational class and the example's `add_indicator=True` create
indicator columns only for features that had missing values at fit. New
missingness on a previously complete feature is filled using its stored median
but receives no new indicator. The educational class tests this explicitly.
An all-feature indicator schema can reserve columns, but a constant indicator
during training provides no examples from which to learn outage behavior.

The scratch implementation rejects all-empty training columns because no median
exists. It validates dense numerical shapes, prohibits infinity, reuses training
medians, and leaves inputs unchanged. It cannot detect reordered unnamed columns,
handle sparse/categorical input, or replace sklearn's estimator API.
Scikit-learn imputers offer explicit empty-feature policies; schema decisions
should be deliberate, not accidental.

[Native histogram boosting](https://scikit-learn.org/stable/modules/ensemble.html#missing-values-support)
learns NaN routing from split gains. For a feature with no training NaNs, prediction
NaNs go to the child with more training samples. The `x2` outage therefore
tests previously unseen missingness for both indicator and native workflows.

## Operational robustness and common mistakes

Monitor per-feature missing rates, joint patterns, rows with any missing value,
critical feature availability, and downstream errors/calibration when labels
arrive. Track the provenance of missingness: no history, not applicable, failed
join, and timeout may share a numerical NaN while requiring different responses.

Do not fit an imputer or scaler before splitting, refit on a test batch, or use
test values as KNN donors. Cross-validation needs a fresh fitted pipeline inside
each fold. Do not use future targets or later records to restore unavailable
features. Forward filling in time series requires causal ordering and suitable
group boundaries; interpolating from future observations can leak.

Predictive accuracy under an unchanged missingness policy does not establish
robustness. In document intelligence, include extraction status and evaluate
failed-field cohorts. For a critical unavailable feature, an evaluated fallback
or manual review policy may be more appropriate than unconditional scoring.
These are design considerations, not deployed behavior in this study.

## Executed experiment

**Hypothesis specified before scoring:** imputation choices and indicators may
change prediction quality; additional feature loss may degrade frozen models.
Median-filled values may also differ from complete synthetic feature distributions.

**Configuration:** `make_classification`, 2,400 rows, eight features, five
informative and one redundant, `class_sep=1.0`, `weights=[0.5,0.5]`,
`flip_y=0.02`, `shuffle=False`, seed 42. The first five features are
informative because column shuffling is disabled. Stratified 60/20/20 splitting
uses seed 42 at both split steps. Masks use seeds 101/102/103 for
train/validation/test, with:

- `x0`: independent missing probability 0.15 (MCAR component).
- `x1`: probability 0.35 when observed `x2>0`, otherwise 0.05 (MAR component).
- `x3`: probability 0.50 when its complete value exceeds 0.8, otherwise 0.05
  (MNAR component). Other columns remain observed.

The combined mask contains MNAR; component names do not claim that the overall
dataset is simultaneously MCAR and MAR. The MAR driver stays available in
training and matched evaluation. It disappears only in the outage stress test.

Random forests use 120 trees, minimum leaf size 3, seed 42, and one worker.
KNN uses five uniform neighbors after training-only standardization.
Histogram boosting uses 120 stages, 15 leaves, L2 regularization 1.0, seed 42,
and no early stopping. Other parameters use recorded library defaults.
Execution used Python 3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0, and a one-thread
numerical-library limit.

**Selection:** highest validation ROC-AUC, with dictionary order breaking ties.
All candidates freeze before any test evaluation; stress results do not select
or retune a workflow.

| Workflow | Validation ROC-AUC |
|---|---:|
| Median + random forest | 0.9649 |
| Median + indicators + random forest | 0.9672 |
| Scaled KNN + random forest | **0.9695** |
| Native histogram boosting | 0.9682 |

**Measured missingness:** rates are realized fractions, not requested probabilities.

| Partition/scenario | Rows | Positives | x0 missing | x1 missing | x2 missing | x3 missing |
|---|---:|---:|---:|---:|---:|---:|
| Training | 1,440 | 724 | 0.1514 | 0.2028 | 0 | 0.1660 |
| Validation | 480 | 242 | 0.1375 | 0.1771 | 0 | 0.1375 |
| Matched test | 480 | 241 | 0.1583 | 0.1813 | 0 | 0.1833 |
| Extra dropout test | 480 | 241 | 0.4896 | 0.4938 | 0 | 0.5146 |
| x2 outage test | 480 | 241 | 0.1583 | 0.1813 | 1 | 0.1833 |

Extra dropout applies independent probability 0.40 to each of `x0/x1/x3`
after the matched test mask, using seed 104. If initial missing probability is
`p`, combined probability is `p+(1-p)*0.40`; it is not simply 40% total.
The other stress scenario hides every `x2` on matched test data. Both preserve
the same underlying rows and labels.

**Measured test results:** each cell reports ROC-AUC / Brier score. Higher AUC
and lower Brier are desirable for their respective objectives.

| Workflow | Matched | Extra dropout | x2 outage |
|---|---|---|---|
| Median | 0.9644 / 0.0809 | 0.9395 / 0.1056 | 0.9222 / 0.1224 |
| Median + indicators | 0.9646 / 0.0785 | 0.9386 / 0.1071 | 0.9205 / 0.1239 |
| Scaled KNN | 0.9680 / 0.0756 | 0.9457 / 0.0936 | 0.9430 / 0.0991 |
| Native histogram boosting | 0.9717 / 0.0625 | 0.9359 / 0.1016 | 0.9023 / 0.1344 |

The validation-selected KNN's matched test complete-row group has 272 rows,
130 positives, AUC 0.9776, and Brier 0.0643. Its any-missing group has 208 rows,
111 positives, AUC 0.9545, and Brier 0.0904. All workflows and scenarios report
both groups with counts in the ignored JSON. Empty groups are null; single-class
groups have null AUC. Groups need not represent comparable populations.

**Distribution diagnostic:** for MNAR-masked `x3`, complete synthetic truth
has test mean -0.4588 and population-style variance 2.7990. Observed-only entries
have mean -0.7571 and variance 2.4028; filling with the training median yields
mean -0.7503 and variance 1.9625. Complete truth is available only because this
is a controlled synthetic experiment. Diagnostics for `x0/x1` are also recorded.

**Interpretation candidates — pending author review:**

- All four frozen workflows lost AUC and increased Brier under both tested
  feature-loss scenarios. This supports investigating availability as a separate
  evaluation condition for this generator.
- Indicators changed matched AUC only slightly here; they did not consistently
  improve the tested stress metrics. This does not show that indicators are
  generally unnecessary.
- The native workflow's stronger matched test metrics and larger outage loss
  cannot be attributed solely to native missing routing: its learner is different.
- The MNAR diagnostic is consistent with selective hiding of high `x3` values
  and distribution distortion after deterministic filling. It does not establish
  detection or correction of MNAR in observed-only data.

**Limitations:** one seed, one split, fixed configurations, no uncertainty
intervals, no repeated-split study, no measured production behavior, and no latency
benchmark. Correlated synthetic features can partly substitute for missing ones.
The mixed mechanism is not an experiment isolating each component's effect.
Group-score differences combine feature loss and selection; Brier combines
calibration and discrimination and is not a standalone calibration proof.

Reproduce with `python 01-classical-machine-learning/34-missing-data/example.py`
from the root. The ignored `outputs/comparison.json` is regenerable and is not
a public asset; this record is the intentional public measurement summary.

## Suggested follow-ups — not executed

Repeat across seeds/splits and report paired uncertainty; vary independent
missing probabilities over a prespecified grid; hold the classifier fixed when
comparing native versus imputed handling; test an evaluated fallback on feature
outage; inspect imputation error on separately hidden synthetic ground truth;
or perform an MNAR sensitivity analysis over assumed missing-value shifts.
None of these follow-ups has a reported outcome here.
