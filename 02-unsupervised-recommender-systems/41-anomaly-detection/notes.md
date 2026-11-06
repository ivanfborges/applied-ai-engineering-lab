# Anomaly scores, reference populations and alert decisions

## Define normal before fitting

Outlier detection allows contamination in fitting data; novelty detection uses
a reference believed to be normal and scores future observations. This example
uses the latter. A baseline excluding known incidents can still omit legitimate
operating regimes. A rare customer or new document format is not automatically a failure.

Point anomalies concern individual rows. Contextual anomalies depend on the
regime, such as latency given traffic and model version. Collective anomalies
concern sequences or windows. Row detectors only see the supplied features.
For temporal use, build features using information available at scoring time;
split chronologically into fit, calibration and test periods.

## Robust statistics and their blind spot

For feature $j$, fit the median $m_j$ and normal-consistent scale $a_j$:

$$
m_j=\operatorname{median}_i x_{ij},\qquad
a_j=1.4826022185\operatorname{median}_i|x_{ij}-m_j|.
$$

The baseline score is $s(x)=\max_j |x_j-m_j|/a_j$. This maximum is an explicit
aggregation choice: more features create more opportunities for an extreme
score. Calibrate the combined score, rather than treating a marginal cutoff
as a joint false-positive guarantee. Gaussian scaling does not make skewed
telemetry Gaussian or provide a calibrated tail probability.

Zero MAD occurs for constant and discrete features. The implementation rejects
it instead of silently adding an arbitrary epsilon; a domain rule or a different
feature model may be appropriate. Median/MAD resist extreme fitting values but
cannot describe a slanted normal region. A robust covariance distance can capture
linear dependence with an approximately elliptical reference, but is not
automatically suitable for multimodal or high-dimensional data.

## Isolation as a path-length calculation

Sample $\psi$ rows per tree without replacement, split a varying feature at a
random threshold, and stop at singleton/unsplittable leaves or
$\lceil\log_2\psi\rceil$ depth. A stopped leaf with $n$ rows contributes an
additional expected search path $c(n)$:

$$
c(0)=c(1)=0,\qquad
c(n)=2H_{n-1}-\frac{2(n-1)}n\quad(n\ge2).
$$

With corrected mean path $\bar h(x)$, the original score is
$s(x)=2^{-\bar h(x)/c(\psi)}$. Short paths give larger scores. The scratch forest
requires at least two fitting rows and samples; duplicates can prevent splitting.
An entirely unsplittable reference gives score 0.5 even far outside its support:
the score is not a universal distance or probability.

The library's `score_samples` has the opposite orientation. Its decision function
subtracts `offset_`; changing `contamination` sets that cutoff rather than changing
the random tree mechanism with an otherwise fixed configuration. This study
calibrates a separate cutoff and does not use estimator `predict`.
[API contract](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

## One-Class SVM learns support

The primal problem separates the reference data from the origin in kernel space:

$$
\min_{w,\rho,\xi}\frac12\lVert w\rVert^2+
\frac1{\nu n}\sum_i\xi_i-\rho,
\qquad w^T\phi(x_i)\ge\rho-\xi_i,\quad\xi_i\ge0.
$$

For the RBF kernel, $K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)$. Larger gamma makes
similarity more local. Scaling changes these distances; the example fits
RobustScaler only on training features. Nu bounds training violations and the
support-vector fraction in the standard formulation; finite solver tolerance
matters. It does not specify the fraction of future alerts. Kernel fitting can
be costly as sample count grows; benchmark resources on the actual workload.
[API contract](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html).

## Calibration is an alert policy

For higher-is-more-anomalous scores, set
$t=Q_{1-\alpha}(s(X_{cal}))$ and alert when $s(x)>t$. Here the empirical quantile
uses NumPy `method="higher"`: choose the upper observed order statistic, with
strict comparison at ties. It can produce fewer calibration alerts than
$\alpha n_{cal}$, especially in small or tied samples. This is a simple empirical
policy, not a distribution-free coverage claim or guaranteed future FPR.

Calibration normality, independence and representativeness are assumptions.
With 200 rows, the 0.5% tail setting selects the sample maximum and yields no
calibration alerts. That is not evidence of zero future false positives. No
threshold is chosen by looking at test precision or recall in this comparison.

AP summarizes precision weighted by recall increments; it is not trapezoidal
PR area. It depends on prevalence. Here 40/340 positives is an artificial
11.76%, far above many monitoring settings. Precision is $TP/(TP+FP)$, recall
is $TP/(TP+FN)$ and FPR is $FP/(FP+TN)$. Under a fixed population model,
precision equals $\pi\,TPR/[\pi\,TPR+(1-\pi)FPR]$, showing why small FPR can
still cause many false alerts at low prevalence. Without reliable incident
labels, report burden and reviewed cases rather than invented precision/recall.

## Executed experiment — November 6, 2026

**Executor:** Codex. **Interpretation status: author review required.**
New nonvisual run at 09:42:45 UTC (06:42:45 America/Sao_Paulo).

**Hypothesis:** featurewise deviations can miss anomalous relationships; changing
thresholds for a frozen score changes decisions without changing AP.

**Configuration:** Windows x64; Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1. Seed 41. Independent standard-normal latent coordinates
$z_1,z_2$ generate latency $250+35z_1$ and queue
$40+8(0.8z_1+0.6z_2)$. Fit/calibration/normal test sizes: 800/200/300.
Test injections: 25 independent-coordinate Gaussian rows with location (285,25)
and scales (12,4), plus 15 with location (375,70) and scales (15,6).
MAD takes the maximum featurewise robust deviation. IsolationForest uses 200
trees, 256 samples, `contamination="auto"`, seed 41 and one job. OneClassSVM
uses training-fitted RobustScaler defaults, RBF, `nu=0.05`, `gamma="scale"`.
All other estimator parameters retain library defaults. Both detector and scaler
are frozen before calibration; test labels are evaluation-only.

Command: `python 02-unsupervised-recommender-systems/41-anomaly-detection/example.py`
(executed with the local stable reference environment's Python).

**Observed results at $\alpha=0.02$:**

| Detector | AP | Threshold | Alerts | TP / FP / FN / TN | Precision | Recall | FPR | Relationship recall | Extreme recall |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| MAD | 0.693984 | 2.648598 | 20 | 17 / 3 / 23 / 297 | 0.850000 | 0.425000 | 0.010000 | 0.080000 | 1.000000 |
| Isolation Forest | 0.830465 | 0.680306 | 17 | 15 / 2 / 25 / 298 | 0.882353 | 0.375000 | 0.006667 | 0.000000 | 1.000000 |
| One-Class SVM | 0.965663 | 0.422586 | 39 | 37 / 2 / 3 / 298 | 0.948718 | 0.925000 | 0.006667 | 0.880000 | 1.000000 |

All three had 3/200 calibration alerts at this setting; future test FPR differed.
Thresholds have model-specific units and must not be compared numerically.

**Observed policy sweep, without refitting:**

| Detector | Alerts at alpha 0.05 / 0.02 / 0.005 | Recall at alpha 0.05 / 0.02 / 0.005 |
|---|---|---|
| MAD | 33 / 20 / 15 | 0.525 / 0.425 / 0.375 |
| Isolation Forest | 41 / 17 / 16 | 0.725 / 0.375 / 0.375 |
| One-Class SVM | 43 / 39 / 29 | 0.950 / 0.925 / 0.700 |

AP was unchanged within each detector across these three policies. Calibration
alert counts were 9, 3 and 0 respectively, for each detector.

**Interpretation candidates — author review required:** the SVM represented this
correlated reference and these injections favorably. Isolation Forest ranked
better than MAD by AP yet recalled fewer anomalies at the 2% calibration setting,
including no relationship injections. Distinguish ranking from the chosen policy;
do not generalize this result to other feature geometries or incident classes.

**Limitations:** one IID synthetic seed, assumed clean reference, two artificial
fault distributions, no time dependence, no resource benchmark, no production
labels and no business-cost evaluation. No hyperparameter search or real-system
benefit is established. The scratch forest is checked by numerical/behavioral
tests; it is not included as a fourth ranked detector.

**Reproduction check:** a separate run on the same date at 09:46:51 UTC
used the same versions/configuration and reproduced all nine metric rows.
Its timestamped local JSON record is separate from the initial run. This checks
repeatability in this environment, not variability across seeds or machines.

**Local software verification:** both scripts executed; the focused Day 41
suite passed 41 tests with no skips or warnings. Repository parsing passed for
265 Python files; internal-link checks passed for 836 targets in 188 Markdown
files. Dependency compatibility and `git diff --check` passed. These checks do
not establish full-suite CI, rendered GitHub math or scientific author review.

## Practical monitoring decisions

| Signal and unit | Context to retain | Investigation before model changes |
|---|---|---|
| Request latency and queue depth | Service/version and workload | Units, schema, deployment changes, saturation |
| Tokens per completed document | Input length and task type | Legitimate document mix, retries and truncation |
| Empty retrieval rate per time window | Index version, traffic volume | Missing index data, filters, low denominators |
| Tool calls per agent task | Workflow type and retry budget | Repeated loops, tool failures and execution traces |

These are possible applications, not deployments performed here. Establish
actions, owners, severity and investigation capacity. Monitor missingness,
scores and alert counts by relevant segment. Add persistence, deduplication and
escalation when the operational unit warrants them. A score distribution shift
does not identify concept drift or root cause by itself. Review incidents before
retraining so the new baseline does not normalize failures; retain model,
feature and threshold versions with rollback options.

Hard validity/SLO rules can be more direct than a learned detector. For seasonal
series, consider contextual baselines or forecast residuals; for persistent
changes, consider change-point methods. PCA reconstruction can miss anomalies
inside its retained subspace; DBSCAN noise and t-SNE/UMAP separation are not
incident labels.

## Suggested experiments — not executed

Vary seeds and injection prevalence, contaminate fitting/calibration separately,
add irrelevant features, and replay chronological day/night regimes. Compare
gamma settings using a development protocol that preserves a final untouched
test population. Test model/threshold stability and delayed incident labels.
No measurements for these extensions are claimed.
