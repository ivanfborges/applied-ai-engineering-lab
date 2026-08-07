# Interview Questions: Statistical EDA

## 1. Why report both mean and median?

They encode different notions of center. The mean uses magnitudes and is
appropriate when expectation matters, but it is sensitive to extremes. The
median uses order and is robust. Their gap can reveal asymmetry or influential
observations, although it does not identify the cause.

## 2. Why does sample variance commonly divide by \(n-1\)?

The sample mean is estimated from the same data, which consumes one degree of
freedom. Under IID sampling, dividing the sum of squared deviations by \(n-1\)
makes sample variance unbiased for population variance. This does not make the
sample standard deviation exactly unbiased.

## 3. When is median plus IQR preferable to mean plus standard deviation?

When the objective is a representative central range under strong skew,
heavy tails, or contamination. Mean and standard deviation may still be
required when expected cost or squared-error behavior is the actual question.
The choice follows the decision, not a rule that robust measures are always
better.

## 4. What should you state when reporting skewness or kurtosis?

State whether the statistic is a population moment or a finite-sample
estimator, whether a bias correction is applied, and whether kurtosis is raw or
excess. Different libraries can return different values from the same small
sample because conventions differ.

## 5. Is kurtosis just a measure of peakedness?

That description is incomplete. Kurtosis is the standardized fourth central
moment, so extreme deviations receive fourth-power weight. Tail contribution
is usually the more operationally useful interpretation.

## 6. Should observations outside an IQR fence be removed?

No automatic action follows from the flag. First inspect lineage, units,
parsing, subgroup membership, temporal context, and domain constraints. The
observation could be an error, a valid rare event, or exactly the phenomenon
the system must detect.

## 7. Why can a three-standard-deviation rule fail on skewed data?

The mean and standard deviation are themselves influenced by skew and extreme
values, while the usual intuition for the threshold is Gaussian. A log-normal
tail can contain many legitimate values that the heuristic labels unusual.
Domain rules, quantiles, IQR, or MAD-based diagnostics may be more informative.

## 8. Pearson or Spearman correlation?

Pearson measures linear association in the original values. Spearman measures
linear association of ranks and therefore monotonic association. Use the one
that matches the question, inspect the scatter plot, account for ties and
missing values, and do not treat either as causal evidence.

## 9. Can zero correlation coexist with perfect dependence?

Yes. With a symmetric \(x\), the deterministic relationship \(y=x^2\) can have
Pearson correlation zero because positive and negative cross-products cancel.
Correlation is not a general test of independence.

## 10. What is wrong with selecting features from the largest correlation
matrix entries?

Large values may reflect leakage, duplicated features, common trends,
confounding, subgroup composition, or chance under multiple comparisons. Low
Pearson correlation can also hide nonlinear predictive information. Feature
selection must consider model class, validation design, availability at
prediction time, and domain plausibility.

## 11. How can Simpson's paradox affect an EDA conclusion?

An aggregate association can differ from or reverse every subgroup association
when group proportions differ. Report relevant conditional summaries and
understand the process that assigns observations to groups before interpreting
the aggregate.

## 12. A service has mean latency 400 ms. Is that enough to assess it?

No. Ask for sample count, median, P90/P95/P99, maximum, error and timeout rates,
time window, request mix, component timings, and segments such as model,
region, or request type. A reasonable mean can coexist with unacceptable tail
latency.

## 13. Where does leakage enter EDA?

Through features recorded after the outcome, target-derived fields, full-data
preprocessing statistics, future observations in time series, or exploratory
choices evaluated on a nominal test set. EDA must respect the deployment-time
information boundary.

## 14. How would you perform EDA for a RAG system?

Inspect document/chunk lengths, chunks per document, query and context tokens,
retrieval and reranking scores, answer tokens, latency, cost, and evaluation
scores. Segment by language, document type, parser, query type, model version,
and outcome. Investigate missingness and tails, preserve temporal boundaries,
and distinguish exploratory patterns from validated quality claims.

## Interview-ready summary

EDA is the process of characterizing data quality, distributions, unusual
observations, and relationships before making modeling claims. The senior-level
skill is choosing statistics whose assumptions fit the data and decision:
comparing mean with median, standard deviation with IQR or MAD, stating
estimator conventions, investigating rather than blindly deleting outliers,
and matching Pearson or Spearman to the relationship. In production AI, the
same reasoning applies to tokens, latency, retrieval scores, costs, and
evaluation metrics, with special attention to tails, subgroups, time, and
leakage.
