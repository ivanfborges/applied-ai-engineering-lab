# References

## Statistical foundations

- Donald B. Rubin (1976), [Inference and missing data](https://doi.org/10.1093/biomet/63.3.581), *Biometrika* 63(3), 581–592. Original formulation of missingness and ignorability; the publisher page may require institutional access.
- Stef van Buuren and Karin Groothuis-Oudshoorn (2011), [mice: Multivariate Imputation by Chained Equations in R](https://www.jstatsoft.org/article/view/v045i03), *Journal of Statistical Software* 45(3), 1–67. Conditional imputation models, stochastic completion, and pooling.

## Implementation documentation

- [scikit-learn: imputation](https://scikit-learn.org/stable/modules/impute.html): single/multiple imputation distinctions, empty-feature policies, and indicators.
- [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html): donor selection, distance, and missing-feature behavior.
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): observed-value fit statistics and NaN-preserving transforms.
- [Histogram gradient boosting: missing values](https://scikit-learn.org/stable/modules/ensemble.html#missing-values-support): learned routing and prediction behavior when training contained no NaNs.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html): training boundaries and preprocessing leakage.
- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html): the synthetic generator; this is not a public dataset source.

Stable documentation can change. The executed experiment records its installed
versions in [notes.md](notes.md#executed-experiment); behavior should be checked
against the installed library when reproducing it.
