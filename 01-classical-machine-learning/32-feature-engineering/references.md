# References

Official scikit-learn documentation consulted on 2026-10-07:

- [Preprocessing user guide](https://scikit-learn.org/stable/modules/preprocessing.html):
  scaling, categorical representations, discretization, and polynomial terms.
- [KBinsDiscretizer API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html):
  quantile methods, subsampling, narrow-bin removal, and encoding options.
- [TargetEncoder API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html):
  smoothing and internal cross-fitting; the important distinction between
  `fit_transform` and `fit(...).transform(...)`.
- [Common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html):
  train-only fitting and pipelines during validation.

The scripts use the repository's common NumPy, pandas, and scikit-learn
dependencies. Compatibility handling keeps linear quantiles explicit on
releases exposing `quantile_method`; older supported releases use their linear
quantile behavior. Execution was verified with scikit-learn 1.9.0, not every
release allowed by the shared dependency bounds.

## Data provenance

All data are generated locally in [example.py](example.py) from a deterministic
NumPy generator with seed 42. There is no public or private business dataset and
no network download. Customer attributes, events, event availability delays,
and binary outcomes are artificial. The outcome generator intentionally
contains historical and nonlinear signal.

The small fixed arrays in [from_scratch.py](from_scratch.py) are code-defined
arithmetic examples. Neither data source supports a real-world benchmark claim.
