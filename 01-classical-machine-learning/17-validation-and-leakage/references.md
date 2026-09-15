# References

- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
  Holdout roles, ordinary and stratified folds, grouped data and temporal
  evaluation. Used to check the splitter and pipeline design.
- [scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
  Leakage prevention and the supervised feature-selection failure mode.
  The noise experiment is independently implemented with an explicit paired
  fold comparison.
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
  Expanding windows, test_size, max_train_size, gap in rows and equally spaced
  samples. The toy label-delay assumption is defined by this study.

The code uses only in-memory synthetic data. No public dataset was downloaded.
Documentation pages may evolve; executed package versions are recorded in
[notes.md](notes.md), while supported dependencies remain centralized in the
[root configuration](../../pyproject.toml).
