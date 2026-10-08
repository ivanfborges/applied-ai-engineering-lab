# References

- Geurts, P., Ernst, D., and Wehenkel, L. (2006).
  [Extremely randomized trees](https://orbi.uliege.be/handle/2268/9357).
  *Machine Learning*, 63, 3-42. Original research; the institutional record
  provides an author preprint.
- [scikit-learn ensemble guide: extremely randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees).
  Split randomization and the bias/variance motivation.
- [ExtraTreesClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).
  Bootstrap, OOB, feature candidates, and class-probability aggregation.
- [RandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
  Greedy split strategy and configurable row sampling.
- [make_classification API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).
  Source of the synthetic generator, including redundant features and label
  reassignment. No public dataset was downloaded.
- [Probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).
  Reliability and the distinction between probability scores and calibration.

Links checked on 2026-10-06. The executed example used scikit-learn 1.9.0;
the moving stable documentation showed 1.9.1 at review time. Explicit
experimental settings and environment are recorded in [notes.md](notes.md).
