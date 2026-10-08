# References

Primary sources checked on 2026-10-07. Stable documentation can describe a
different installed release; the executed example records its actual versions.

- [Chawla, Bowyer, Hall, and Kegelmeyer (2002), SMOTE: Synthetic Minority Over-sampling Technique](https://www.jair.org/index.php/jair/article/view/10302).
  Original method; the paper's results are not benchmarks for this study.
- [imbalanced-learn: oversampling](https://imbalanced-learn.org/stable/over_sampling.html).
  Random duplication, interpolation, ill-posed geometry, and mixed-feature variants.
- [imbalanced-learn: common pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).
  Resampling leakage, natural evaluation distributions, and pipeline placement.
- [scikit-learn: compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html).
  Definition of the balanced class-frequency heuristic.
- [scikit-learn: tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html).
  Separating probability estimation from actions and keeping tuning data separate.
- [scikit-learn: average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
  Recall-weighted AP and its distinction from trapezoidal PR area.
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html).
  Reliability evaluation and the limits of interpreting Brier loss alone.
- [scikit-learn: make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html).
  Synthetic generator parameters; label noise affects observed class counts.

The scripts require the existing NumPy and scikit-learn dependencies.
imbalanced-learn is a reference for maintained sampler workflows and is not
required to run the educational implementation.
