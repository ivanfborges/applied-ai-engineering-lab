# References

- Cortes, C. and Vapnik, V. (1995).
  [Support-vector networks](https://link.springer.com/article/10.1007/BF00994018),
  *Machine Learning*, 20, 273–297. Original soft-margin support-vector work;
  the publisher page provides the abstract, bibliographic details, and access
  options. The examples here do not reproduce the paper's OCR experiments.
- [scikit-learn SVM user guide](https://scikit-learn.org/stable/modules/svm.html):
  primal/dual formulation, support-vector attributes, kernels, multiclass
  strategies, and operational caveats.
- [SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):
  `C`, `gamma`, tolerance, probability estimation, and fitted attributes.
- [LinearSVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html):
  solver, default squared-hinge loss, and intercept treatment differences.
- [RBF SVM parameter example](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html):
  joint effects of `C` and `gamma`; further reading for an optional visual phase.
- [make_moons API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
  and [make_blobs API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html):
  the two synthetic generators used here. No external observational dataset
  is used or downloaded.

Links were consulted on 2026-10-07. The stable documentation can advance beyond
the installed version; [notes.md](notes.md#executed-experiments) records the
versions actually used for the measured runs.
