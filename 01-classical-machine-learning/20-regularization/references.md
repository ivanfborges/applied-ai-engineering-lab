# References

- [scikit-learn: Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
  — native loss normalization, intercept handling, and available solvers.
- [scikit-learn: ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
  — exact alpha/l1_ratio convention and optimization diagnostics.
- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
  — preprocessing boundaries, pipelines, cross-validation, and randomness.

These official pages were checked during implementation. Their stable URLs
can advance to newer releases; the executed examples used scikit-learn 1.9.0.
The algebra in [notes.md](notes.md) states its own normalization explicitly,
and [the tests](tests/test_regularization.py) check it against analytic
solutions and the installed library.
