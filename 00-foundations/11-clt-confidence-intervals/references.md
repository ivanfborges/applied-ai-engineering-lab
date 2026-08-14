# References

## Theory and statistical practice

- George Casella and Roger L. Berger, *Statistical Inference*, second edition,
  Duxbury, 2002 — sampling distributions, asymptotic inference, and interval
  estimation.
- Larry Wasserman, *All of Statistics*, Springer, 2004 — the Central Limit
  Theorem, the delta method, bootstrap, and confidence intervals.
- Edwin B. Wilson, "Probable Inference, the Law of Succession, and Statistical
  Inference," *Journal of the American Statistical Association*, 1927 — the
  score interval implemented for binomial proportions in this topic.
- NIST/SEMATECH, [Confidence Limits for the
  Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm) —
  normal and Student-\(t\) interval construction for a population mean.

## Implementation documentation

- Python standard library,
  [`statistics.NormalDist`](https://docs.python.org/3/library/statistics.html#statistics.NormalDist) —
  inverse normal CDF used by the first-principles intervals.
- NumPy,
  [`Generator.exponential`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.exponential.html) —
  synthetic exponential samples used by the CLT and coverage experiments.
- SciPy, [`scipy.stats.t`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html) —
  Student-\(t\) critical values used in the practical example.
- Matplotlib, [Animations using
  Matplotlib](https://matplotlib.org/stable/users/explain/animations/animations.html) and
  [`PillowWriter`](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html) —
  bounded offline GIF generation for CLT convergence and confidence coverage.
- Plotly, [`Surface`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Surface.html) and
  [interactive HTML export](https://plotly.com/python/interactive-html-export/) —
  the rotatable standard-error surface and its self-contained local export.

The executable study uses only deterministic synthetic data and the
repository's shared NumPy and SciPy dependencies. It does not use a public
dataset.
