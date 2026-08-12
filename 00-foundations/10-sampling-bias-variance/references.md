# References

## Sampling and estimation

- NIST/SEMATECH, [Populations and
  Sampling](https://www.itl.nist.gov/div898/handbook/ppc/section1/ppc134.htm) —
  target populations, sample adequacy, representativeness, variability, and
  precision.
- NIST/SEMATECH, [Confidence Limits and the Finite Population
  Correction](https://www.itl.nist.gov/div898/handbook/prc/section2/old.prc271.htm) —
  why sampling without replacement from a finite population changes
  uncertainty.
- U.S. Census Bureau, [Current Population Survey
  Weighting](https://www.census.gov/programs-surveys/cps/technical-documentation/methodology/weighting.html) —
  inverse selection probabilities and adjustments for non-response.
- U.S. Census Bureau, [Producing Summary
  Statistics](https://www.census.gov/programs-surveys/cps/technical-documentation/methodology/producing-summary-statistics.html) —
  why population estimates from a survey require the supplied sample weights.

## Implementation and validation design

- NumPy, [`Generator.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html) —
  random sampling with replacement controls and non-uniform probabilities.
- NumPy, [`numpy.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html) —
  variance denominators and `ddof` conventions.
- scikit-learn, [Cross-validation: evaluating estimator
  performance](https://scikit-learn.org/stable/modules/cross_validation.html) —
  IID assumptions and group- and time-aware validation strategies.
- scikit-learn,
  [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) —
  non-overlapping group partitions used by the identity-leakage experiment.

## Visual and interactive implementation

- Matplotlib, [Animation
  API](https://matplotlib.org/stable/api/animation_api.html) and
  [`PillowWriter`](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html) —
  bounded GIF generation without ImageMagick.
- Plotly, [Interactive HTML
  export](https://plotly.com/python/interactive-html-export/) —
  standalone offline visualizations and the embedded-JavaScript file-size
  trade-off.

## Further reading

- William G. Cochran, *Sampling Techniques*, third edition, Wiley, 1977.
- Sharon L. Lohr, *Sampling: Design and Analysis*, third edition, Chapman and
  Hall/CRC, 2021.
- Leslie Kish, *Survey Sampling*, Wiley, 1965.

The executable topic uses only NumPy from the repository's shared runtime.
