# References

Primary sources and official documentation consulted for this implementation:

- Kohavi, Longbotham, Sommerfield and Henne (2009),
  [Controlled experiments on the web: survey and practical guide](https://ai.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf).
  Experiment design, randomization and practical constraints. Published online
  in 2008; journal volume 18 appeared in 2009.
- Fabijan et al. (2019),
  [Diagnosing Sample Ratio Mismatch in Online Controlled Experiments](https://www.microsoft.com/en-us/research/publication/diagnosing-sample-ratio-mismatch-in-online-controlled-experiments-a-taxonomy-and-rules-of-thumb-for-practitioners/).
  SRM as a symptom of trustworthiness problems and investigation guidance.
- Wasserstein and Lazar (2016),
  [The ASA Statement on p-Values: Context, Process, and Purpose](https://doi.org/10.1080/00031305.2016.1154108).
  Interpretation of p-values and the limits of threshold-only decisions.
- SciPy 1.17,
  [chi2_contingency](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.contingency.chi2_contingency.html),
  [binomtest](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.binomtest.html)
  and [normal distribution](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.norm.html).
  Numerical reference for uncorrected Pearson testing, allocation testing,
  exact single-proportion intervals for Monte Carlo rejection rates and normal
  probabilities/quantiles. Local patch versions are recorded in the notes.

The implementation uses no public dataset: all user IDs, assignments and
outcomes are generated locally. Neither online experimentation infrastructure
nor production data is supplied by these references.

Visual-phase implementation references:

- [Plotly standalone HTML export](https://plotly.com/python/interactive-html-export/):
  embedded JavaScript for offline figures, with precomputed scenario controls.
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html):
  GIF encoding without FFmpeg or ImageMagick.
- [SciPy Welch t-test](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.ttest_ind.html):
  independent reference for user-mean inference and its confidence interval.
