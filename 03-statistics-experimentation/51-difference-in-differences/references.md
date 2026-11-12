# References

- [Callaway and Sant'Anna: Introduction to DiD with Multiple Time Periods](https://bcallaway11.github.io/did/articles/multi-period-did.html).
  Authors' explanation of canonical ATT, parallel trends, group-time effects
  and why already-treated comparison units can contaminate conventional TWFE.
  The linked R package is further reading; it is not an implementation dependency.
- [Roth (2022), Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends](https://www.aeaweb.org/articles?id=10.1257/aeri.20210236).
  Primary research on low power of pre-trends tests and distortions from
  conditioning analysis on a successful pretest.
- [statsmodels: RegressionResults.get_robustcov_results](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).
  Official reference for clustered covariance, finite-sample correction,
  cluster degrees of freedom and explicit Student-t inference.
- [Plotly: Intro to Animations](https://plotly.com/python/animations/).
  Official reference for graph-object frames and play controls. The geometric
  derivation keeps axis ranges fixed across frames to avoid rescaling artifacts.

Sources consulted on 2026-11-12. All study data are generated locally in
[example.py](example.py); no external dataset is downloaded or reused.
