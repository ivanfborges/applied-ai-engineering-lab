# References

Primary sources consulted for this implementation on November 6, 2026:

- Liu, Ting and Zhou (2008), [Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf).
  Random partitioning, subsampling, leaf correction and path normalization.
- scikit-learn, [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
  Score orientation, offset, contamination, sample size and depth semantics.
- scikit-learn, [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html).
  Kernel, gamma, nu and decision-function contract.
- SciPy, [median_abs_deviation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html).
  Raw versus normal-consistent MAD; independent test oracle.
- Plotly, [interactive HTML export](https://plotly.com/python/interactive-html-export/)
  and [dropdown controls](https://plotly.com/python/dropdowns/).
  Embedded runtime and coordinated offline control updates for the visual lab.
- Matplotlib, [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
  GIF export without an external video encoder.

The example creates all data locally using NumPy; no public or private dataset
is downloaded. API documentation URLs are rolling references. Executed package
versions and the synthetic configuration are recorded in [notes.md](notes.md).
