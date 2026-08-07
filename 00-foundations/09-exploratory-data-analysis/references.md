# References

## Exploratory analysis and statistical definitions

- NIST/SEMATECH, [What is Exploratory Data
  Analysis?](https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm) —
  goals and philosophy of EDA.
- NIST/SEMATECH, [Measures of
  Scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm) —
  standard deviation and robust alternatives including IQR and MAD.
- NIST/SEMATECH, [Measures of Skewness and
  Kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm) —
  definitions, estimator conventions, and tail interpretation.

## Library conventions

- NumPy, [`numpy.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html)
  — denominator and `ddof` behavior.
- NumPy, [`numpy.quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
  — quantile methods and interpolation conventions.
- pandas, [`Series.skew`](https://pandas.pydata.org/docs/reference/api/pandas.Series.skew.html)
  and [`Series.kurt`](https://pandas.pydata.org/docs/reference/api/pandas.Series.kurt.html)
  — sample skewness and unbiased excess-kurtosis conventions used by the
  practical example.
- pandas, [`DataFrame.corr`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
  — Pearson, Kendall, and Spearman correlation interfaces.
- SciPy, [`pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)
  and [`spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
  — definitions, inference interfaces, and warnings for constant inputs.

## Visualization and offline interaction

- Matplotlib, [Animation
  API](https://matplotlib.org/stable/api/animation_api.html) — `FuncAnimation`
  and animation-writer concepts used by the bounded GIF generators.
- Matplotlib,
  [`PillowWriter`](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)
  — Pillow-backed GIF export without an external video encoder.
- Plotly, [Interactive HTML
  export](https://plotly.com/python/interactive-html-export/) — standalone
  browser artifacts and the trade-off between embedded JavaScript and file
  size.

## Books

- John W. Tukey, *Exploratory Data Analysis*, Addison-Wesley, 1977.
- Peter J. Huber and Elvezio M. Ronchetti, *Robust Statistics*, second
  edition, Wiley, 2009.

The Python examples in this topic require only the repository's existing NumPy
and pandas dependencies; SciPy is listed for further reading, not imported.
