# References

Sources consulted for this topic:

- [scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
  OLS estimator, intercept behavior, coefficient attributes, and numerical
  least-squares implementation used by the example.
- [NumPy: numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
  Direct least-squares solving, rank, and minimum-norm solutions. Used as an
  independent numerical reference in the noiseless test.
- [NIST/SEMATECH: How can I tell if a model fits my data?](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm).
  Residual definitions and why aggregate fit statistics are insufficient.
- [MIT OpenCourseWare, 18.S096, Lecture 6: Regression Analysis](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/fc0c08db6b497cc3bd09020aec39f9b5_MIT18_S096F13_lecnote6.pdf).
  David Kempthorne, Fall 2013. OLS, projection, Gauss-Markov assumptions, and
  the distinction between covariance assumptions and normal-model inference.

The examples use only synthetic data generated in [example.py](example.py).
There is no external dataset or claim about measured production latency.
Stable documentation URLs can change with library releases; the executed
environment is recorded in [notes.md](notes.md).
