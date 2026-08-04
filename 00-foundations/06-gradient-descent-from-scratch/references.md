# References

## Core Theory

1. Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*.
   [Book site and downloadable text](https://web.stanford.edu/~boyd/cvxbook/).
   Chapters 9 and 11 provide broader context for descent and unconstrained
   optimization methods.

2. Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd edition.
   [Springer book page](https://link.springer.com/book/10.1007/b98874).
   A rigorous reference for line search, convergence, first-order, and
   second-order optimization.

3. Stanford CS229, *Supervised Learning: Linear Regression and Gradient
   Descent*.
   [Lecture notes (PDF)](https://cs229.stanford.edu/notes_archive/cs229-notes1.pdf).
   Derives batch and stochastic gradient descent for linear regression.

4. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, *Deep Learning*,
   Chapter 8: Optimization for Training Deep Models.
   [Online chapter](https://www.deeplearningbook.org/contents/optimization.html).
   Connects first-order optimization to large-scale neural network training.

## Library Documentation

5. NumPy documentation, [Array objects](https://numpy.org/doc/stable/reference/arrays.html).
   Reference for the vectorized array operations used by the from-scratch
   implementation.

6. scikit-learn documentation,
   [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
   Documents the ordinary least-squares estimator used as a comparison.

7. Matplotlib documentation,
   [`pyplot` examples](https://matplotlib.org/stable/gallery/pyplots/index.html).
   Reference for the saved regression and convergence figures.

## Data

No external dataset is used. `example.py` creates synthetic linear observations
locally with NumPy and a fixed random seed.
