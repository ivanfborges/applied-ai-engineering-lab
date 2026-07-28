# References

## Books and Course Material

- Gilbert Strang, *Introduction to Linear Algebra*, 6th edition, Wellesley-Cambridge Press, 2023. Chapters on vector spaces, orthogonality, least squares, and eigenvalues.
- Stephen Boyd and Lieven Vandenberghe, *Introduction to Applied Linear Algebra: Vectors, Matrices, and Least Squares*, Cambridge University Press, 2018. [Official book site and free PDF](https://web.stanford.edu/~boyd/vmls/).
- MIT OpenCourseWare, *18.06 Linear Algebra*. [Course materials](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, 4th edition, Johns Hopkins University Press, 2013. Reference for QR, SVD, least squares, and numerical stability.

## Official Documentation

- NumPy, [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html): stable least-squares interface used by `example.py`.
- NumPy, [`numpy.linalg.qr`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html): QR factorization used to construct an orthonormal basis.
- NumPy, [`numpy.linalg.matrix_rank`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html): numerical rank estimation based on singular values.
- scikit-learn, [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html): practical PCA behavior, centering, solvers, and attributes.
- scikit-learn, [Random Projection](https://scikit-learn.org/stable/modules/random_projection.html): practical discussion of Johnson-Lindenstrauss random projections.

## Further Reading

- William H. Press et al., *Numerical Recipes: The Art of Scientific Computing*, 3rd edition, Cambridge University Press, 2007. See the chapters on linear algebra and least squares.
- Sanjoy Dasgupta and Anupam Gupta, “An Elementary Proof of a Theorem of Johnson and Lindenstrauss,” *Random Structures & Algorithms*, 2003. [Paper](https://doi.org/10.1002/rsa.10073).

## Scope Note

The scripts use small synthetic arrays defined directly in code. No public dataset, embedding service, or benchmark result is used.
