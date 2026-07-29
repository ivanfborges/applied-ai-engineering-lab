# References

## Books

- Gilbert Strang. *Introduction to Linear Algebra*, 6th ed. Wellesley-Cambridge
  Press, 2023. Chapters on eigenvalues, symmetric matrices, positive-definite
  matrices, and SVD.
- Gene H. Golub and Charles F. Van Loan. *Matrix Computations*, 4th ed. Johns
  Hopkins University Press, 2013. Numerical methods for eigenvalue problems,
  least squares, and SVD.
- Christopher M. Bishop. *Pattern Recognition and Machine Learning*. Springer,
  2006. Section 12.1 covers PCA and related latent-variable interpretations.
- Ian T. Jolliffe and Jorge Cadima. “Principal component analysis: a review and
  recent developments.” *Philosophical Transactions of the Royal Society A*,
  2016. https://doi.org/10.1098/rsta.2015.0202

## Papers and Tutorials

- Karl Pearson. “On Lines and Planes of Closest Fit to Systems of Points in
  Space.” *Philosophical Magazine*, 1901.
  https://doi.org/10.1080/14786440109462720
- Harold Hotelling. “Analysis of a Complex of Statistical Variables into
  Principal Components.” *Journal of Educational Psychology*, 1933.
  https://doi.org/10.1037/h0071325
- Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp. “Finding Structure
  with Randomness: Probabilistic Algorithms for Constructing Approximate
  Matrix Decompositions.” *SIAM Review*, 2011.
  https://doi.org/10.1137/090771806
- Jonathon Shlens. “A Tutorial on Principal Component Analysis,” 2014.
  https://arxiv.org/abs/1404.1100

## Course Material

- Carnegie Mellon University, 10-601 Machine Learning: dimensionality
  reduction, PCA, and SVD lecture material.
  https://www.cs.cmu.edu/~tom/10601_sp09/lecture.html
- Stanford University, CS229 Linear Algebra Review and machine-learning course
  notes. https://cs229.stanford.edu/section/cs229-linalg.pdf

## Official Documentation

- scikit-learn: `PCA`.
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- scikit-learn: `TruncatedSVD`.
  https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
- scikit-learn: choosing the right estimator for decomposition and
  dimensionality reduction.
  https://scikit-learn.org/stable/modules/decomposition.html
- NumPy: `numpy.linalg.eigh`.
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html
- NumPy: `numpy.linalg.svd`.
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html

## Notes on the Examples

The scripts use synthetic data generated locally with NumPy. No public dataset
or reported benchmark result is used.
