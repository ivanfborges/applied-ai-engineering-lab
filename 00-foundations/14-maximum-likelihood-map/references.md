# References

## Statistical estimation and machine learning

- Goodfellow, I., Bengio, Y., & Courville, A. (2016).
  [*Deep Learning*, Chapter 5: Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html).
  Sections 5.5 and 5.6 connect maximum likelihood, conditional log-likelihood,
  and Bayesian point estimation.
- Murphy, K. P. (2022).
  [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html).
  A modern treatment of likelihood-based estimation, conjugate models, priors,
  and posterior inference.
- Stanford University CS229.
  [Probability theory review](https://cs229.stanford.edu/section/cs229-prob.pdf).
  Concise review of MLE, MAP, and common probabilistic models.

## Numerical implementation

- NumPy.
  [`numpy.logaddexp`](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html).
  The implementation uses this identity to evaluate logistic NLL without
  explicitly taking logarithms of rounded probabilities.
- NumPy.
  [Floating-point error handling](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html).
  Context management for expected logarithmic behavior at Bernoulli boundaries.

## Visual implementation

- Matplotlib.
  [Animations using Matplotlib](https://matplotlib.org/stable/users/explain/animations/animations.html).
  `FuncAnimation` and Pillow-backed GIF generation used by the accumulation
  and optimization-path demonstrations.
- Matplotlib.
  [3D surface example](https://matplotlib.org/stable/gallery/mplot3d/surface3d.html).
  Reference for the lightweight offline logistic-objective surface.

## Optimization nuance

- Loshchilov, I., & Hutter, F. (2019).
  [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
  The original AdamW paper distinguishes L2 penalties from decoupled weight
  decay for adaptive optimizers.

## Scope

These references support the definitions and implementation choices. They do
not validate the synthetic examples as evidence about any production model,
user population, or deployment policy.
