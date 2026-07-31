# References

## Books

- Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong.
  *Mathematics for Machine Learning*. Cambridge University Press, 2020.
  [Official book site](https://mml-book.github.io/)
- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT
  Press, 2016. See Chapter 6 for feedforward networks and backpropagation and
  Chapter 8 for optimization.
  [Online edition](https://www.deeplearningbook.org/)

## Courses and Technical Notes

- Stanford CS231n.
  [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- Stanford CS231n.
  [Optimization](https://cs231n.github.io/optimization-1/)
- Terence Parr and Jeremy Howard.
  [The Matrix Calculus You Need for Deep Learning](https://explained.ai/matrix-calculus/)
- *Dive into Deep Learning*.
  [Calculus](https://d2l.ai/chapter_preliminaries/calculus.html)
- *Dive into Deep Learning*.
  [Automatic Differentiation](https://d2l.ai/chapter_preliminaries/autograd.html)

## Official Documentation

- NumPy.
  [`numpy.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html)
  — numerical gradients on sampled arrays. The examples in this topic instead
  use centered finite differences directly to make the approximation explicit.
- PyTorch.
  [Automatic differentiation package](https://pytorch.org/docs/stable/autograd.html)
- JAX.
  [Automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html)

## Further Reading

- Barak A. Pearlmutter. “Fast Exact Multiplication by the Hessian.” *Neural
  Computation*, 1994.
  [DOI](https://doi.org/10.1162/neco.1994.6.1.147)
- Sepp Hochreiter. “The Vanishing Gradient Problem During Learning Recurrent
  Neural Nets and Problem Solutions.” *International Journal of Uncertainty,
  Fuzziness and Knowledge-Based Systems*, 1998.
  [DOI](https://doi.org/10.1142/S0218488598000094)

## Data

No external dataset is used. `example.py` creates a small synthetic linear
dataset with a fixed random seed, and `from_scratch.py` uses fixed scalar
inputs.
