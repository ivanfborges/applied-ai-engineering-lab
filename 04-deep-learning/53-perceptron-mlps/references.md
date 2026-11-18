# References

- Leshno, Lin, Pinkus and Schocken (1993),
  [Multilayer Feedforward Networks With a Nonpolynomial Activation Function Can Approximate Any Function](https://pinkus.net.technion.ac.il/files/2021/02/neural.pdf).
  Primary approximation result; supports the distinction between a sufficiently
  wide representation with biases and a successful learning procedure. The
  continuous nonpolynomial case stated in the notes is a restricted consequence
  of the paper's broader result, rather than a theorem about a fixed width.
- [scikit-learn MLPClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
  Activation, solver and convergence settings used by `example.py`.
- [scikit-learn supervised neural networks guide](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).
  Library context for feedforward classification and optimization.
- Hendrycks and Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415).
  Original GELU definition used in the activation visualization; the lab uses
  the exact Gaussian-CDF form, not the common tanh approximation.
- [Plotly standalone HTML export](https://plotly.com/python/interactive-html-export/).
  Embedded runtime export for the offline interactive pages.
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
  GIF export without a separate animation executable.

Sources checked on 2026-11-18. The executed environment is recorded separately
in [notes.md](notes.md); documentation URLs may describe a different release
in the future. All study data are synthetic and generated in code. No public
dataset download is needed.
