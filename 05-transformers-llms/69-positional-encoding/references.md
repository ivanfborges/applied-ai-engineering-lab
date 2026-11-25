# References

Primary sources checked on 2026-11-25. Published paper results belong to their
authors and are not reproduced benchmarks in this study.

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
  Section 3.5 defines additive sinusoidal positions and discusses fixed offsets.
- Shaw, Uszkoreit and Vaswani (2018),
  [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155).
  Relative key/value representations; broader than a scalar relative-logit bias.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).
  Section 3 derives rotary query/key interactions. The paper's decay discussion
  should not be read as monotonic decay of every individual dot product.
- Press, Smith and Lewis,
  [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409).
  Fixed head-specific distance biases, slope choices and task-bounded extrapolation
  experiments. This study supplies its own illustrative slopes and does not train
  the paper's language models.
- [PyTorch 2.15 MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html).
  API used for the CPU control: `batch_first=True`, zero dropout, evaluation mode
  and Boolean masks where `True` blocks attention.

Earlier repository studies: [attention](../66-attention-mechanism/),
[encoder-decoder composition](../67-transformer-architecture-i/) and
[multi-head attention](../68-transformer-architecture-ii/).
The present numerical helpers are educational NumPy implementations rather than
copied framework or paper training code.

Visual implementation references, checked on 2026-11-25:

- [Plotly JavaScript function reference](https://plotly.com/javascript/plotlyjs-function-reference/)
  for live plot updates; [3D scatter plots](https://plotly.com/python/3d-scatter-plots/)
  for inspectable spatial coordinates.
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  for fixed fitted bases and explained variance. Vector addition uses its basis
  as an uncentered linear map, explicitly distinct from the affine transform.
- [Matplotlib PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html)
  for actual GIF export through the existing Pillow dependency.
