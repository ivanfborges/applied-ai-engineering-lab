# References

Primary sources consulted for this study on 2026-11-25:

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/html/1706.03762v7),
  sections 3.1 and 3.3: original residual/Post-LN arrangement and shared
  position-wise FFN. The original ReLU block differs from this lab's GELU choice.
- Ba, Kiros and Hinton (2016), [Layer Normalization](https://arxiv.org/abs/1607.06450):
  per-example statistics and learned affine transformation.
- He et al. (2015), [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385):
  residual learning and optimization motivation; its image experiments are
  separate from the Transformer diagnostics here.
- Xiong et al. (2020), [On Layer Normalization in the Transformer Architecture](https://proceedings.mlr.press/v119/xiong20b.html):
  initialization gradients, normalization placement and warmup. This study
  does not reproduce their theoretical assumptions or training results.
- Zhang and Sennrich (2019), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467):
  normalization without centering; implemented in the separate visual lab.
- Shazeer (2020), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202):
  gated feature transformations; SwiGLU is implemented in the separate visual lab.
- PyTorch 2.15: [`LayerNorm`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html),
  [`MultiheadAttention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)
  and [`GELU`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GELU.html).
  API behavior was checked against the installed 2.15.0+cpu environment;
  dependency ranges remain centralized in the root project configuration.
- PyTorch 2.15: [`BatchNorm1d`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html),
  channel/sequence layout, population training variance and unbiased running
  variance updates. Visual comparisons use explicitly calibrated frozen moments.
- Matplotlib 3.11.2: [`PillowWriter`](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html),
  GIF export without external encoders.
- Plotly: [Interactive HTML export](https://plotly.com/python/interactive-html-export/),
  embedded runtime for self-contained offline figures. This lab additionally
  embeds controls and deterministic numerical fixtures.

No public dataset was downloaded or used. All numerical inputs and targets
are generated synthetic arrays/tensors. Paper results are attributed external
evidence, not measurements made by this repository.
