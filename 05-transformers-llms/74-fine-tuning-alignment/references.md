# References

Primary sources checked on 2026-11-25. These explain methods, not results
produced by this repository. The executable datasets are generated entirely
in code; no public dataset or pretrained model is used.

- Wei et al., [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652).
  Instruction tuning and evaluation on unseen task types; this toy does not reproduce FLAN.
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155).
  Demonstrations, reward modeling and policy optimization in InstructGPT.
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290).
  Reference-adjusted preference loss and its KL-regularized derivation;
  [full text](https://arxiv.org/html/2305.18290v3), Sections 3–4 and Appendix A.
- Houlsby et al., [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751).
  Bottleneck adapter modules with shared pretrained parameters.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
  Low-rank update parameterization, initialization and compatible weight merging.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
  Quantized frozen bases, NF4, double quantization and paged optimizers;
  quantization is theory only here.
- PyTorch 2.15 documentation: [cross-entropy](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cross_entropy.html),
  [log-sigmoid](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.logsigmoid.html)
  and [Linear](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html).
  The numerical environment is recorded separately in [notes.md](notes.md).

Visual implementation references (checked on 2026-11-25):

- Plotly, [interactive HTML export](https://plotly.com/python/interactive-html-export/).
  Offline interactive export and local JavaScript packaging; this lab shares
  one adjacent runtime across its generated pages.
- Matplotlib, [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
  GIF export through Pillow without FFmpeg.
- NumPy, [singular value decomposition](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html).
  Matrix factorization used for truncated reconstruction and pooled PCA.
  Runtime versions are preserved with each [visual run](VISUAL_LAB.md).
