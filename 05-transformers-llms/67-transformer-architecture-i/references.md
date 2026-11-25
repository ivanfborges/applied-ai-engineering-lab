# Primary references

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
  Sections 3.1–3.5 describe the encoder/decoder stacks, attention, FFNs,
  embeddings and positions. This study does not reproduce its translation
  datasets, training recipe or reported scores.
- [PyTorch 2.15 — `nn.Transformer`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html).
  Version matched to the local execution: batch-first inputs, post-norm baseline,
  mask shapes and boolean `True` meaning blocked.
- [PyTorch 2.15 — scaled dot-product attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).
  Boolean `True` means allowed in this lower-level API. The study uses
  `nn.Transformer` rather than passing its mask unchanged to direct SDPA.

Primary sources checked on 2026-11-25. These links explain the architecture and
API contract; local synthetic results are recorded separately in [notes.md](notes.md).
No external dataset or pretrained model was used.

## Visual implementation contracts

- [PyTorch 2.15 — MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html):
  `need_weights=True` and `average_attn_weights=False` expose individual head
  probabilities. The visual instrumentation checks Q/K/V reconstruction against
  these actual module operations and the complete model forward.
- [Plotly — interactive HTML export](https://plotly.com/python/interactive-html-export/):
  standalone HTML can bundle the runtime for offline opening. This lab uses no CDN.
- [scikit-learn — PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html):
  fit/transform and explained variance; a single fitted transformation is retained
  across the stages and checkpoints compared inside each geometric artifact.
