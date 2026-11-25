# Primary references

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762),
  Section 3.2: projected heads, scaled dot products, concatenation and output
  projection. The implementation uses this mechanism without recreating a
  complete Transformer or reproducing the paper's experiments.
- [PyTorch 2.15 MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html):
  versioned API guidance for the installed reference version, per-head output
  maps and boolean blocked masks. `need_weights=False` is relevant when inspecting
  maps is unnecessary; no optimized-path speed claim is made in this study.
- [PyTorch 2.15 functional scaled_dot_product_attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html):
  comparison of mask conventions; its boolean `True` means permitted, unlike
  `nn.MultiheadAttention`. This day's reference uses the module, not SDPA directly.
- Ainslie et al. (2023), [GQA: Training Generalized Multi-Query Transformer Models
  from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): shared KV-head
  design and conversion/training context. Discussed only; not reproduced here.
- Dao et al. (2022), [FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness](https://arxiv.org/abs/2205.14135): dense attention execution
  and memory traffic. The visual lab executes an educational CPU online-softmax
  identity; it does not reproduce a GPU kernel or the paper's benchmarks.
- [Plotly JavaScript function reference](https://plotly.com/javascript/plotlyjs-function-reference/)
  and [3D scatter plots](https://plotly.com/javascript/3d-scatter-plots/):
  offline interactive rendering, reactive updates and rotatable coordinate views.
  The exported lab bundles the runtime instead of fetching it from a CDN.

Architecture references and the versioned MHA documentation were consulted on
2026-11-25. Runtime versions are recorded separately with executed results.
No external dataset is used; all numerical inputs are generated locally.
