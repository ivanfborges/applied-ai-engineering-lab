# Primary references

- Vaswani et al. (2017), [Attention Is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need).
  Sections 3.2.1–3.2.3: scaled dot-product attention, projections and masking;
  the scaling argument states assumptions about query/key components.
- PyTorch, [functional scaled dot-product attention, version 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).
  Boolean masks, tensor shapes, explicit dropout and backend-dependent numerical
  behavior. Consulted 2026-11-24; local comparison uses 2.15.0+cpu.
- Dao et al. (2022), [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
  Memory-hierarchy-aware exact attention. Background only: this study does not
  implement or benchmark FlashAttention.
- Jain and Wallace (2019), [Attention is not Explanation](https://aclanthology.org/N19-1357/).
  Empirical counterexamples to treating attention weights alone as faithful
  explanations. Its studied tasks are not a universal theorem about every
  possible explanatory method or model.

All example vectors and Gaussian draws are synthetic and defined in code.
No public dataset, pretrained model, text corpus or external service is used.

## Visual export documentation

- Plotly, [interactive HTML export](https://plotly.com/python/interactive-html-export/):
  standalone documents bundle the runtime when fully self-contained. The lab
  embeds the licensed local Plotly.js runtime and uses local JavaScript controls.
- Matplotlib, [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html):
  CPU GIF encoding through Pillow, used with `FuncAnimation`. No external
  video encoder or browser renderer is required for PNG/GIF generation.
