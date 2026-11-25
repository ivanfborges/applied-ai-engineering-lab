# Primary references

Consulted on 2026-11-25. Paper results are external evidence; they are not
experiments reproduced in this topic.

- Devlin et al. (2019), [BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding](https://aclanthology.org/N19-1423/).
  Sections 3.1–3.2 distinguish MLM/NSP from task-specific fine-tuning.
- Radford et al. (2018), [Improving Language Understanding by Generative
  Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
  Causal language-model pretraining and discriminative adaptation.
- Raffel et al. (2020), [Exploring the Limits of Transfer Learning with a Unified
  Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html).
  Sections 2.1 and 3.1.4 cover architecture and sentinel-based corruption;
  Section 3.2 compares architecture variants.
- PyTorch 2.15, [TransformerEncoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html)
  and [TransformerDecoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerDecoderLayer.html).
  Building-block interfaces, target masks and encoder-memory masks.
- PyTorch 2.15, [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html).
  Scaling, Boolean visibility and explicit zero-dropout evaluation semantics.
- Matplotlib 3.11.2, [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).
  Local GIF export without FFmpeg; the visual laboratory uses computed frames.
- Plotly, [Interactive HTML export](https://plotly.com/python/interactive-html-export/)
  and [3D scatter plots](https://plotly.com/python/3d-scatter-plots/).
  Embedded JavaScript supports standalone offline inspection of numerical data.

Related local studies: [attention](../66-attention-mechanism/),
[encoder-decoder training](../67-transformer-architecture-i/),
[multi-head attention](../68-transformer-architecture-ii/),
[positions](../69-positional-encoding/) and
[FFN, residuals and normalization](../70-transformer-ffn-residual-norm/).
