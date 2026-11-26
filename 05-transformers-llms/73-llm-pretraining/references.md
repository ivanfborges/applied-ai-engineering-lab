# References and reading boundaries

- Devlin et al. (2018), [BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding](https://arxiv.org/abs/1810.04805).
  Bidirectional representations and masked-token pretraining. This lab uses a
  simpler all-mask corruption policy and omits next-sentence prediction.
- Kaplan et al. (2020), [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
  Empirical loss relationships with model size, data and training compute;
  conclusions depend on the experimental regime.
- Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
  Fixed-compute allocation between parameters and training tokens. The synthetic
  runs in this folder neither reproduce its experiments nor fit a scaling law.
- PyTorch 2.15, [TransformerEncoderLayer](https://docs.pytorch.org/docs/2.15/generated/torch.nn.TransformerEncoderLayer.html)
  and [Transformer mask conventions](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html).
  Versioned APIs for the educational block and blocked-position Boolean masks.

The papers supply theoretical and empirical context, not datasets for this
implementation. Both examples generate their synthetic data locally. No public
corpus, pretrained checkpoint or external API is used by the executable study.

## Visual lab sources

- PyTorch 2.15, [MultiheadAttention](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html).
  `need_weights=True` and `average_attn_weights=False` expose the actual
  per-head weights from the measured model's snapshot forward pass.
- Plotly, [sliders](https://plotly.com/python/sliders/).
  Interactive checkpoint and parameter controls; the lab's HTMLs use a local
  JavaScript bundle and require no hosted service.
- Hoffmann et al., [Table 1 and the fixed-compute allocation study](https://arxiv.org/html/2203.15556v1).
  Historical Gopher/Chinchilla counts are sourced separately from the lab's
  adjustable illustrative scaling coefficients and synthetic measurements.
