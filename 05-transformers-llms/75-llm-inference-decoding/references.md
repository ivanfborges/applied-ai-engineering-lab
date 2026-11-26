# References

Primary sources consulted on 2026-11-26; documentation pages can change.

- Holtzman, Buys, Du, Forbes and Choi,
  [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751),
  ICLR 2020. Motivation for nucleus sampling and the distinction between
  likelihood and useful open-ended generation. Its model experiments are not
  reproduced by the five-logit synthetic controls here.
- Hugging Face,
  [Transformers generation configuration](https://huggingface.co/docs/transformers/main_classes/text_generation).
  Generation length, sampling, EOS and stop-string controls. Names/defaults
  belong to that library; this study defines an independent NumPy contract
  without importing Transformers or claiming library parity.
- NumPy,
  [Random stream compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html).
  Conditions on generator, calls, build and environment for reproducibility.
- PyTorch,
  [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
  Sources of nondeterminism and limits across releases and platforms. PyTorch
  is background reading, not a dependency of this topic.

All probabilities, fragments and transition models in the code are synthetic
and defined locally. No public dataset, pretrained model or external inference
service is used. Repository dependencies are declared in
[pyproject.toml](../../pyproject.toml).

Visual-laboratory references, additionally consulted on 2026-11-26:

- Hugging Face, [How caching works](https://huggingface.co/docs/transformers/main/en/cache_explanation).
  Reuse of past keys and values while new queries attend over the growing
  context. The local operation/storage accounting is explicitly theoretical.
- Plotly, [3D surface plots](https://plotly.com/python/3d-surface-plots/).
  Parameter-space exploration. This lab uses independent flat Mesh3d tiles
  rather than smoothly interpolating discrete candidate changes.
- Streamlit's installed **1.67.0** development references were used for local
  controls, session state, AppTest and the running-app agent API. The code uses
  the repository's declared Streamlit range; a different installation can have
  different testing and accessibility APIs.
