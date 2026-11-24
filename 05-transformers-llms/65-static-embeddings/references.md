# Primary references and provenance

- Mikolov, Chen, Corrado and Dean (2013),
  [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
  CBOW, Skip-gram and word relationship evaluation; the implementation here
  intentionally uses a small full-softmax educational variant.
- Mikolov, Sutskever, Chen, Corrado and Dean (2013),
  [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
  Negative sampling, subsampling and phrase representations. The visual lab
  implements SGNS; subsampling and phrase learning remain background material.
- Pennington, Socher and Manning (2014),
  [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162/).
  Weighted log-count reconstruction. The batch optimizer and synthetic
  configuration here do not reproduce the paper's training procedure or results.
- Levy and Goldberg (2014),
  [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html).
  SGNS and shifted PMI; distinguish noise assumptions and independent optima
  from a finite-dimensional fit.

References checked on 2026-11-24. Paper results belong to their original
experiments and are not measurements made in this repository.

## Data and implementation scope

All example text is **synthetic**, constructed in [example.py](example.py).
It intentionally repeats selected contextual roles and includes two uses of
`bank`. The standalone [GloVe smoke](from_scratch.py) uses two invented
three-token sentences. There is no public dataset, external model or artifact
download. Both learners and gradients are educational NumPy implementations;
primary papers provide the mathematical sources, not copied implementation code.

The [dated record](notes.md#execution-record--2026-11-24) identifies execution
configuration, versions, results and pending interpretation. No visual assets
accompany the original analytical study. The separate
[visual laboratory](VISUAL_LAB.md) implements SGNS, a controlled expected-noise
association experiment, actual training snapshots and synthetic judged retrieval.
Its records do not replace the historical analytical observations. Primary
papers were consulted for this implementation; their benchmark scores are not
repository results. No production validation is claimed.
