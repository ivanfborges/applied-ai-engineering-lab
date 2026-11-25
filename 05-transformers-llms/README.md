# NLP, Transformers and LLMs

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 5 spans global Days 63–77. **8 of 15 studies are implemented.**
Author interpretation review remains pending; implementation is not approval.
Days 71–77 remain planned in the roadmap, including model families,
LLM tokenization, training and inference.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 63 | [Classical NLP](63-classical-nlp/) | Tested Unicode/count core; separate 13-group offline visual lab with real NLTK morphology, local BPE, TF-IDF/SVD geometry, Counter language models and judged synthetic retrieval/OCR experiments; author interpretation review required |
| 64 | [Bag of Words, TF-IDF and Text Similarity](64-bow-tfidf-text-similarity/) | Tested analytical TF-IDF/cosine; separate 18-section offline visual lab with true 3D vocabulary vectors, CSR bytes, BM25, query contributions, SVD, lexical RRF and judged synthetic retrieval; author interpretation review required |
| 65 | [Word2Vec, GloVe and Static Embeddings](65-static-embeddings/) | Tested full-softmax CBOW/Skip-gram and GloVe gradients; synthetic cosine neighborhoods and analogy failures/successes; author interpretation review required |
| 66 | [Attention Mechanism](66-attention-mechanism/) | Tested Q/K/V attention, row normalization and masks; CPU PyTorch SDPA agreement, synthetic causal/value controls and seeded scaling experiment; author interpretation review required |
| 67 | [Transformer Architecture I](67-transformer-architecture-i/) | Tested NumPy encoder-decoder composition, shifted targets and causal invariance; CPU PyTorch synthetic reversal with disjoint-sequence teacher-forced/autoregressive evaluation; author interpretation review required |
| 68 | [Transformer Architecture II: Multi-Head Attention](68-transformer-architecture-ii/) | Tested NumPy/PyTorch parity; separate offline visual lab with 3D projection/QKV controls, ten GIFs, maps/graphs, diversity interventions, theoretical KV/complexity and CPU online softmax; author interpretation review required |
| 69 | [Positional Encoding](69-positional-encoding/) | Tested positional identities/cache offsets; separate offline lab with 13 demonstrations, 12 GIFs, exact/PCA geometry, rotary pair contributions, ALiBi and measured float32/64 precision; author interpretation review required |
| 70 | [Feed-Forward Networks, Residual Connections and Layer Normalization](70-transformer-ffn-residual-norm/) | Tested FFN/LayerNorm forward and VJP; eleven-group offline lab with nine GIFs, twelve interactive HTMLs, actual forward/backward traces, normalization/gating controls and paired CPU depth diagnostics; no training; author interpretation review required |

Day 63 asks which textual differences a representation should preserve. Its
[notes and dated analytical record](63-classical-nlp/notes.md) distinguish exact
counting properties from retrieval-quality claims. Analytical morphology uses
mechanical suffix removal and a closed English word/POS fixture. The separate
[visual guide](63-classical-nlp/VISUAL_LAB.md) documents actual NLTK morphology,
explicit linguistic resources and new synthetic evaluation protocols. Neither
the original six-document example nor the visual lab establishes a real-data
benchmark or production system.

Day 64 asks how corpus-dependent weighting changes lexical similarity. Its
[dated analytical record](64-bow-tfidf-text-similarity/notes.md#execution-record--2026-11-24)
separates numerical agreement, observed synthetic scores and pending
interpretation candidates. Its separate
[visual guide](64-bow-tfidf-text-similarity/VISUAL_LAB.md) documents actual
BM25, lexical rank fusion, synthetic judgments and optional local search UI.
No real-data relevance benchmark, dense-model comparison or production search
engine is claimed.

Day 65 asks whether context fitting guarantees useful vector geometry. Its
[dated analytical record](65-static-embeddings/notes.md#execution-record--2026-11-24)
reports all three objectives, original-dimensional cosine search and analogy
rankings on 32 synthetic sentences. Low reconstruction loss and selected
positive similarity margins did not establish useful neighborhoods in every
case. Negative sampling remains theory only; no semantic benchmark or production
retriever was evaluated.

Day 66 asks how compatibility scores become a weighted value representation.
Its [dated analytical record](66-attention-mechanism/notes.md#execution-record--2026-11-24)
separates numerical equivalence and controlled synthetic changes from pending
interpretations. The NumPy reference supports rectangular cross-attention and
explicit visibility masks, rejecting fully blocked rows. No trained semantic
model, performance benchmark or production inference system was evaluated.

Day 67 asks how attention blocks compose into a conditional sequence model.
Its [dated analytical record](67-transformer-architecture-i/notes.md#execution-record--2026-11-25)
reports one fixed-budget CPU training run on synthetic reversal sequences,
separating teacher-forced predictions from greedy generation. The single-head
NumPy block retains parameters across calls but remains untrained. Tests cover
causal invariance, source sensitivity, target shifting and finite gradients.
No natural-language benchmark, training ablation or production system was evaluated;
interpretation candidates remain pending author review.

Day 68 asks what multiple routing distributions add at fixed model width.
Its [dated analytical record](68-transformer-architecture-ii/notes.md#execution-record--2026-11-25)
reports matched-weight NumPy/PyTorch forward agreement, a future-token
perturbation control and a head-width/scaling sweep on untrained synthetic
vectors. Tests include an independent per-head loop and identical-map/distinct-value
counterexample. GQA/MQA and KV cache calculations remain theory only; no model
quality, pruning or runtime benchmark was evaluated. Interpretation candidates
remain pending author review.

Its separate [visual guide](68-transformer-architecture-ii/VISUAL_LAB.md) documents
the seed-168 lab, actual intermediate tensors, direct/PCA projection spaces,
constructed routing advantage, causal-prefix controls and identical-map/distinct-value
interventions. Ten learning modules connect all fifteen visual questions through
offline HTML, GIFs and inspectable frames. MHA/GQA/MQA/cache comparisons are
theoretical; the online-softmax identity is computed in CPU NumPy and does not
reproduce a FlashAttention GPU kernel. Visual measurements and browser checks
remain distinct from the earlier analytical run and author interpretation review.

Day 69 asks how content-only attention acquires information about token order.
Its [dated analytical record](69-positional-encoding/notes.md#execution-record--2026-11-25)
separates equivariance, fixed-coordinate and fixed-causal-mask interventions.
NumPy checks cover sinusoidal offset identities, rotary norm/relative-dot-product
properties and supplied-slope ALiBi distributions. A one-layer cache comparison
detects inconsistent new-token position IDs. No training, long-context quality
benchmark or production cache runtime was evaluated; interpretation candidates
remain pending author review.

Its separate [visual guide](69-positional-encoding/VISUAL_LAB.md) documents
seed-269 synthetic controls, exact coordinates versus fixed PCA bases,
uncentered linear vector addition, multifrequency rotary contributions and
matched causal attention comparisons. Twelve GIFs complement offline Plotly
controls; the precision scene measures float32/64 phase construction rather
than trained-model extrapolation. The RAG scene constructs a synthetic prompt
and an evaluation design without running an LLM. Visual measurements remain
separate from the original analytical run and author interpretation review.

Day 70 asks how feature transformations, additive paths and norm placement
affect propagation through depth. Its
[dated analytical record](70-transformer-ffn-residual-norm/notes.md#execution-record--2026-11-25)
reports 24-block initialization diagnostics for three seeds and a separate
shared-final-norm control. NumPy forward/backward formulas agree with PyTorch;
block composition is tested against the official encoder layer. Random-target
MSE and gradient magnitudes do not rank model quality. No training,
latency benchmark or production system was evaluated; interpretation
candidates remain pending author review.

Its separate [visual guide](70-transformer-ffn-residual-norm/VISUAL_LAB.md)
documents seed-270/271 depth-6/12/24 CPU measurements, exact normalization
geometry and axes, signed SwiGLU products, theoretical budgets and actual
forward/backward tensors. Nine GIFs and twelve standalone offline HTMLs are
generated alongside PNGs and CSVs. These runs remain separate from the
analytical record; all exports and preview-selection decisions await human
curation, and interpretations remain pending author review.
