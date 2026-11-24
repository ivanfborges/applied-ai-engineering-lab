# NLP, Transformers and LLMs

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 5 spans global Days 63–77. **4 of 15 studies are implemented.**
Author interpretation review remains pending; implementation is not approval.
Days 67–77 remain planned in the roadmap, including Transformer architecture,
multi-head attention, LLM tokenization, training and inference.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 63 | [Classical NLP](63-classical-nlp/) | Tested Unicode/count core; separate 13-group offline visual lab with real NLTK morphology, local BPE, TF-IDF/SVD geometry, Counter language models and judged synthetic retrieval/OCR experiments; author interpretation review required |
| 64 | [Bag of Words, TF-IDF and Text Similarity](64-bow-tfidf-text-similarity/) | Tested analytical TF-IDF/cosine; separate 18-section offline visual lab with true 3D vocabulary vectors, CSR bytes, BM25, query contributions, SVD, lexical RRF and judged synthetic retrieval; author interpretation review required |
| 65 | [Word2Vec, GloVe and Static Embeddings](65-static-embeddings/) | Tested full-softmax CBOW/Skip-gram and GloVe gradients; synthetic cosine neighborhoods and analogy failures/successes; author interpretation review required |
| 66 | [Attention Mechanism](66-attention-mechanism/) | Tested Q/K/V attention, row normalization and masks; CPU PyTorch SDPA agreement, synthetic causal/value controls and seeded scaling experiment; author interpretation review required |

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
