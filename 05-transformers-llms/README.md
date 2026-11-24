# NLP, Transformers and LLMs

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 5 spans global Days 63–77. **1 of 15 studies is implemented.**
Author interpretation review remains pending; implementation is not approval.
Days 64–77 remain planned in the roadmap, including weighting and similarity,
embeddings, attention, Transformers, LLM tokenization, training and inference.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 63 | [Classical NLP](63-classical-nlp/) | Tested Unicode/count core; separate 13-group offline visual lab with real NLTK morphology, local BPE, TF-IDF/SVD geometry, Counter language models and judged synthetic retrieval/OCR experiments; author interpretation review required |

Day 63 asks which textual differences a representation should preserve. Its
[notes and dated analytical record](63-classical-nlp/notes.md) distinguish exact
counting properties from retrieval-quality claims. Analytical morphology uses
mechanical suffix removal and a closed English word/POS fixture. The separate
[visual guide](63-classical-nlp/VISUAL_LAB.md) documents actual NLTK morphology,
explicit linguistic resources and new synthetic evaluation protocols. Neither
the original six-document example nor the visual lab establishes a real-data
benchmark or production system.
