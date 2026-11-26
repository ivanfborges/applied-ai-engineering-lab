# NLP, Transformers and LLMs

[Repository overview](../README.md) · [Full roadmap](../ROADMAP.md)

Module 5 spans global Days 63–77. **15 of 15 studies are implemented.**
Author interpretation review remains pending; implementation is not approval.
The following RAG module remains planned in the roadmap.

| Day | Implemented study | Evidence and review status |
|---:|---|---|
| 63 | [Classical NLP](63-classical-nlp/) | Tested Unicode/count core; separate 13-group offline visual lab with real NLTK morphology, local BPE, TF-IDF/SVD geometry, Counter language models and judged synthetic retrieval/OCR experiments; author interpretation review required |
| 64 | [Bag of Words, TF-IDF and Text Similarity](64-bow-tfidf-text-similarity/) | Tested analytical TF-IDF/cosine; separate 18-section offline visual lab with true 3D vocabulary vectors, CSR bytes, BM25, query contributions, SVD, lexical RRF and judged synthetic retrieval; author interpretation review required |
| 65 | [Word2Vec, GloVe and Static Embeddings](65-static-embeddings/) | Tested full-softmax CBOW/Skip-gram and GloVe gradients; separate visual lab with actual SGNS training, synthetic cosine neighborhoods and analogy failures/successes; author interpretation review required |
| 66 | [Attention Mechanism](66-attention-mechanism/) | Tested Q/K/V attention, row normalization and masks; CPU PyTorch SDPA agreement, synthetic causal/value controls and seeded scaling experiment; author interpretation review required |
| 67 | [Transformer Architecture I](67-transformer-architecture-i/) | Tested NumPy encoder-decoder composition, shifted targets and causal invariance; CPU PyTorch synthetic reversal with disjoint-sequence teacher-forced/autoregressive evaluation; author interpretation review required |
| 68 | [Transformer Architecture II: Multi-Head Attention](68-transformer-architecture-ii/) | Tested NumPy/PyTorch parity; separate offline visual lab with 3D projection/QKV controls, ten GIFs, maps/graphs, diversity interventions, theoretical KV/complexity and CPU online softmax; author interpretation review required |
| 69 | [Positional Encoding](69-positional-encoding/) | Tested positional identities/cache offsets; separate offline lab with 13 demonstrations, 12 GIFs, exact/PCA geometry, rotary pair contributions, ALiBi and measured float32/64 precision; author interpretation review required |
| 70 | [Feed-Forward Networks, Residual Connections and Layer Normalization](70-transformer-ffn-residual-norm/) | Tested FFN/LayerNorm forward and VJP; eleven-group offline lab with nine GIFs, twelve interactive HTMLs, actual forward/backward traces, normalization/gating controls and paired CPU depth diagnostics; no training; author interpretation review required |
| 71 | [Encoder-only, Decoder-only and Encoder-Decoder Models](71-encoder-decoder-models/) | Tested attention and target shifts; paired causal/bidirectional CPU blocks, source-conditioned targets and padding isolation on synthetic vectors; no training or language benchmark; author interpretation review required |
| 72 | [Tokenization for LLMs](72-tokenization/) | Tested character BPE ranks, real byte-BPE IDs/Unicode reconstruction and hypothetical context/cost budgets; eight synthetic probes, no model quality or provider billing evaluation; author interpretation review required |
| 73 | [LLM Pretraining](73-llm-pretraining/) | Tested CLM/MLM and explicit NumPy gradients; separate 31-artifact offline lab with actual probability/embedding/attention evolution, ten synthetic training scenarios, prefix-only leakage controls and analytical scale/compute; author interpretation review required |
| 74 | [Fine-Tuning, Instruction Tuning and Alignment](74-fine-tuning-alignment/) | Tested DPO derivatives, response masks and LoRA freezing/merge; separate 14-demo CPU visual lab with 13 GIFs, 21 PNGs and 14 offline HTMLs: reward/KL, matrix ranks, quantization simulation, forgetting and proxy failures; author interpretation review required |
| 75 | [LLM Inference and Decoding](75-llm-inference-decoding/) | Tested NumPy temperature/top-k/top-p, stable ties and generation stops; separate offline lab with synthetic probability/sampling controls, beam enumeration and theoretical cache accounting; no real-model quality or serving benchmark; author interpretation review required |
| 76 | [Prompt Engineering with Rigor](76-prompt-engineering/) | Three inspectable prompts and tested reference scoring/retries on authored responses; separate synthetic visual lab, offline replay and opt-in provider adapter; no executed LLM comparison claimed; author interpretation review required |
| 77 | [Structured Outputs](77-structured-outputs/) | Tested contracts/gates; separate 13-demo CPU lab with 6 GIFs, 4 PNGs, 8 offline HTMLs: toy grammar, masks, actual 108-record funnel, strictness and 100k-request retry simulations; no provider evaluation; author interpretation review required |

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
case. Negative sampling is not implemented in those analytical scripts. The
separate [visual lab](65-static-embeddings/VISUAL_LAB.md) implements SGNS on a
larger synthetic corpus and records its own configurations and results. Neither
phase establishes a semantic benchmark or production retriever.

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

Day 71 asks how information visibility and prediction targets distinguish
BERT, GPT and T5-style families. Its
[dated analytical record](71-encoder-decoder-models/notes.md#execution-record--2026-11-25)
reports random-block perturbations, causal prefix invariance and source
padding controls in both encoding and cross-attention. A NumPy reference
agrees with CPU PyTorch SDPA on a rectangular masked case. These are
connectivity checks, not pretrained-model comparisons, language-quality
measurements or serving benchmarks. Interpretation candidates require author
review; the generated JSON remains ignored.

Day 72 asks how learned text units affect sequence length and request budgets.
Its [dated analytical record](72-tokenization/notes.md#execution-record--2026-11-25)
reports deterministic character-BPE merges and actual counts under two named
pretrained byte encodings. The notes distinguish WordPiece inference from its
reconstructed training score and SentencePiece from the algorithms it supports.
Hypothetical cost and weight calculations are separate from measured token
counts. Its separate [visual guide](72-tokenization/VISUAL_LAB.md) documents
sixteen CPU visual groups and a ten-section live dashboard, including locally
trained WordPiece/SentencePiece models, Unigram paths, Unicode, multilingual/OCR
counts and token-safe chunking. No LLM API or language-quality benchmark was
executed. Interpretations and preview selection require author review; generated
exports, local models and full JSON traces stay ignored.

Day 73 asks how self-supervision turns text into prediction targets and what
training loss can establish. Its
[dated analytical record](73-llm-pretraining/notes.md#execution-record--2026-11-25)
reports CLM and MLM runs on twelve synthetic sentences with four reserved
animal-food combinations, plus a separate NumPy bigram fit. Analytical entropy
references explain nonzero training loss; token exposure and theoretical `6ND`
accounting are separated from measured wall time. CLM/MLM losses are different
objectives, not a capability ranking. No real-text benchmark or empirical
scaling law is claimed; interpretations require author review and generated
JSON records remain ignored.

Its separate [visual guide](73-llm-pretraining/VISUAL_LAB.md) records new
quick/full experiments on an expanded symbolic grammar, with pooled PCA,
per-head attention from the actual models and measured data/capacity scenarios.
The full ablation's unmasked accuracy changed from 99.06% with complete inputs
to 30.31% with only the available prefix, versus 46.25% in both causal protocols.
Nine GIFs, ten PNGs and twelve local HTMLs distinguish measured dynamics from
assumed scaling surfaces and compute estimates. All generated artifacts remain
ignored pending curation; no real-text or production conclusion is claimed,
and interpretation remains pending author review.

Day 74 asks when behavior adaptation warrants changing model parameters. Its
[dated analytical record](74-fine-tuning-alignment/notes.md#execution-record--2026-11-25)
reports SFT and DPO using the same low-rank update on a synthetic response
selector, with independent training/evaluation inputs and a frozen SFT reference.
Task accuracy, chosen probability, preference margins and KL are reported
separately. That analytical example does not evaluate natural-language instruction
generalization, PPO or actual QLoRA. Interpretations require author review;
the original generated JSON stays ignored.

Its separate [visual guide](74-fine-tuning-alignment/VISUAL_LAB.md) maps fourteen
demos to actual synthetic training, analytical objectives and theoretical storage.
Thirteen GIFs, twenty-one PNGs and fourteen offline HTMLs show reward-model
fitting, exact finite-policy optimization, DPO, LoRA/SVD, an implemented adapter,
uniform quantization, conflicting-task forgetting and reward misspecification.
These visual configurations and records are separate from the analytical study.
There is no natural-language, human-feedback, measured-memory or production
benchmark. All generated assets stay ignored pending author curation;
interpretation review remains pending. The root inventory includes both phases.

Day 75 asks how conditional token scores become a controlled output sequence.
Its [dated analytical record](75-llm-inference-decoding/notes.md#execution-record--2026-11-26)
reports five temperature conditions with 10,000 draws each, 36 filtering
configurations, seeded context-dependent sequences and separate EOS/text-stop/
token-budget outcomes. Numerical tests include threshold-crossing tokens,
finite extreme logits and matches spanning synthetic fragments. This is an
educational non-streaming decoder, without pretrained weights, tokenizer
integration or model-quality/serving benchmarks. Interpretations require author
review; the dated JSON remains ignored.

Its separate [visual guide](75-llm-inference-decoding/VISUAL_LAB.md) covers
probability/sampling controls, exhaustive comparison on a fixed beam-search tree
and theoretical cache accounting. Selected previews are public; full exports stay
ignored. These synthetic demonstrations do not measure real-model quality or
serving performance.

Day 76 asks how prompt specifications become observable extraction behavior.
Its [dated evaluator record](76-prompt-engineering/notes.md#execution-record--2026-11-26)
reports strict parsing, typed validation, reference equality and complete-output
coverage for nine hand-authored response fixtures. The separate retry control
stops on a schema-valid wrong total. V1/V2/V3 are inspectable instructions, not
evaluated model configurations; no prompt ranking, API cost/latency or injection
robustness is claimed. Interpretations require author review; generated JSON
records remain ignored.

The separate [visual guide](76-prompt-engineering/VISUAL_LAB.md) distinguishes
synthetic simulation, offline replay and an explicit optional real-provider
adapter. An adapter implementation or replay of fixtures is not an executed
model comparison. No provider measurement is added by publishing the guide;
all generated galleries and records remain ignored.

Day 77 asks which checks justify consuming a structured response. Its
[dated analytical record](77-structured-outputs/notes.md#execution-record--2026-11-26)
reports five scripted synthetic sequences with separate parsing, schema,
arithmetic and reference gates. A fabricated but internally consistent invoice
is rejected before any second attempt; malformed responses have a finite total
call budget. A one-step NumPy mask renormalizes permitted token probabilities.
No provider, real-document extraction or production guardrail system was
evaluated. Interpretations require author review; the dated JSON stays ignored.

Its separate [visual companion](77-structured-outputs/VISUAL_LAB.md) adds a
finite-state toy grammar, layered validation of 108 authored responses, independent
typing/extra-field controls and three seeded retry models with 100,000 synthetic
requests each. Curves and surfaces distinguish theoretical reliability/cost from
Monte Carlo observations. A valid synthetic tool proposal is independently
denied permission. Four inspected image previews are selected; complete offline
HTMLs and dated experiment records remain ignored. No real LLM, live tool,
provider cost or production reliability was evaluated; interpretation review
remains pending.
