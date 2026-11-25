# Tokenization visual learning laboratory

This phase adds sixteen connected visual groups and a ten-section live
dashboard. It complements the [analytical study](README.md) without replacing
the original [2026-11-25 analytical record](notes.md#execution-record--2026-11-25).
The visual corpus, experiments and local model configurations are distinct.
All data are handwritten synthetic examples. Interpretation: **author review required**.

## Setup and execution

Use Python 3.11+ and the repository's canonical dependencies from the root:

```powershell
python -m pip install -e ".[dev,nlp]"
python 05-transformers-llms/72-tokenization/visualize.py --setup
python 05-transformers-llms/72-tokenization/visualize.py --quick
python 05-transformers-llms/72-tokenization/visualize.py
python -m streamlit run 05-transformers-llms/72-tokenization/app.py --server.address=127.0.0.1
```

Only initial setup of the two pretrained `tiktoken` encodings needs network
access. The models used through Hugging Face Tokenizers and SentencePiece are
trained from the twelve synthetic sentences in `visual_core.py`; no model hub,
API key, paid request or GPU is needed. `nlp` includes `tiktoken`, `tokenizers`,
`sentencepiece` and the module's existing NLTK dependency. Matplotlib, Pillow,
NumPy, pandas, Plotly and Streamlit come from the shared runtime dependencies.

The generator sets a local `tiktoken` cache unless `TIKTOKEN_CACHE_DIR` already
selects one. Local tokenizer files live under the topic's ignored
`outputs/models-v2/`. Existing earlier model files and runs are retained.

By default, each export run creates its own UTC timestamped directory beneath
`outputs/`. Open that run's `index.html` for a single gallery of all exports.
The live dashboard groups concepts into ten study sections and can tokenize
arbitrary text with all six configurations. Its colored blocks show spaces as
`·`, while tables and JSON retain original strings or exact hexadecimal bytes.

Individual groups are available through the same CLI:

```powershell
python 05-transformers-llms/72-tokenization/visualize.py --section bpe
python 05-transformers-llms/72-tokenization/visualize.py --section multilingual
python 05-transformers-llms/72-tokenization/visualize.py --section rag
python 05-transformers-llms/72-tokenization/visualize.py --section cost
python scripts/validate_repo.py tests --scope 05-transformers-llms/72-tokenization
```

`--section` also accepts `pipeline`, `inference`, `wordpiece`, `unigram`,
`sentencepiece`, `unicode`, `comparison`, `vocabulary`, `embedding`, `attention`,
`ocr` and `context`. Combine any section with `--quick`.
`--output-dir outputs/my-new-run` chooses an explicit directory, resolved
relative to the topic, even when launched from another working directory.
Reusing a nonempty run directory raises an error rather than replacing evidence.

Quick mode uses lower raster resolution, at most eight frames per animation
and three OCR trials. Full mode uses 190-DPI PNGs, all algorithm frames and
twelve OCR trials. The tokenizers, formulas and algorithms are the same.

## Files and responsibilities

| File | Responsibility |
|---|---|
| [visual_core.py](visual_core.py) | Tokenizer training, BPE traces, greedy WordPiece, Viterbi, seeded corruption, chunk spans and measured counts |
| [visual_figures.py](visual_figures.py) | Matplotlib/Pillow PNG and GIF exports of actual computed states |
| [visual_html.py](visual_html.py) | Standalone Plotly pages with embedded scripts/data, controls and formula-based simulations |
| [visualize.py](visualize.py) | Bounded export CLI, artifact inspection, model/configuration provenance and dated manifests |
| [app.py](app.py) | Live Streamlit dashboard with cached local models and live numerical controls |
| [tests/test_visual_core.py](tests/test_visual_core.py) | Independent algorithm, probability, Unicode, normalization and budget contracts |
| [tests/test_visual_app.py](tests/test_visual_app.py) | Actual AppTest navigation and consequential widget interactions |
| [tests/test_visual_rendering.py](tests/test_visual_rendering.py) | Real temporary PNG/GIF/HTML export checks |

## Visual questions and expected exports

Filenames below are output specifications, not links to ignored assets. Every
group is available in the generated gallery; live equivalents are organized in
the dashboard. The experiments JSON provides exact values behind the displays.

| Group / question | What changes and why | Engineering use and boundary | Exports |
|---|---|---|---|
| 01. How do text, bytes and categorical IDs become model input? | Step through identity normalization, the encoding's actual regex pretokenizer, byte-BPE IDs and selected synthetic embedding rows for four strings. | Preserve the tokenizer/model mapping; no pretrained embedding values or Transformer forward pass. | `01_tokenization_pipeline.gif`, `.png` |
| 02. How does BPE construct pieces? | Actual weighted pair counts choose each merge; corpus, new vocabulary entry and compression update. | Understand training and deterministic lexical ties. No boundary markers or production-scale trainer. | `02_bpe_training.gif`, `02_bpe_training_final.png`, `02_bpe_merge_history.csv` |
| 03. How are stored rules applied? | Inspect available ranked pairs in `lowest`, `newer`, `slower`, seen control `newest`, and rejected `café`. | Separate training statistics from inference priorities; no byte fallback. | `03_bpe_inference.gif`, `.html` |
| 04. When do scoring and greedy matching differ? | Change synthetic marginal/joint frequencies; compare frequency and association, then greedy fixed-vocabulary steps and actual HF output. | The association trainer is illustrative. Actual HF training is separately identified and not the original unpublished WordPiece recipe. | `04_bpe_vs_wordpiece.png`, `.html`, `04_wordpiece_inference.gif` |
| 05. Why does a segmentation path win? | A lattice, cumulative log scores, predecessors and backtracking use explicit normalized manual probabilities; sliders change them. | Relate Viterbi to an exhaustive small oracle. These probabilities are not learned. | `05_unigram_segmentation.gif`, `05_unigram_lattice.png`, `05_unigram_interactive.html` |
| 06. What does SentencePiece preserve? | Same corpus, two algorithms; original/normalized strings, whitespace markers, IDs and decoded text are actual library outputs. | Identify loss from default normalization, whitespace collapse or unknown pieces. No equal-quality claim. | `06_sentencepiece_comparison.png`, `06_sentencepiece_whitespace.gif` |
| 07. Where do byte and character boundaries differ? | Inspect code points, UTF-8 bytes, token byte slices and strict full reconstruction for accents, CJK and emojis. | Avoid replacement decoding when a token splits a character. Code points are not grapheme clusters. | `07_unicode_byte_tokenization.gif`, `07_unicode_comparison.png` |
| 08. How do actual tokenizer outputs compare? | Select prose, languages, code, JSON, SQL, legal/technical text, OCR or emojis; arbitrary text and model selections are live. | Tiny locally trained models differ from pretrained encodings. Character BPE uses a separately labeled supported probe in the static summary. | `08_tokenizer_comparison.png` and `app.py` |
| 09. How does vocabulary change train/held-out lengths? | Separate trained BPE merge limits; actual vocabulary, supported lengths, rejected inputs and coverage are measured. | Compression excludes rejected words and does not establish semantic quality. Curves follow results, not an imposed trend. | `09_vocabulary_vs_compression.png`, `09_vocabulary_growth.gif`, `09_vocabulary_tradeoff.html`, `09_vocabulary_metrics.csv` |
| 10. What do vocabulary parameters cost? | Rotate a real 3D surface; change width, vocabulary, tying and ideal FP32/FP16/BF16/INT8 storage. A separate lookup animation highlights rows. | Theoretical weight storage excludes optimizer, gradients, activations, quantization metadata and all other parameters. | `10_embedding_parameters_3d.html`, `10_embedding_memory.png`, `10_embedding_lookup.gif` |
| 11. Which attention workload is being counted? | Sequence lengths 128–4096; dense, causal-valid and single cached-query interactions; 3D interaction-times-width proxy. | Counts are not measured GPU memory, latency or end-to-end FLOPs. FlashAttention need not store all scores. | `11_attention_complexity.gif`, `11_attention_complexity_3d.html`, `11_attention_scaling.png` |
| 12. How do two encodings segment translated content? | Filter the five synthetic approximately equivalent passages; inspect actual IDs/byte fragments and metrics. | Human translation review remains pending. Code points and whitespace words are not universally fair metrics. | `12_multilingual_efficiency.png`, `12_multilingual_heatmap.png`, `12_multilingual_comparison.html`, `12_multilingual_metrics.csv` |
| 13. What does controlled OCR noise do? | Five independent mechanisms, four measured rates, two passages, two encodings and seeded trials; line charts, distributions and changing fragments. | Inflation can be negative. No OCR engine, retrieval accuracy or legal-document understanding is evaluated. | `13_ocr_token_inflation.gif`, `13_ocr_noise_experiment.png`, `13_ocr_noise_interactive.html`, `13_ocr_metrics.csv` |
| 14. How do chunk constraints differ from containment? | Change character/token/sentence rules, size, overlap and tokenizer; inspect exact spans, sentence flags, overlap and final generator prompt count. | Containment checks a known synthetic phrase, not retrieval. Oversized sentences fall back to token-safe spans; no truncation is performed. | `14_rag_chunking.gif`, `14_rag_chunking_comparison.png`, `14_rag_chunking_interactive.html` |
| 15. What fits after output reservation? | Adjust instructions, history, query, retrieval, tools, overhead and reserved output; optional pruning removes lower-priority chunks. | Hypothetical limits; sums assume already-counted components. Irreducible overflow stays visible. Capacity is not effective reasoning quality. | `15_context_window_simulator.html`, `15_context_budget.png` |
| 16. Which factors drive a cost projection? | Prices, input/output lengths, daily/monthly volumes, context reduction and a rotating 3D monthly-cost surface. | Hypothetical USD, not provider billing or production savings. Reducing context has no measured quality guarantee. | `16_llm_cost_simulator.html`, `16_monthly_cost_3d.html`, `16_cost_sensitivity.png` |

Each standalone interactive page includes labeled controls, reset, local state
inspection, hover information and Plotly navigation. Range-based pages also
support play/pause. The live dashboard supports step navigation without automatic
rerun loops. 3D export pages require browser WebGL; all underlying calculations
and raster exports run on CPU.

## Experimental configuration and reproducibility

- **Synthetic character BPE:** weighted words `low:5`, `lower:2`, `newest:6`,
  `widest:3`; lexical tie breaking; ten merges for tracing. Compression sweeps
  use limits 0, 2, 4, 6, 8, 10, 15 and 25. Held-out words are `lowest`, `newer`,
  `slower`, `weirdest`, `café`; one is rejected because its characters are absent.
- **HF WordPiece:** target vocabulary 384; NFKC plus `Whitespace`; ordered
  continuation-symbol initialization removes randomized symbol-ID tie effects.
  The trained vocabulary receives canonical lexical IDs, and continuation pieces
  lose their training-only special status. Only `[UNK]` remains special. These
  newly trained local IDs have no pretrained embedding weights to invalidate.
  Repeated fresh training is tested for matching pieces and IDs.
- **SentencePiece:** both requested sizes 256; `nmt_nfkc`, full character
  coverage, no byte fallback, one thread, no sentence shuffling, no subsampling,
  no BOS/EOS/PAD and `hard_vocab_limit=False`. Actual sizes differ and are recorded.
- **Pretrained encodings:** explicit `cl100k_base` and `o200k_base`; ordinary-text
  policy and full UTF-8 byte reconstruction. The existing
  [resource hashes](references.md#text-and-downloaded-data) still apply.
- **Manual Unigram example:** the complete vocabulary distribution is in code;
  an unused piece retains probability mass. Three complete segmentation paths
  are evaluated; the distribution is normalized again after live weight changes.
- **OCR:** seed 272 plus trial index, separate substitution/inserted-space/
  hyphenated-break/removed-space/deletion arms, probabilities 0, .05, .15 and .30.
  The same trial stream pairs rates, not different corruption mechanisms.
  Substitution only changes mapped letters; the opportunity rate is not a
  realized edit fraction. Full mode has 960 rows; quick mode has 240.
- **Chunking:** original character spans preserve UTF-8. Overlap uses code points
  in character mode and token budgets in token/sentence modes. Reencoded overlap
  token counts describe the repeated text slice; boundary merges can make them
  differ from a unique-token duplication count. The generator prompt is counted
  after delimiters and instruction text are assembled. The standalone RAG page
  offers an explicitly measured grid (sizes 24/40/64, overlap 0/4/8); the live
  dashboard recomputes arbitrary supported settings.

Every run retains `manifest.json` with versions, date, model hashes, corpus
fingerprint and artifact hashes, plus `experiments.json` and CSV measurements.
Reruns use new directories. Model snapshots make exact local replay inspectable;
cross-version library changes still require a separately dated result.

## Evidence classes and limitations

**Measured:** actual segmentation, IDs, byte reconstruction, supported-character
coverage, train/held-out lengths, OCR counts, chunk spans and known-phrase
containment. **Theoretical:** weight storage, attention interaction/proxy counts,
context accounting and hypothetical monetary projections. **Illustrative:**
synthetic embeddings, manual Unigram probabilities and the association-score
simulator. No learned Transformer is evaluated.

The five translations are manually written approximate equivalents and need
author review. Tiny tokenizer vocabularies are not competing foundation models.
Unknown-token behavior and normalizers can lose source information. Sentence-aware
chunking uses simple punctuation boundaries, not a linguistic sentence segmenter.
Encoding a single Unicode code point can exceed a very small token budget; the
chunker raises an explicit error rather than silently corrupting it.

Recommended previews for **human selection only**: `02_bpe_training.gif`,
`05_unigram_lattice.png`, `12_multilingual_efficiency.png`,
`09_vocabulary_vs_compression.png` and `15_context_budget.png`. No output is
automatically unignored or linked as a public preview. Models, exports, records
and browser inspection artifacts remain local. Root README visual highlights
are deferred to human review as requested.

## Execution record — visual phase, 2026-11-25

The initial quick diagnostic run is retained locally and is not the final
visual evidence: it exposed font warnings and preceded the deterministic
WordPiece training correction. Its configuration is not mixed with the original
analytical counts or the final full run.

**Full measurement/export record:** 2026-11-25 22:20:56 UTC
(19:20:56 America/Sao_Paulo), seed 272, CPU. Python 3.11.14, NumPy 2.4.6,
Matplotlib 3.11.2, Pillow 12.3.0, Plotly 6.9.0, `tiktoken` 0.14.0,
`tokenizers` 0.22.2, SentencePiece 0.2.3 and Streamlit 1.67.0.

**Configuration:** the protocols above, twelve OCR trials per arm, full GIF
frame sequences and 190-DPI static plots. The training-corpus SHA-256 is
`d77e6c6b89635fcd24cf1f63d638ad20cd844b8b5a5e5ccdbb38a141ff97ba29`.
The local model metadata records 384 WordPiece entries, 256 SentencePiece BPE
entries and 176 SentencePiece Unigram entries, with each model's serialized hash.

**Hypotheses:** pair ranks explain encoding states; assigned Unigram probabilities
determine the optimal segmentation; learned encodings and controlled corruption
change token counts; chunk boundaries affect known-evidence containment.

**Executed:** `visualize.py --quick`, `visualize.py --output-dir outputs/full-validation`,
`visualize.py --setup`, and `--section cost --quick` from a different working
directory. Full export produced **44 artifacts: 16 PNGs, 12 GIFs, 12 standalone
HTMLs and 4 CSVs**, plus `index.html`, `experiments.json` and `manifest.json`.
PNG dimensions are 2660 × 1615; full GIF frames are 1400 × 850. Every raster
was decoded and every GIF frame loaded by the export inspection. Representative
pipeline, BPE, Unigram, normalization, Unicode, comparison, vocabulary, chunking
and dashboard images were also visually inspected.

A separate **presentation revision** at 22:28:30 UTC retained the full measured
JSON and raster files byte-for-byte, and regenerated the twelve HTMLs so their
token palette matches the PNG/GIF/dashboard SHA-256 mapping. This is not a new
experiment. The local final gallery is in
`outputs/reviewed-20261125T222826968910Z/index.html`; its manifest preserves the
measurement time, parent-manifest hash and presentation-revision time.
The original run directories remain intact. A normal new `visualize.py` run
recreates the current complete presentation in a new directory.

**Observed counts:**

| Synthetic translated probe | `cl100k_base` | `o200k_base` |
|---|---:|---:|
| English | 13 | 13 |
| Portuguese | 21 | 17 |
| Spanish | 19 | 17 |
| Chinese | 26 | 16 |
| Japanese | 43 | 34 |

These are different probes from the shorter analytical examples. They must not
be substituted for, or combined with, the original analytical table.

The 960 OCR rows ranged from **−11.7647% to +350%** token inflation over their
own clean baselines. The sweep learned at most 12 BPE merges on this corpus,
ending with 22 vocabulary entries: 16 training tokens versus the initial 79.
The four supported held-out words used 12 tokens versus their initial 25;
`café` remained rejected, so held-out word coverage was 80%. The manual Unigram
example selected `unbeliev + able`, joint probability .05 and log probability
approximately −2.995732. Exhaustive enumeration found the same best path.

For the fixed chunking comparison at size 40 and overlap 5, character chunks
had maximum 12 embedding tokens, token chunks maximum 40, and sentence-aware
chunks maximum 39. The known evidence phrase was contained by the latter two,
but split by the character rule. Character and token size units differ, so
this is not an equal-budget quality comparison. No text was truncated and no
retriever or answering model was run.

**Interpretation candidates — author review required:** token inflation need not
be positive; tokenizer counts must be measured on the target text/interface;
training compression can outpace supported held-out compression; context-limit
compliance does not ensure evidence containment. No language-wide efficiency,
OCR quality, retrieval quality, answer quality or production savings conclusion
follows from these small synthetic protocols.

**Validation:** 77 topic tests passed across four isolated files, including 33
original analytical cases, 41 visual-core cases, one multi-section AppTest and
two real-export cases. No pytest skips or warnings were reported. The running
Streamlit 1.67.0 agent API returned `ready` for 27 inspected interactions across
the ten sections and selected subviews. Real Chrome navigation exercised all
ten dashboard sections plus cost changes and reset. Twelve offline HTML pages
passed control, reset, animation and numerical checks, including mouse rotation
of all three 3D scenes. The exact browser-inspected HTML payloads were retained
in the presentation revision.

**Rendering limits:** Windows desktop Headless Chrome 158 with SwiftShader;
zero JavaScript/console errors and no external HTTP(S) requests in the inspected
flows. Canvas readback/WebGL performance warnings occurred. Matplotlib reported
regular-weight fallback for the emoji font's unavailable bold face; missing-glyph
warnings were resolved and representative Unicode output was inspected. These
checks do not establish native-GPU performance, mobile/cross-browser behavior,
complete accessibility or GitHub rendering. No remote CI or deployment was run.

Root EN/PT visual highlights and all public-preview decisions remain for human
review. The module index and topic documentation identify the implemented visual
phase separately from author approval.
