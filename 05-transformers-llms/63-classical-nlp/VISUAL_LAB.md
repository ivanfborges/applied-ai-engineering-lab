# Classical NLP — interactive visual learning lab

This is a separate visual phase of [Day 63](README.md). The original six-document
analytical study and its dated record remain unchanged. All new corpora, queries
and relevance judgments are synthetic teaching fixtures. **Author interpretation
review required.** Passing numerical checks and readable exports do not establish
scientific approval, realistic retrieval quality or production readiness.

## Run locally

From the repository root, use stable Python 3.11+ and the canonical dependency
bounds in [pyproject.toml](../../pyproject.toml):

```powershell
python -m pip install -e ".[dev,nlp]"
python -m nltk.downloader -d .local/nltk_data wordnet
python 05-transformers-llms/63-classical-nlp/visualizations.py --list
python 05-transformers-llms/63-classical-nlp/visualizations.py --all --quick --output-dir 05-transformers-llms/63-classical-nlp/assets/generated/quick
python 05-transformers-llms/63-classical-nlp/visualizations.py --all --seed 63
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic tokenization
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic ngrams
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic tfidf
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic retrieval
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic language_model
python 05-transformers-llms/63-classical-nlp/visualizations.py --topic ocr
```

Additional topic names are `normalization`, `morphology`, `vocabulary`, `dtm`,
`vectors`, `cosine`, `rag`. Repeat `--topic` to select several. The generator
defaults to `assets/generated/` relative to this study, regardless of the shell's
working directory. `--output-dir` and `--seed` are configurable. Quick mode reduces
raster resolution and uses four OCR trials instead of twenty; it still generates
every essential demonstration. No API, GPU or cloud account is used.

NLTK tokenizers and stemmers run without downloaded models. Only resource-backed
English morphology needs WordNet. Setup is explicit and separate from rendering;
widgets and generators never trigger downloads. The runtime searches the ignored
project-local resource directory and NLTK's normal resource paths. Missing NLTK
or WordNet is reported while unrelated topics continue; the manifest records
unavailable topics and the report records partial morphology/retrieval branches.
CI installs the extra and WordNet to run the real resource-backed tests.

Optional dashboard, bound only to loopback:

```powershell
python -m streamlit run 05-transformers-llms/63-classical-nlp/app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

The dashboard recomputes edited text, normalization, n-grams, counts, pruning,
TF-IDF, SVD, vector geometry, conditional distributions and retrieval scores.
For morphology, POS is supplied, not predicted. Free-text query inputs have no
assumed relevance judgments; judged-query means refer to the separate fixed
fixture. The repeated OCR curve is cached by seed and trial count, while the
editable query corruption and retrieval recompute for its own controls.

For interactive exports, open the generated files directly where the browser
permits it, or serve them without exposing other repository content:

```powershell
python -m http.server 8873 --bind 127.0.0.1 --directory 05-transformers-llms/63-classical-nlp/assets/generated
```

Plotly is embedded inline, including the actual surface data. Standalone HTML
does not need a CDN or running Python server for its interactive controls.

## Questions answered by each artifact

Each output path below is **relative to the ignored output directory**, not a
public documentation asset link. `manifest.json` and generated `INDEX.md` record
purposes, exact relative paths, structural checks and latest topic status.
Every invocation writes a distinct `experiments-<UTC timestamp>.json`; existing
dated records are retained. Named renders reflect their most recent generation.

| Group and outputs | Representation and calculation | Controls / preserved information / loss / engineering use |
|---|---|---|
| 1: `01_tokenization.gif`, `01_tokenization_explorer.html` | Six raw bilingual sentences; whitespace, regex, NLTK boundary rules, code points, character BPE and UTF-8 bytes. Actual source offsets and corpus vocabulary counts. | Choose sentence and strategy; animate boundaries. Bytes preserve all source bytes, while a learned character alphabet needs an unknown policy. Whitespace and regex differ on punctuation, legal identifiers and contractions. Sequence length affects compute; a token is not necessarily a word. |
| 2: `02_normalization_pipeline.gif`, `02_unicode_normalization.png` | A sequential NFC/casefold/whitespace pipeline, followed by independent normalization alternatives. Diff-based source/output colors and actual code points for NFC/NFD/NFKC/NFKD. | Dashboard can select and order transformations. Preserve original evidence separately. Case, accents, numbers and punctuation can carry meaning; optional English stopword removal can delete negation. Python length counts code points, not visual glyphs. |
| 3: `03_stemming_vs_lemmatization.html`, `03_morphological_transformations.gif` | Actual Porter, English Snowball, WordNet and Portuguese Snowball outputs. Distinct surface/output counts and their reduction; Sankey links and grouped mappings. | Select algorithm. WordNet uses explicit POS; raw output and candidates are retained. Over/under-conflation, irregulars and nonwords matter more than vocabulary reduction alone. Portuguese has its own stemmer. |
| 4: `04_ngram_sliding_window.gif`, `04_ngram_explorer.html` | Sliding contiguous windows for orders 1–4; $G(m,n)=\max(0,m-n+1)$. Repeated tokens, empty input, $n>m$ and character windows. | Edit sentence, unit and order, then advance windows. Preserves local adjacency and repetitions; loses global order and scope. HTML word mode is explicitly NFC/lowercase with a browser Unicode regex; Python uses the shared NFC/casefold analyzer. |
| 5: `05_vocabulary_growth.png`, `05_feature_sparsity.png`, `05_feature_space_explorer.html` | 45 code-defined documents; cumulative word ranges 1, 1–2, 1–3. Separate fixed-order observed counts versus $K^n$ bounds. CSR storage = data + indices + indptr bytes. | Select `min_df`, `max_features` and order. Occupancy and pruning are real scikit-learn computations. ID-like tokens are deliberately present. Counts and memory accounting do not measure generalization, fit quality or peak RAM. |
| 6: `06_document_term_matrix.gif`, `06_bag_of_words_comparison.png` | Three documents become tokens, vocabulary, column indices, counts and a matrix. Compare unigram and bigram features with equivalent CountVectorizer preprocessing. | Dashboard accepts document lines and order. `dog bites man` and `man bites dog` have equal unigram vectors but different bigrams. Fixed feature ordering is required for comparison; adjacency is still not full semantics. |
| 7: `07_tfidf_heatmaps.png`, `07_idf_decay.png`, `07_tfidf_surface_3d.html` | Common/rare terms expose TF, DF, smoothed IDF, TF×IDF and L2 normalization. Each heatmap uses its own explicitly labeled scale. Surface values follow the formula exactly. | Change integer corpus size, DF and TF. DF must lie in 1…D. Selected marker and fixed-TF section update. Surface is **unnormalized**; relative document weights after L2 need all terms. Rare terms can be noise as well as useful evidence. |
| 8: `08_document_vector_space_3d.html`, `08_cosine_similarity_heatmap.png` | 15 documents, five topics; unigram, word 1–2 and character 3–5 TF-IDF. Original-space cosine and document-fitted three-component TruncatedSVD. | Choose representation and eight predefined queries; hover original text, rank and score, or rotate camera. Large markers mean top lexical scores, not known relevance. Projection can distort relationships; it never supplies the ranking. |
| 9: `09_cosine_similarity.gif`, `09_vector_geometry_3d.html` | Interpretable signed vectors, dot product, norms, actual angular separation and cosine; zero-vector case is explicit. | Change azimuth, elevation and magnitude. Direction stays meaningful under scaling. Negative cosine belongs to generic signed coordinates, not ordinary nonnegative TF-IDF. A library zero score is a convention, not a defined zero-vector angle. |
| 10: `10_ngram_transition_graph.html`, `10_language_generation.gif`, `10_smoothing_comparison.png` | Counter unigram/bigram/trigram models; BOS context, EOS prediction, add-alpha and fixed-seed sampling. Directed observed edges encode conditional probabilities. | Dashboard changes order, alpha, seed and context. Graph can inspect complete distributions. Smoothing gives unseen events mass; unseen unsmoothed contexts are undefined, with no implicit backoff. Only observed graph edges are drawn even when smoothing assigns mass elsewhere. |
| 11: `11_preprocessing_ranking_comparison.gif`, `11_retrieval_metrics.png` | Six contrasting administrative sentences, seven queries and prior binary judgments. Seven preprocessing settings; positive-score ranks, Recall@3, untruncated MRR and binary nDCG@3. | Dashboard changes stopwords, morphology, order and query. Relevance judgments are labels, not score-derived truths. Removing `not` or `without` can create collisions; bigrams need not repair them. Per-query rankings and misses are saved in the dated JSON. |
| 12: `12_ocr_robustness.png`, `12_ocr_corruption_explorer.html` | Clean fixed 16-document index, six judged queries, paired repeated corruptions over five rates; word, word 1–2, `char` and `char_wb`. | Explore rate/representation in HTML, and arbitrary query/seed/rate in the dashboard. Edits include substitutions, deletions, insertions, visual confusions and accent changes. Bands are ±1 trial SD, not confidence intervals. Synthetic selection probability is not calibrated OCR error rate. |
| 13: `13_hybrid_rag_architecture.png` | **Illustrative architecture**, showing extraction/OCR, preservation, sparse and model-native dense branches, fusion, optional reranking, context and generation. | Diagram only; no dense embeddings, BM25, rank fusion or LLM execution. Language-aware lexical transformations may help one branch and damage a pretrained dense model's expected input. No dense ranked list is fabricated. |

## Important implementation details

The trained BPE is an educational **character-pair merge algorithm**, not a
pretrained model tokenizer. It uses the six tokenization sentences as its whole
training corpus, up to forty merges with frequency ties resolved lexicographically,
and an explicit `<unk>` for unseen characters. Merges may span spaces. Its actual
alphabet, vocabulary, merges and training-corpus SHA-256 are included in the HTML
and dated report. Corpus-observed vocabulary differs from the model vocabulary.
The UTF-8 demo emits one token per actual byte from the 256-value byte alphabet.
Character segmentation counts Unicode code points, not grapheme clusters.

Linguistic boundary examples use NLTK Treebank for English contractions and
Toktok for the supplied Portuguese sentences. This is rule-based boundary
tokenization, not a downloaded language model, morphological parser or sentence
segmenter. Arbitrary dashboard text that is not one of the supplied PT examples
uses the English-oriented Treebank baseline; it is not automatically language
detected. No pretrained segmentation is invented.

WordNet's default lemmatizer is permissive and can return an unchanged input.
Actual morphology candidates are exposed through its documented `_morphy`
interface. For `saw`, the contextual verb reading in “I saw a bird” should be
`see`, while the noun in “I bought a saw” stays `saw`; POS alone need not force
that contextual choice. The lab retains the real resource output and labels
the contextual expectation, rather than substituting a manually chosen result.
`leaves` supplies a noun/verb contrast. Mechanical conflation of
`universe`/`university`, separation of `mouse`/`mice` or `run`/`ran`, and stems
such as `studi` are diagnostic cases whose usefulness depends on the task.
They are not a statistical morphology-accuracy evaluation.

The retrieval lemma experiment uses supplied POS for a small synthetic vocabulary;
unlisted tokens default to noun. Stopword policies use the explicit teaching
set in `nlp_lab/core.py`, not an automatically downloaded corpus. The preservation
policy retains `not`, `no`, `never`, `without`; `with` is still removed, a choice
whose errors must be inspected. All comparisons use fixed corpora and positive
scores only; zero matches return no ranking, and ties use document ID.

## Source and verification

- [visualizations.py](visualizations.py): CLI, topic registry, manifest, dated
  reports and structural asset validation.
- [nlp_lab/core.py](nlp_lab/core.py): text, counting, geometry, statistical models,
  synthetic corpora, judgments, retrieval metrics and corruption.
- [nlp_lab/figures.py](nlp_lab/figures.py): PNG/GIF calculations and rendering.
- [nlp_lab/interactive.py](nlp_lab/interactive.py): shared Plotly figures and
  self-contained HTML controls.
- [app.py](app.py): optional local dashboard.
- [tests/test_visual_core.py](tests/test_visual_core.py): numerical contracts,
  library parity, boundary policies, real morphology and deterministic trials.
- [tests/test_visual_exports.py](tests/test_visual_exports.py): real exports into
  temporary directories and external HTML dependency rejection.
- [tests/test_visual_app.py](tests/test_visual_app.py): all dashboard sections and
  selected parameter changes through AppTest.

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/63-classical-nlp
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
python -m pip check
git diff --check
```

Repository tests run each file in a separate process. Real exports test Pillow
readability and standalone HTML structure, not browser interaction. AppTest
does not establish WebGL behavior, mobile accessibility, GitHub rendering or
scientific validity. Full local generation and browser inspection are recorded
separately below. No suite without an explicit scope is claimed here.

## Execution record — visual phase, 2026-11-24

The AI assistant executed all thirteen export groups on Windows with stable
Python 3.11.14. The main run is `20261124T141129681212Z`; a targeted
`20261124T141514902282Z` run refreshed vectors/OCR with shared normalization,
and `20261124T142351336329Z` refreshed tokenization with wrapped labels. A prior
quick-mode smoke run is retained separately under `quick/` and is not the final
numerical reference. HTML-only refreshes retained experiment values while fixing
browser rendering; the final structural record is
`asset-validation-20261124T142712Z.json`. All identifiers refer to ignored files
in the generated directory. The original [analytical record](notes.md) remains
intact; these are new protocols and not renamed historical runs.

Versions: NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, NLTK 3.10.4,
WordNet 3.0, Matplotlib 3.11.2, Plotly 6.9.0, Pillow 12.3.0, NetworkX 3.6.1,
Streamlit 1.67.0 and pytest 9.2.0. The manifest classifies all 28 named files
as regenerable: **8 GIFs, 11 PNGs and 9 standalone HTMLs**, approximately 40.8 MB
combined, excluding dated JSON records and the preliminary quick set. All files
were read structurally; representative animation frames, charts and screenshots
were inspected. None is selected or unignored for public preview publication.

**Experiment A — vocabulary and sparse storage**

- Hypothesis: adding observed local windows increases feature dimensionality;
  actual corpus occupancy and storage remain far below the fixed-order bound.
- Configuration: the same 45 synthetic documents, ranges 1–1, 1–2, 1–3,
  `min_df=1`, no feature cap, int64 count data and the shared analyzer. No seed
  is needed for corpus construction.
- Observed result:

| Range | Shape | nnz | Density | Dense array bytes | CSR array bytes |
|---|---|---:|---:|---:|---:|
| 1–1 | 45 × 89 | 511 | 0.127591 | 32,040 | 6,316 |
| 1–2 | 45 × 230 | 1,005 | 0.097101 | 82,800 | 12,244 |
| 1–3 | 45 × 408 | 1,457 | 0.079357 | 146,880 | 17,668 |

- Interpretation candidate — **author review required**: this constructed corpus
  gains columns faster than occupied cells; sparse arrays avoid storing its many
  zeros. Fixed-order $K^n$ is a combinatorial ceiling, not these cumulative counts.
- Limitation: this is array accounting on a small repeated synthetic corpus,
  not a peak-memory, runtime, overfitting or retrieval-quality benchmark.

**Experiment B — negation ablation and judged lexical retrieval**

- Hypothesis: deleting terms that encode opposition creates representation
  collisions and can damage a fixed query's ranking. Bigrams need not improve
  an already easy lexical fixture.
- Configuration: six contrasting English sentences, seven queries and explicit
  binary relevant-document IDs defined in `core.py`; TF-IDF/cosine, positive-score
  rankings, document-ID tie breaks, $k=3$. Compare seven source-defined
  preprocessing settings. Lemmatization uses supplied tiny-vocabulary POS.
- Observed result:

| Setting | Recall@3 | MRR | nDCG@3 |
|---|---:|---:|---:|
| Original unigrams | 1.000000 | 1.000000 | 1.000000 |
| Drop teaching stopwords | 0.857143 | 0.642857 | 0.698970 |
| Preserve negation | 1.000000 | 1.000000 | 1.000000 |
| Porter stemming | 1.000000 | 1.000000 | 1.000000 |
| WordNet with supplied POS | 1.000000 | 1.000000 | 1.000000 |
| Original unigrams + bigrams | 1.000000 | 1.000000 | 1.000000 |
| Preserved-negation unigrams + bigrams | 1.000000 | 1.000000 | 1.000000 |

Per-query scores, exact ranks and misses are retained in the full run's JSON;
no aggregate score substitutes for that failure inspection. MRR is not truncated
at three, whereas Recall and nDCG are.

- Interpretation candidate — **author review required**: destructive token
  deletion harms this teaching fixture, while morphology and bigrams show no
  aggregate improvement over its already perfect unigram baseline.
- Limitation: labels are hand-authored teaching assumptions, the queries reuse
  corpus vocabulary heavily, and there is no held-out real query population,
  production index, statistical comparison or validated negation understanding.

**Experiment C — repeated synthetic OCR corruption**

- Hypothesis: local character windows can preserve lexical evidence after some
  word spellings are damaged, while also admitting partial or spurious matches.
- Configuration: a clean 16-document index, six pre-judged queries including one
  Portuguese query, four representations, selection rates 0, 0.1, 0.2, 0.35, 0.5,
  twenty paired trials at each rate, base seed 63. The same corrupted queries
  are used across methods. All representations share NFC/casefold/whitespace
  preprocessing before their respective analyzers. Character ranges are 3–5.
- Observed MRR means across trials, selected rates:

| Representation | Rate 0 | Rate 0.2 | Rate 0.5 |
|---|---:|---:|---:|
| Word unigram | 1.000000 | 0.423611 | 0.016667 |
| Word 1–2 | 1.000000 | 0.423611 | 0.016667 |
| Character 3–5 | 1.000000 | 0.984722 | 0.900000 |
| Within-word character 3–5 | 1.000000 | 0.995833 | 0.908333 |

The plot includes every configured rate and ±1 trial SD. All 400 method/rate/trial
rows retain Recall@3, RR and nDCG@3, plus selected actual corrupted-query examples,
retrieved IDs and top-document text. These bands are not confidence intervals.

- Interpretation candidate — **author review required**: these character
  representations preserve more relevant matches under this synthetic corruption
  generator. Equality of the two word baselines and strong character performance
  are observations for this configuration, not universal method rankings.
- Limitation: artificial noise is not calibrated to an OCR engine, language or
  document population. Trials share the same six queries; changing index/domain,
  longer identifiers, true OCR confusions or relevance judgments may change the
  outcome. Selection probability is not measured edit distance.

**Computed mechanism checks and illustrative scope**

The local BPE learned 25 merges and has 72 vocabulary entries including `<unk>`.
WordNet returned `saw` for both supplied noun and verb tags, with actual verb
candidates `saw` and `see`; supplied noun/verb `leaves` produced `leaf` and `leave`.
These are resource outputs, not automatic contextual annotation. The SVD retained
approximately 29.99%, 29.20% and 27.71% of variance for word, word 1–2 and character
features respectively; these values describe a lossy projection, not retrieval
quality. Counter-model conditional probabilities normalized within
$2.23\times10^{-16}$ for the checked contexts/orders. The seeded bigram draw
produced `the was approved </s>`, illustrating that normalized next-token
statistics do not guarantee grammatical text. The RAG architecture is purely
illustrative: dense embedding, fusion, reranking and generation were not executed.

**Executed verification and boundaries**

The final focused command passed **108 tests** across four isolated files:
52 analytical, 50 visual-core, four real-export and two dashboard tests; no
skips or warnings in that final run. Syntax parsed 442 public Python files;
1,611 internal links across 279 files passed before the final record text was
added. Installed compatibility passed for 81 packages using the stable
interpreter and `uv --no-cache --offline pip check`; `git diff --check` passed.
These are local checks, not a GitHub Actions result or full-repository test run.

Chrome 158.0.8127.37 at 1440 × 1000 passed twenty HTML interaction checks,
including actual 3D mouse rotation, changed surface values, invalid DF rejection,
empty/repeated n-grams, morphology selection and OCR examples. The running
Streamlit app passed 25 agent-API requests over all twelve sections and selected
parameter changes, then all twelve sections in the browser, including the edited
`é` input yielding two UTF-8 byte tokens. Completed browser checks had no
page/console errors or external page requests; external page HTTP(S) was blocked.
Three Canvas2D readback warnings appeared in the export inspection and one in
the app inspection. Desktop headless WebGL is not native-GPU, mobile,
cross-browser, GitHub-rendering or accessibility evidence.

Initial checks exposed an incorrect expected WordNet output, missing display
glyphs and a JavaScript newline-escaping bug in the transition HTML; these were
corrected before completed checks. Browser harness waits were also corrected to
observe finished reruns and the actual WebGL canvas rather than stale state.
Local snapshots, screenshots and diagnostic logs remain ignored. No private raw
study, LinkedIn draft or root README content was changed in this visual phase.

Larger-corpus relevance evaluation, real OCR, automated POS accuracy, dense
retrieval and production performance remain **not executed**.

## Preview recommendations and publication review

All 28 named outputs remain ignored. Do not link ignored assets as public
previews. Candidates for a small reviewed set are the n-gram sliding-window GIF,
document-term construction GIF and OCR robustness PNG. Morphology, TF-IDF and
document-space HTMLs are useful local explorations; inline Plotly makes them
several megabytes each, so their delivery should be intentional.

Before publication, the author should review the relevance labels, language/POS
assumptions, WordNet ambiguity, OCR corruption model, projection limitations and
interpretation candidates. No preview is automatically unignored or embedded in
the public README. Root EN/PT progress counts remain unchanged during this
visual phase; any evidence-description update is only recommended after review.
