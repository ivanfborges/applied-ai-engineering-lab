# Day 64 visual laboratory

This is a separate visual study, extending the [analytical example](README.md).
All documents, categories, legal identifiers, queries and relevance judgments
are synthetic. The judgments were defined in code for the educational task;
they are not author-approved annotations or externally validated ground truth.
**Interpretations require author review.** No generated asset is selected for Git.

## Run locally

Python 3.11+; dependencies are centralized in the repository's
[pyproject.toml](../../pyproject.toml). NumPy, SciPy, scikit-learn, Matplotlib,
Plotly and Pillow run the core laboratory. Streamlit is needed only for the
dashboard; it is already in the repository's shared runtime. No new extra,
paid API, model download, GPU or external dataset is required.

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --list
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --quick
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --all
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --section cosine
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --section tfidf --section retrieval
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --asset 21_document_similarity_heatmaps.png
python -m streamlit run 05-transformers-llms/64-bow-tfidf-text-similarity/retrieval_dashboard.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
python scripts/validate_repo.py tests --scope 05-transformers-llms/64-bow-tfidf-text-similarity
```

Use the stable interpreter selected by your local workflow in place of `python`.
`--quick` generates the essential sections at lower raster resolution, with
the same numerical definitions. `--all` adds storage, TF controls, projections,
n-grams, postings, BM25, mismatch, drift, fusion and evaluation. `--asset` renders
one named asset. `--show` opens generated HTML pages after a successful run;
the generator otherwise opens no browser. Opening HTML needs neither a server
nor an internet connection. The optional dashboard needs a local Python server.

Every run creates a new dated directory under `outputs/`, with `png/`, `gif/`,
`html/`, a generated `CATALOG.md`, and a dated JSON record under `results/`.
`--output-dir PATH` accepts an empty directory and refuses to overwrite an
existing run. Records include versions, parameters, results, asset hashes,
dimensions and frame durations. All these paths remain ignored. The generated
catalog links the assets locally; public Markdown does not link ignored exports.

The [generator](visualizations.py) dispatches named sections. Reusable numerical
logic lives in [text_lab/core.py](text_lab/core.py), scientific rendering in
[text_lab/figures.py](text_lab/figures.py), and offline HTML in
[text_lab/interactive.py](text_lab/interactive.py). The
[dashboard](retrieval_dashboard.py) calls the same numerical core.

## Catalog

Filenames below are output names, not links to versioned previews. Each GIF
ends on its completed state and loops; repeated final frames may be coalesced
by Pillow while preserving playback duration. HTML includes the Plotly runtime.

| Filename | Format | Question / interpretation boundary |
|---|---|---|
| `01_text_to_vector.gif` | GIF | Eight stages from source text to CSR, with a highlighted token/column mapping; default word analyzer only. |
| `02_representation_comparison.png` | PNG | Binary, count, relative TF, weighted and unit matrices; all cells calculated, same vocabulary, separate color scales. |
| `03_tfidf_transformation.gif` | GIF | Counts × IDF, then L2 division; frequent and rare term weights are shown. |
| `04_sparse_matrix_structure.png` | PNG | Nonzero occupancy and a small dense view; storage structure, not retrieval speed. |
| `05_csr_representation.gif` | GIF | Row-wise values, column indices and pointer offsets; implicit zeros are absent. |
| `06_sparse_vs_dense_memory.png` | PNG | Actual array-component bytes, log axis; excludes object overhead and peak RSS. |
| `07_term_frequency_comparison.png` | PNG | TF factors for 0–100 occurrences; length and average length fixed at 100. |
| `08_term_frequency_interactive.html` | HTML | Adjust TF, document/average length, k1 and b; TF is clamped to document length. |
| `09_idf_comparison.png` | PNG | Classical, smoothed and positive BM25 IDF for N=10/100/1,000. |
| `10_idf_rarity_animation.gif` | GIF | DF increases from 1 to 10 at fixed N=10; classical self-information falls. |
| `11_idf_interactive.html` | HTML | Adjust N and DF with 1 ≤ DF ≤ N; no logarithm of a zero probability. |
| `12_tfidf_weight_surface_3d.html` | HTML | Raw/sublinear TF × smoothed IDF; actual weight surface, not an embedding landscape. |
| `13_bow_vector_space_3d.html` | HTML | Exactly three vocabulary coordinates; switch counts, weighting and normalization. |
| `14_tfidf_vector_transformation.gif` | GIF | IDF changes directions; L2 then rescales along each ray, with fixed axes. |
| `15_normalized_vector_space_3d.html` | HTML | Nonzero unit vectors on the positive-octant sphere; no projection. |
| `16_cosine_similarity_2d.html` | HTML | Angle and two magnitudes; coordinates, dot, norms, cosine and distance recomputed. |
| `17_cosine_similarity_3d.html` | HTML | Azimuth, elevation, magnitudes, projection, unit vectors and unit sphere. |
| `18_cosine_similarity_animation.gif` | GIF | Rotation through parallel, orthogonal and opposite signed vectors; TF-IDF cannot be negative. |
| `19_l2_normalization.gif` | GIF | Radial rescaling to the unit circle; the zero vector stays at the origin. |
| `20_similarity_metrics_comparison.png` | PNG | A long off-angle document wins raw dot but loses cosine; zero-row exception is explicit. |
| `21_document_similarity_heatmaps.png` | PNG | Binary/count/word/character cosine; all cells, symmetry and nonzero diagonal. |
| `22_document_similarity_clusters.png` | PNG | Average linkage on 1−cosine; no Ward/cosine mismatch. |
| `23_document_projection_2d.html` | HTML | Count/TF-IDF SVD, categories, IDs, text hover and document-fitted query transform. |
| `24_document_projection_3d.html` | HTML | Same fitted 3-component projection with a genuine rotatable third SVD axis. |
| `25_bow_vs_tfidf_projection.png` | PNG | Separate SVD fits and explained variance; projection distances are approximate. |
| `26_ngram_feature_comparison.png` | PNG | Word-order, negation and OCR pair similarities under four feature spaces. |
| `27_ngram_tokenization.gif` | GIF | Actual unigram, bigram, combined and character features for the same text. |
| `28_ocr_similarity_comparison.png` | PNG | `document` versus `docurnent` with shared `retrieval`; partial character overlap only. |
| `29_query_ranking_comparison.png` | PNG | Five actual lexical methods; method-specific score axes. |
| `30_term_score_contributions.png` | PNG | Normalized feature products and BM25 per-term contributions sum to each score. |
| `31_inverted_index_search.gif` | GIF | Posting lookups, union candidates, actual scores and top-k; counts are not latency. |
| `32_bm25_vs_tfidf_ranking.png` | PNG | Repetition and filler alter two rankings; score magnitudes cannot be compared directly. |
| `33_bm25_saturation_3d.html` | HTML | TF × relative length surface, adjustable k1/b; average length=100 and valid TF/length domain. |
| `34_bm25_length_normalization.gif` | GIF | Vary b at fixed TF and relative lengths; length effect disappears at b=0. |
| `35_vocabulary_mismatch.png` | PNG | Synonym/OCR pairs: exact versus partial lexical overlap, no semantic scores. |
| `37_corpus_drift_animation.gif` | GIF | Frozen versus refitted statistics, feature weights and rankings; unseen vocabulary stays absent from frozen state. |
| `38_hybrid_retrieval_pipeline.png` | PNG | Actual word/character candidates into RRF and hypothetical context selection; no LLM generation. |
| `39_reciprocal_rank_fusion.gif` | GIF | One-based rank contributions accumulate from two real lexical lists. |
| `40_hybrid_retrieval_comparison.png` | PNG | Two lexical methods versus their rank fusion; no lexical/dense comparison. |
| `41_retrieval_evaluation.png` | PNG | Macro Precision@3, Recall@3, MRR and nDCG@3 under fixed synthetic judgments. |
| `42_per_query_failure_analysis.png` | PNG | Query-level recall failures, including vocabulary mismatch; labels are not tuned to scores. |

Asset 36 (lexical versus learned semantic similarity) is **not implemented or
executed**: no sentence-transformer model is configured or downloaded. There is
no placeholder semantic score. The rest of Level 3 uses the local dashboard,
two actual lexical rankings, RRF and the judged educational fixture.

## Mathematical contracts

Word analysis uses lowercased Unicode tokens of at least two characters, no
stop-word removal, and no stemming. Character TF-IDF uses `char_wb`, 3–5-grams.
Legal identifiers `SYN-204` and `SYN-205` are fictional. Default punctuation
splitting is deliberately visible; this is not exact structured-field filtering.

The three-axis corpus contains only `machine`, `learning` and `data`, in that
explicit column order. SVD instead fits 36 documents across six code-defined
categories using `random_state=64`, three components and ten iterations.
The projected query uses each document-fitted transform; variance ratios are
displayed. Reduced-space distances must not be substituted for original cosine.

BM25 uses positive IDF and one contribution per distinct query term:

$$
IDF_{BM25}(t)=\log\left(1+\frac{N-DF(t)+0.5}{DF(t)+0.5}\right),
$$

$$
s(q,d)=\sum_{t\in\operatorname{unique}(q)}IDF_{BM25}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/avgdl)}.
$$

Defaults are $k_1=1.2$, $b=0.75$. Counts, DF, token lengths and average length
share the same analyzer. Empty documents are included in this teaching corpus's
$N$ and average length. The implementation requires $k_1>0$; it does not reproduce
Lucene's norm compression, index behavior or all supported boundary settings.
BM25 agreement with a production engine is not claimed.

Cosine helpers assign zero to pairs involving zero rows; interactive geometry
instead displays mathematical cosine/angle as undefined when a magnitude is
zero. The retrievers return no zero-score hits. They retain input-order tie
breaking by document ID. Contributions sum to cosine after both vectors are
normalized, or to BM25 after term weighting. No score is a relevance probability.

RRF uses $\sum_m1/(60+r_m(d))$ over word/character top-5 lists, with one-based
ranks and zero contribution for missing documents. Context selection takes the
top three fused IDs. It is a lexical-fusion fallback, not a neural hybrid claim.

Evaluation uses 14 documents and eight queries. Unlisted grades are zero;
positive grades define binary relevance, and nDCG uses $2^{grade}-1$ gain.
Precision@3 always divides by three even when fewer documents are returned.
Recall divides by all relevant IDs, MRR scans the full positive ranking, and
nDCG@3 uses the ideal ordering over the fixed judgments. Macro means weight
each query equally. RRF is limited to its fused candidate pool.

## Experiments and evidence boundaries

The dated JSON stores actual section results and a hypothesis/configuration/
interpretation/limitation envelope. The concrete questions and assumptions above
define each visual experiment. Particular hypotheses include reduced marginal
TF contribution under saturation, falling IDF as prevalence grows, invariance
of cosine under positive scaling, and changes in lexical overlap under n-grams.
These are mechanism checks on chosen fixtures, not estimated population effects.

### Executed visual record — 2026-11-24

The reviewed complete run is `outputs/2026-11-24-reviewed-v5/`, run ID
`20261124T152438196116Z`. Its `results/experiments-20261124T152438196116Z.json`
and `CATALOG.md` retain all 18 section results and 41 structurally readable
assets: **19 PNGs, 12 GIFs and 10 offline HTMLs**, totaling 51,361,224 bytes.
Every raster frame was decoded successfully. This is artifact storage, not a
download, runtime or memory benchmark. Earlier complete/focused generation
records remain in their own directories; they are not pooled with this run.
The first drift fixture differs from the reviewed one.

The actual generation command used the local stable interpreter in place of
`python`, with native numerical threads limited to one:

```powershell
python 05-transformers-llms/64-bow-tfidf-text-similarity/visualizations.py --all --output-dir 05-transformers-llms/64-bow-tfidf-text-similarity/outputs/2026-11-24-reviewed-v5
```

Environment: Windows, Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1, Matplotlib 3.11.2, Plotly 6.9.0 and Pillow 12.3.0.
Streamlit 1.67.0 was used for the separately exercised dashboard.
The reviewed full generator completed without warnings or failed sections.
Earlier layout warnings motivated a corrected, separately dated render.

Selected observations from that run:

| Mechanism | Observed result | Interpretation candidate / limitation |
|---|---|---|
| NumPy TF-IDF parity | Largest absolute difference from scikit-learn: $1.11\times10^{-16}$ across raw/sublinear TF and normalized/unnormalized matrices | Supports these numerical configurations only; author review required. |
| Array storage | Dense/CSR bytes: 4,000/644 for 10×50; 640,000/19,524 for 80×1,000; 12,000,000/109,204 for 300×5,000 | Conditional on float64 values, observed index dtypes and chosen density; excludes allocator/object/peak memory. |
| OCR pair | Word unigram cosine 0.500000; bigram 0; combined 0.214885; character 0.588067 | Partial character overlap increased here; no general OCR or relevance gain follows. |
| Repetition/filler ranking | TF-IDF: IDs 0,1,3,4,2; BM25: 1,0,3,4,2 | The top pair changes under these exact documents and parameters; scores have different scales. |
| Drift | Top pair changes from 0,1 at N=3 to 1,0 at N=5; final alpha/beta IDF at N=9 is 1.105361/1.916291 | Refit changes relative weights; this designed fixture does not determine refresh policies. |
| Projection | First two SVD components explain 0.205806 for counts and 0.091847 for TF-IDF | These views retain little variance; low-dimensional distances are particularly unsafe substitutes for original retrieval scores. |

The original six-document analytical experiment and its November 24 record
remain separate from these visual protocols.

### Judged educational evaluation

Results below use the fixed eight-query task without tuning judgments to the
scores. Grade choices and interpretation remain pending author review.

| Method | Precision@3 | Recall@3 | MRR (full positive ranking) | nDCG@3 |
|---|---:|---:|---:|---:|
| Binary cosine | 0.458333 | 0.812500 | 0.875000 | 0.853279 |
| Count cosine | 0.458333 | 0.812500 | 0.875000 | 0.853279 |
| Word TF-IDF | 0.458333 | 0.812500 | 0.875000 | 0.853279 |
| Character TF-IDF | 0.416667 | 0.750000 | 0.906250 | 0.831559 |
| BM25 | 0.458333 | 0.812500 | 0.875000 | 0.853279 |
| Lexical RRF | 0.458333 | 0.812500 | 0.906250 | 0.853279 |

All methods have Recall@3 zero for `refund`. Character overlap can retrieve a
relevant reimbursement document later in its full ranking, which affects MRR
without fixing Recall@3. All methods miss one judged automotive synonym among
their top three. Character retrieval also loses a relevant RAG candidate in
the top three. Inspect the per-query record rather than treating macro scores
as a universal ordering of methods. No confidence interval, held-out query
generalization or production benchmark is established.

### Interactive verification

Chrome 158.0.8127.37 loaded all ten HTML pages offline and passed 25 checks,
including real angle/magnitude/DF/length controls, zero-vector handling, numeric
surface agreement, vocabulary-vector normalization, mouse rotation, BM25 b=0
and the count/TF-IDF SVD menu. No page/console errors or external HTTP(S)
requests occurred. Ten Canvas2D readback-performance warnings were observed.
WebGL reported ANGLE with Intel HD Graphics 630 / Direct3D11; this is not a
GPU performance benchmark or a cross-browser validation.

The local Streamlit server's agent API produced 12 ready snapshots across all
methods, no-results queries, BM25 parameters, sparse storage and evaluation.
Six additional browser checks exercised actual query entry, method/view
selection, desktop rendering and a selected 430×900 mobile storage view.
The dashboard had no JavaScript/console errors or external page requests.
Representative static plots and first/middle/final GIF states were inspected;
this is distinct from the structural decoding of every frame.

Numerical, application and export tests were executed separately from the full
generator: **61 passed in four isolated files**, without skips or warnings.
The canonical syntax check parsed 454 public Python files; internal-link checks
passed for 1,644 links in 284 Markdown files. Dependency compatibility passed
for 81 installed distributions. `git diff --check` passed with the module
index's LF-to-CRLF conversion warning. All checks used the stable interpreter:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/64-bow-tfidf-text-similarity
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
uv --no-cache pip check --offline --python .local/environments/n09-full-2026-11-13/Scripts/python.exe
git diff --check
```

The checks do not certify Linux, every mobile control, accessibility,
GitHub math rendering or scientific author approval. Static type checking and
repository-wide tests/CI were not run for this scoped delivery.

## Preview candidates and publication

Recommended review set: `03_tfidf_transformation.gif`, the browser view of
`12_tfidf_weight_surface_3d.html`, `18_cosine_similarity_animation.gif`,
`21_document_similarity_heatmaps.png`, and `29_query_ranking_comparison.png`.
They answer distinct weighting, surface, geometry, document and query questions.
The four PNG/GIF candidates total 365,117 bytes; the standalone surface HTML
is 4,863,743 bytes including Plotly. They remain within the repository's default
per-file preview targets without removing labels or changing animation timing.
All remain ignored pending the author's final publication decisions. Root
READMEs and the ignore policy are unchanged in this phase; no binary preview
was selected or published. Existing root progress edits from the analytical
phase are preserved.

Tests live in [test_visual_core.py](tests/test_visual_core.py),
[test_visual_exports.py](tests/test_visual_exports.py) and
[test_visual_app.py](tests/test_visual_app.py). Rendering tests use temporary
directories; they do not overwrite historical runs. Automated tests and browser
inspection do not approve interpretations on the author's behalf.
