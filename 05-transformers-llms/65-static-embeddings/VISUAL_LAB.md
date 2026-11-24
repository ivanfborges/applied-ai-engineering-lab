# Static embeddings: visual laboratory

This is a separate visual phase of Day 65. The original 32-sentence analytical
experiment and its dated [record](notes.md#execution-record--2026-11-24) remain
unchanged. Visual runs use a larger, separately generated synthetic corpus and
their own timestamped configurations. Interpretations require author review.

## Run a question at a time

From the repository root with stable Python 3.11+ and the canonical dependencies:

```powershell
python -m pip install -e ".[dev]"
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo all --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo all --quick --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo cbow --seed 42 --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo negative_sampling --quick --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo glove --epochs 400 --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo retrieval --no-show
python 05-transformers-llms/65-static-embeddings/visualize_embeddings.py --demo experiments --extended --no-show
```

The commands are usage examples; the execution record below identifies which
were actually run. No cloud API, pretrained model, external corpus, GPU,
Gensim, PyTorch or FFmpeg is required. NumPy handles training; pandas exports
measurements; Matplotlib/Pillow export images; scikit-learn supplies PCA and
TF-IDF; Plotly is bundled inside offline HTML. Dependencies remain in the root
[pyproject.toml](../../pyproject.toml), with `dev` adding pytest.

`--demo` accepts `representations`, `window`, `cbow`, `skipgram`,
`negative_sampling`, `pmi`, `glove`, `cosine`, `analogies`, `polysemy`,
`experiments`, `retrieval` or `all`. A selected demo fits only the models it
needs. `--corpus-size`, `--repetitions`, `--epochs`, `--window`, `--dimension`,
`--seed` and `--output-dir` control the run. Geometry exports require dimension
at least two. `--quick` uses 50 extra common-word sentences and 80 epochs;
default uses 180 and 240; `--extended` uses 500 and 400 and adds a third sweep
seed. Explicit size/epoch flags override those defaults. `--no-show` makes the
headless intention explicit; exports always use the nonblocking Agg backend.

Every invocation creates a new UTC timestamp subdirectory under the output
directory, avoiding replacement of historical runs:

```text
outputs/<timestamp>/
  figures/     PNG figures
  gifs/        Pillow GIFs
  html/        standalone offline pages
  results/     run.json, hyperparameters.csv, model snapshot NPZ files
```

Open HTML directly in a browser; no server is needed. Each file embeds Plotly
and has no external script dependency. Word selection and labels work locally;
3D plots support rotation, zoom and category legends. Browser verification is
distinct from code/export verification and scientific interpretation.
The generated `index.html` is a lightweight navigation catalog linking the
individual artifacts without loading every animation at once.

## Corpus and experimental protocol

[embedding_lab.py](embedding_lab.py) generates grammatical template variants for
healthcare, technology, transport, geography, royalty, food, finance, river
contexts and shared-context antonyms. Lowercase tokens include `brasilia` rather
than an accent-preserving city token. Support sentences retain the evaluation
vocabulary before random template draws. `surgeon` and `architect` occur once
per generated collection; repetitions duplicate that collection and raise their
counts without creating independent evidence.

The size option controls **additional common-template draws**, not final
sentence count. Bank-specific sentences are added separately. In the sense
experiment the financial ratio controls those additional bank sentences;
fixed support still includes both senses at ratios zero and one. This prevents
vocabulary dropout but makes the requested ratio different from the total
corpus sense fraction. Read actual frequencies and configuration in `run.json`.
Every frame of the frequency GIF is an independently trained condition, not a
training epoch or an interpolated trajectory. Fits share vocabulary and initial
seed. A common PCA does not align independently fitted semantic spaces: compare
each model's own cosine neighborhoods and avoid interpreting absolute cross-fit
displacements as a general semantic measurement.
Synthetic category membership and relevance judgments are deliberately designed
diagnostics, not standardized NLP benchmarks or representative language data.
Category purity averages over represented, categorized source words; every
vocabulary entry still competes as a neighbor. `bank` is assigned the finance
category for this diagnostic, although its contexts deliberately contain both
senses. Surface-form vocabulary coverage is measured rather than assumed.

## Visual questions and output catalog

All filenames below are regenerable outputs. They are not Markdown links to
ignored local artifacts. Selected Git candidates are identified separately.

| Outputs | What to inspect | Valid inference and boundary |
|---|---|---|
| `01_one_hot_vs_dense_embeddings.png` | Matrix values, shown/full shapes, sparsity and cosine contrast for the same words | One-hot identities are orthogonal. Random dense similarity is not learned semantics. |
| `02_sliding_context_window.gif` | Actual center/context occurrences, CBOW target and directed Skip-gram events | Fixed symmetric radius clips at sentence boundaries. Dynamic random-radius windows are not used. |
| `03_cbow_architecture.gif`, `04_skipgram_architecture.gif` | Side-by-side selection, lookup, hidden state, logits, full softmax, mean NLL, both gradients and simultaneous updates | All displayed numbers come from a two-dimensional four-word forward/backward pass. This is full softmax, not SGNS. |
| `05_negative_sampling_dynamics.gif` | Actual two-table SGD, positive/noise scores, stable losses, gradient and parameter displacement, cosine and norms | The direct objective changes dot products. No monotonic cosine or distance claim is made. The 2D trace is a small controlled update fixture. |
| `06_negative_sampling_loss.png`, `07_negative_sampling_distribution.png` | Logistic loss/derivative signs; unigram versus exponent-0.75 mass | Noise is not a semantic incompatibility label. Shown frequency bars are a subset; full distributions normalize over all words. |
| `43_sgns_embeddings_3d.html`, `44_sgns_training_evolution.gif` | Full-corpus SGNS geometry, actual snapshots, word metadata and original-space neighbors | The same seed, corpus and epoch count do not give SGNS and full-batch softmax equal update budgets. Inspect both successes and failures; no general superiority is established. |
| `08_cbow_training_evolution.gif`, `09_skipgram_training_evolution.gif`, `10_cbow_embeddings_3d.html`, `11_skipgram_embeddings_3d.html` | Actual recorded vectors; initial, early and final states; final word norms, frequencies and original-space neighbors | One PCA is fitted across all animation snapshots and axes stay fixed. Final 3D pages fit PCA on that model's final vectors. Do not compare coordinates between independently fitted spaces. |
| `12_cooccurrence_matrix.png`, `13_conditional_probabilities.png`, `14_pmi_ppmi_heatmaps.png` | Weighted counts, observed logs, conditional distributions and PMI/PPMI | Marginals use the complete vocabulary, though plots show subsets. Unobserved PMI is masked, not assigned a finite fabricated value; PPMI conventionally truncates to zero. |
| `15_shifted_pmi_vs_learned_dot_products.png` | Prespecified 3×3 counts, theoretical association scores versus learned rank-two dot products | Unigram noise gives PMI minus log K. Smoothed noise uses the generalized target. This experiment optimizes the exact noise expectation, separately from sampled SGNS. |
| `16_glove_weighting_function.png`, `17_glove_matrix_reconstruction.gif`, `18_glove_reconstruction_residuals.png`, `19_glove_training_loss.png`, `20_glove_embedding_evolution.gif`, `21_glove_embeddings_3d.html` | Corpus/count/log/objective pipeline, capped weights, actual reconstructed states, residuals, own loss and learned geometry | Zero-count targets/residuals are masked. Optimized residuals are weighted; the residual heatmap is unweighted. Batch GD differs from original AdaGrad; final representation is center plus context. |
| `22_cosine_similarity_interactive.html`, `23_cbow_similarity_heatmap.png`, `24_skipgram_similarity_heatmap.png`, `25_glove_similarity_heatmap.png`, `42_similarity_comparison.png` | Change angle, both magnitudes and normalization; compare actual cosine matrices on [-1,1] | Positive scaling preserves cosine. Zero cosine is undefined for a zero vector; the explorer displays `null`. Scores are not calibrated semantic probabilities. |
| `26_idealized_vector_analogy.gif`, `27_learned_vector_analogy.gif`, `28_vector_analogies_3d.html` | Source vectors, relationship offset, translated query, expected fixture word and actual top candidates | Idealized vectors are explicitly constructed. Learned offsets use raw vectors and original-dimensional cosine, excluding sources. Failures remain visible. The engineer/technology/healthcare relation is exploratory with no correctness label. |
| `29_polysemy_embedding_space.html`, `30_polysemy_frequency_experiment.gif`, `31_antonym_similarity.png`, `32_static_vs_contextual_embeddings.png` | One `bank` row under changing additional sense frequencies, actual antonym cosine, order/negation controls | Bank need not lie halfway between categories. Antonyms may share contexts. The contextual pipeline is conceptual: no contextual encoder or fabricated contextual output is shown. |
| `33_architecture_comparison.png`, `34_window_size_comparison.png`, `35_embedding_dimension_comparison.png`, `36_negative_sampling_comparison.png`, `37_rare_word_experiment.png`, `38_quality_vs_computational_cost.png` | Architecture repeats; radius 1/3/5; dimensions 2/10/30/100; SGNS K=1/5/15; epochs, corpus size, rare words, time and raw table bytes | Category purity and two analogy answers are synthetic diagnostics. Architecture error bars are sample SD across seeds. Different objectives, event counts and optimizers prevent a general method ranking. |
| `39_semantic_retrieval_comparison.png`, `40_document_query_embedding_space.html`, `41_retrieval_failure_cases.png` | TF-IDF, mean Skip-gram, mean GloVe and IDF-weighted Skip-gram; six judged queries; original-space rankings and OOV abstention | Eight synthetic documents and manual grades do not establish a production retrieval advantage. Negation/polysemy failures should be inspected per query. |

## Mathematical choices

The analytical [kernels](from_scratch.py) are reused for full-softmax CBOW,
Skip-gram and GloVe. Optional recorder hooks copy pre-update parameter states,
including initialization and final parameters, without modifying the objective
or original analytical defaults. All gradients refer to the same old parameters.
The four-word architecture traces use radius one and learning rate 0.4; the
two-dimensional SGNS update fixture uses 20 steps and learning rate 0.25.
These small mechanism traces are separate from the full-corpus training runs.

SGNS maintains independent center/context matrices and minimizes

$$
L=\frac{1}{B}\sum_{b=1}^{B}\left[
\operatorname{softplus}(-v_{w_b}^Tu_{c_b})+
\sum_{k=1}^{K}\operatorname{softplus}(v_{w_b}^Tu_{n_{bk}})\right].
$$

Stable `logaddexp` avoids exponential overflow. The negative gradient with
respect to a dot product is $\sigma(z)$; the positive gradient is
$\sigma(z)-1$. Repeated center/context/noise IDs accumulate with `np.add.at`.
Negatives are sampled independently from normalized token counts raised to
0.75; collisions with positives are retained rather than silently changing the
noise distribution. SGNS uses shuffled batches of at most 256 events. Its
recorded epoch curve evaluates a fixed, separately seeded noise sample and is
labeled a diagnostic; it is not the stochastic training-batch loss.

The PMI control uses all-positive 3×3 counts and $K=5$. For noise $q(c)$ its
independent optimum is

$$
z^*_{wc}=\log\frac{p(c\mid w)}{Kq(c)}.
$$

Only $q(c)=p(c)$ yields $\operatorname{PMI}(w,c)-\log K$. The rank-two
factorization shares parameters and runs a finite optimization budget; equality
is not guaranteed. This control removes noise sampling variance by optimizing
its expectation. The actual corpus SGNS run retains sampling variability.
The fixed counts are `[[35, 8, 4], [5, 28, 10], [8, 3, 20]]`, dimension two,
1,200 batch updates, learning rate 0.8 and the main run seed. Noise probabilities
come from column marginals, using exponents one and 0.75 respectively.

Window counts are inverse-distance weighted for GloVe/PMI, whereas the
full-softmax predictive events use uniform context occurrences. Vocabulary,
dimension and seed are controlled where stated; these are different objectives
and effective optimization budgets. Never compare raw loss magnitudes across
learners. PCA explains projected variance, not semantic correctness.
Sweeps hold dimension 10 and radius 3 except for the factor being varied, and
use half the main epoch budget. Architecture comparisons repeat seeds 65/66
under the default seed; extended mode adds 67. The main corpus size/dimension
flags do not silently redefine those prespecified sweep baselines.

## Retrieval judgments and operational implications

The exact documents, queries and manual grades are inspectable in
`DOCUMENTS` and `QUERIES` in [embedding_lab.py](embedding_lab.py).
Recall@3 counts any positive grade as relevant; reciprocal rank uses the full
ranking; nDCG@3 uses gain $2^g-1$. Ties use document ID. OOV queries abstain and
receive zero metrics, including the deliberately unrepresented `kubernetes`
query. Zero pooled vectors are storage placeholders, not valid cosine vectors.

TF-IDF fits the document collection only. Static embeddings fit the separate
synthetic template corpus; document IDF supplies the optional pooling weights.
This is a small, explicitly transductive document-index example, with no tuning
on the manual query judgments. Metrics include the unanswerable OOV query.
Coverage must accompany any apparent retrieval gain.

The order control permutes exactly the same token multiset, so mean pooling is
identical up to floating-point roundoff. The compliance/negation contrast also changes `complied` to `comply`;
it cannot isolate negation as a causal factor. Neither cosine nor averaging
provides contradiction detection. Retrieval-trained contextual encoders can
condition on richer text signals, but suitability still requires representative
judgments and operational evaluation. No contextual model comparison was run.

## Verification and execution record

### Executed default visual run — 2026-11-24

Codex executed `--demo all --no-show` with the stable Python 3.11.14 environment.
The selected run is `20261124T194328510207Z` (UTC start); the first complete pass,
`20261124T192711196602Z`, remains a separate ignored record. The first pass
exposed caption accumulation and an uninformative matrix slice; a regression
test reproduced the caption issue before its fix. Corrected rendering does not
replace or relabel the historical analytical experiment.

**Hypothesis:** context objectives can fit their own observations, but useful
neighborhoods, analogy recovery and judged retrieval need separate evaluation.

**Configuration:** 301 synthetic sentences, 1,953 tokens, 113 vocabulary entries,
seed 65, dimension 10, radius three, 240 epochs/updates, learning rate 0.8,
float64. The corpus SHA-256 is
`a3fc80efe774e223ab9253987be0b0057c2f772fa98283f96762c5fce319b1f7`.
Numeric thread pools were explicitly limited to one for generation; this is not
a hard operating-system CPU/RAM cap. GloVe uses inverse-distance counts,
$x_{max}=10$, $\alpha=0.75$, batch GD and summed center/context vectors.
CBOW/Skip-gram retain input vectors and use full-batch softmax. SGNS uses K=5,
token-unigram exponent 0.75 and shuffled minibatches: 8,106 positive pairs yield
32 updates per epoch, **7,680 updates total**, versus 240 full-batch updates.
Equal epoch counts therefore do not imply comparable optimization budgets.

Versions: NumPy 2.4.6, pandas 3.1.1, Matplotlib 3.11.2, Pillow 12.3.0,
Plotly 6.9.0 and scikit-learn 1.9.1. These are execution versions, not a lockfile.

**Observed results:**

| Learner | Initial own loss | Final own loss | Synthetic purity@3 | Correct predefined analogies |
|---|---:|---:|---:|---:|
| CBOW full softmax | 4.727435 | 3.643951 | 0.098958 | 0/2 |
| Skip-gram full softmax | 4.727452 | 3.926063 | 0.151042 | 0/2 |
| GloVe weighted log-count residual | 2.447680 | 0.122415 | 0.109375 | 0/2 |
| SGNS fixed-noise diagnostic | 4.157093 | 1.553659 | 0.552083 | 2/2 |

Losses have different units and targets. For `king - man + woman`, the first
candidates were `apple`, `apple`, `wears` and `queen` in table order; for
`paris - france + germany`, they were `carries`, `carries`, `a` and `berlin`.
Unsuccessful neighbors/analogies were retained. SGNS success on two engineered
relations does not establish general analogy accuracy or method superiority.
Original-space category coverage was 64/66 declared entries; unseen surface
forms were not silently treated as represented.

In the first two-dimensional SGNS fixture update, the positive dot product rose
from **0.362559 to 0.465569**, while cosine fell from **0.999356 to 0.936268**.
This update includes positive and noise contributions. It concretely separates
the dot-product objective from angular similarity. The controlled rank-two PMI
fits had score RMSE **0.575962** for unigram noise and **0.576470** for exponent
0.75; finite shared representations did not recover every independent optimum.

Animation PCA retained approximately 64.0% variance for CBOW, 57.7% for
Skip-gram, 34.0% for GloVe and 33.0% for SGNS in two dimensions. Fixed axes and
one projection remove artificial per-frame axis changes; they do not recover
the variance discarded by projection.

| Retrieval method | Recall@3 | MRR, full ranking | nDCG@3 |
|---|---:|---:|---:|
| TF-IDF | 0.833333 | 0.833333 | 0.833333 |
| Mean Skip-gram full-softmax vectors | 0.583333 | 0.588889 | 0.522951 |
| Mean GloVe vectors | 0.833333 | 0.638889 | 0.688488 |
| IDF-weighted Skip-gram vectors | 0.750000 | 0.722222 | 0.629235 |

The fully OOV sixth query abstained in every method. The TF-IDF result reached
the five-answerable-query ceiling in this tiny fixture. This does not establish
performance on paraphrase-rich, multilingual or production retrieval.

**Interpretation candidates — author review required:** all learners fit their
own objectives, yet low losses and positive within-category contrasts did not
guarantee useful analogies. SGNS benefited from a different objective and many
more updates; this experiment cannot attribute the difference to architecture
alone. Static pooling gave cosine one to display precision for an order permutation and produced
cosine 0.973422 on the compliance/negation contrast, which also changes word
inflection. These diagnostics illustrate limitations, not contradiction
classification performance.

**Outputs:** 24 PNGs, 12 GIFs and eight standalone interactive HTML pages, plus
the navigation catalog, CSV/JSON measurements and actual parameter/vector
snapshot NPZ files. A presentation-only replay widened one matrix page,
separated vector labels and corrected a conceptual caption. Numerical
`run.json` was preserved; `presentation_review.json` records original/replayed
hashes and ignored backups. No model was rerun or selected based on an analogy
answer during that presentation replay.

The small full-corpus configuration and 22 controlled sweep fits were executed.
The optional extended CLI budget and a real contextual encoder were **not
executed**. No numerical experiment failed; initial rendering issues and a
browser resize-harness race were corrected and preserved as local audit
evidence. All interpretation candidates remain pending author review.

### Verification and preview selection

The final scoped run passed **74 tests** in three isolated files: 44 analytical,
27 visual numerical and three export/regression checks. No skips, failures or
warnings were emitted in that final test run. Real export tests use temporary
directories and do not replace public preview files.
Repository syntax parsing passed for **462 Python files**; internal-link checks
passed for **1,692 targets in 289 Markdown files**. An offline dependency check
verified **81 installed packages** as compatible. `git diff --check` passed,
with an existing LF/CRLF conversion notice for the unchanged module index.
Root EN/PT READMEs, module index, historical analytical notes, raw source and
the protected Day 64 files matched their visual-phase baseline hashes. The
complete curriculum suite and Linux CI were not run in this focused validation.

All 24 PNGs were verified, and all 137 frames across 12 GIFs were decoded with
positive durations and looping checked. Contact sheets covered every PNG and
representative initial/middle/final GIF states; selected frames were inspected
at full size. Saved vector snapshots were checked against their actual input
and output tables, and repeated main-run losses/analogy rankings matched the
first pass exactly. These are within-environment checks, not cross-platform
reproducibility or author scientific approval.

Chrome **158.0.8127.37** passed **31 browser checks** on the final eight
interactive pages and catalog: controls, zero-vector handling, original-space
neighbors, labels, source exclusions, real mouse rotation/zoom, OOV abstention
and selected mobile resizing. There were no page/console errors or external
HTTP(S) requests. Fourteen Canvas2D readback-performance warnings were recorded.
The observed renderer was ANGLE / Intel HD Graphics 630 / D3D11. Training and
image generation use CPU; browser 3D requires WebGL and was not benchmarked on
software fallback, other GPUs or other browsers. Dense 2D labels can overlap;
the 3D pages provide hover details and optional labels.

The six selected previews below were copied without re-encoding and their
hashes matched the inspected sources. Git candidate checks verified the exact
allowlist in the root ignore rules before documentation links were added.
All HTML, alternate figures, timestamped records and intermediate images remain
ignored. Total selected size is **1,492,620 bytes** (about 1.42 MiB), within the
[preview budgets](../../docs/visual-assets.md); no size exception is needed.

| Selected preview | Bytes | Learning question |
|---|---:|---|
| [Recorded SGNS evolution](outputs/previews/44_sgns_training_evolution.gif) | 335,389 | How do actual word vectors move under a fixed projection? |
| [Negative-sampling dynamics](outputs/previews/05_negative_sampling_dynamics.gif) | 623,535 | Why can a larger dot product coexist with lower cosine? |
| [GloVe reconstruction](outputs/previews/17_glove_matrix_reconstruction.gif) | 259,418 | Which observed cells constrain the weighted objective? |
| [Learned analogy](outputs/previews/27_learned_vector_analogy.gif) | 66,498 | What does a failed learned offset look like? |
| [Cosine comparison](outputs/previews/42_similarity_comparison.png) | 116,629 | How do the measured angular patterns differ? |
| [Polysemy frequency](outputs/previews/30_polysemy_frequency_experiment.gif) | 91,151 | What changes when additional bank-sense frequencies change? |

Run proportional checks from the root:

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/65-static-embeddings
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

[tests/test_visual_core.py](tests/test_visual_core.py) checks actual gradients,
distributions, traces, masks, PCA mapping and hand-calculated retrieval metrics.
[tests/test_visual_exports.py](tests/test_visual_exports.py) generates real
exports in temporary directories. Parsing an HTML file is not browser testing.

## Navigation proposal pending author review

After the author reviews the implementation and selected previews, update the
root EN/PT evidence description and existing module-index Day 65 row to mention
the visual lab, actual SGNS, recorded training, fixed PCA, judged synthetic
retrieval and its limits. Counts stay at 65 total studies and 3/15 for this
module. This visual phase does not apply that navigation change automatically.
No commit, push, deployment or production verification is part of this run.

Concrete wording proposed for the EN Day 65 evidence row:

> Tested full-softmax CBOW/Skip-gram, SGNS and GloVe; recorded training states,
> fixed-PCA animations, offline 3D, synthetic analogy/hyperparameter diagnostics
> and manually judged retrieval. Unequal update budgets; interpretation review pending.

Proposed PT evidence sentence:

> O Dia 65 inclui laboratório visual offline com SGNS real, estados de treino
> registrados, PCA fixa, reconstrução GloVe, analogias e recuperação sintética
> com julgamentos manuais. Os orçamentos de atualização diferem; a revisão das
> interpretações pelo autor permanece pendente.

Primary sources and the distinction between analytical and visual experiments
are in [references.md](references.md). Suggested larger-corpus, contextual-model
and extended-budget experiments remain suggestions unless separately executed.
