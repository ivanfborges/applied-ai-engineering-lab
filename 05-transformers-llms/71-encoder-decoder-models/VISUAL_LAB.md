# Day 71 visual laboratory

The laboratory asks which information can reach each output, how the targets
change across objectives, and which computations decoding can reuse. It runs
small synthetic CPU experiments; it does not download models or evaluate
pretrained language capabilities.

## Run and open the results

From the repository root, install the canonical extras if needed:

```sh
python -m pip install -e ".[dev,deep-learning]"
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --quick
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --section attention
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --section perturbation
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --section geometry
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --section complexity --no-gif
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --quick --show
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --quick --output-dir outputs/day71_my_run
```

Sections are `architecture`, `attention`, `perturbation`, `geometry`,
`objectives`, `generation`, `cache`, `complexity` and `rag`. Attention includes
the dedicated cross-attention artifacts. Default behavior generates all nine.
Quick mode reduces raster resolution, generation length, cache length and
surface grid size; it retains every section and all 31 named visual outputs.
`--no-gif` intentionally skips animations. Pillow exports GIFs without FFmpeg.

The default directory is a **new dated folder** inside the topic's ignored
`outputs/`. A custom directory is relative to the current working directory
and must be empty. Existing runs are never overwritten. Each run contains
`index.html`, `manifest.json`, `numerical_data.json` and its generated files.
Open `index.html` or any interactive HTML directly in a browser; no server or
internet connection is required. Each interactive page embeds Plotly, which
increases file size. `--show` opens the gallery and a PNG in local viewers.

Code is split by responsibility:
[visual_experiments.py](visual_experiments.py) computes numerical fixtures;
[visual_figures.py](visual_figures.py) renders PNG/GIF;
[visual_html.py](visual_html.py) embeds offline controls and data;
[visualizations.py](visualizations.py) coordinates exports and validation.
The numerical core reuses [from_scratch.py](from_scratch.py) and the existing
causal-mask adapter. No dependency or environment change is required beyond
the repository's existing packages.

## How to interpret the visualizations

Green denotes encoder operations; blue denotes decoder operations; purple
denotes cross-attention; orange marks interventions or selected/generated
tokens; gray marks inaccessible positions or reference input points.

Read a visibility mask before interpreting attention probabilities. A zero
forced by masking is structurally different from a small probability on an
accessible key. Availability permits an effect; it does not guarantee a large
effect. Hover values are computed numbers, not linguistic explanations.

The architecture pipeline and RAG generation/checking stages are conceptual
diagrams. Attention, perturbations, toy losses, generation and cache equality
are computed synthetic experiments. Complexity surfaces and memory curves
are theoretical calculations. **No empirical model-quality or serving
benchmark is included.** Interpretations remain subject to author review.

## Artifact map and legitimate conclusions

Names below refer to local generated files, not public Markdown links into
ignored directories.

| Group and files | Operation and what changes | What the display supports | What it cannot establish |
|---|---|---|---|
| Architecture: `architecture_comparison.png`, `architecture_information_flow.gif` | A component is highlighted through embedding, position, attention, residual, normalization and FFN steps. Random causal/conditional blocks also supply actual output-projection probabilities and greedy continuations. The encoder returns contextual states. | Distinguishes the source encoder/memory branch from target cross-attention; decoder-only has no cross-attention. | Faithful BERT/GPT/T5 parameterizations, useful language generation or translation. The Portuguese sentence is a reference, not a generated translation. |
| Attention: `attention_masks.png`, `attention_logits.png`, `attention_probabilities.png`, `attention_masking_process.gif`, `interactive_attention_inspector.html` | Shared source embeddings and Q/K/V projection matrices produce scores, visibility, additive masked scores and row softmax. Select mechanism and query. | Future causal keys have exactly zero probability; cross-attention is rectangular; raw scores and normalized weights differ. Gray × marks blocked cells. | Attention weights as model reasoning or model-quality evidence. A full mask does not force uniform attention. |
| Cross-attention: `cross_attention_heatmap.png`, `cross_attention_information_flow.gif`, `interactive_cross_attention.html` | Decoder input Q reads encoded English source K/V. Advance the target query; connection widths/opacity follow computed probabilities. Inspect Q, every source K/V and the weighted output. | A 7-by-6 matrix maps seven decoder-input positions to six source positions. Output is the weighted sum of source values. | Learned bilingual alignment, translation accuracy or causality in human language. |
| Perturbation: `token_perturbation.png`, `token_perturbation.gif`, `causal_invariance.png` | Change only feature 0 of the final source input by +10; compare full-state matrices and L2 changes. Separately perturb the last target input with source memory fixed. GIF frames interpolate measured endpoints and include a token-position strip. | Earlier causal outputs stay invariant within tolerance; bidirectional and source-conditioned outputs may change. | A universal minimum effect on accessible positions, sensitivity across seeds or linguistic importance. Interpolated GIF frames are not new model executions. |
| Geometry: `hidden_representation_geometry.html`, `hidden_representation_perturbation.html`, `hidden_representation_evolution.gif` | Fit one PCA to all three families' input embeddings, baseline outputs and source-perturbed outputs. Select architecture/state set; rotate the view. The two pages default to context and intervention views respectively. GIF frames interpolate projected endpoints. | Common projection coordinates; visible input-to-context and baseline-to-intervention movements. Hover reports the original 16D displacement. | Full distance preservation, semantic clusters, learned token relationships or comparable model quality. Independently initialized layers do not acquire semantic coordinate alignment merely by sharing PCA. |
| Objectives: `masked_language_modeling.gif`, `causal_language_modeling.gif`, `span_corruption.gif`, `pretraining_objectives_comparison.png` | Toy attention plus a fixed random vocabulary head supplies probabilities and NLL. Highlight prediction positions, loss support, visibility and shifted labels. CLM also plots a computed unmasked control. T5-style spans become sentinel-delimited source and target sequences. | Distinguishes corruption from attention masking, selected-token losses from full target losses, and source conditioning from target causality. | Reproduction of original pretraining, learned reconstruction, comparable perplexity across objectives or a guaranteed loss decrease after removing the mask. |
| Generation: `autoregressive_generation.gif`, `generation_probability_distributions.html` | A fixed random bigram table supplies next-token logits. Select greedy/sampling, temperature and generation step. Append actual selected tokens until EOS or the configured limit. | Temperature changes vocabulary distributions; greedy argmax is invariant to positive temperature; sampled prefixes can diverge. | Trained GPT behavior, translation, factuality or attention-temperature experiments. |
| Cache: `kv_cache_generation.gif`, `kv_cache_computation.png`, `kv_cache_memory.html` | Prefill K/V from a predetermined synthetic token stream, then append one token. Compare exact attention with prefix recomputation and count cumulative projected token-vectors. Configure theoretical layers/KV heads/head size/precision/batch separately. | Reusing projected K/V preserves this single-head output; storage grows with prefix length and declared dimensions. | Wall-clock speedup, actual allocator memory, full multi-layer cached generation or elimination of attention over prior tokens. |
| Complexity: `attention_complexity.png`, `attention_complexity_log_scale.png`, `attention_complexity_3d.html` | Evaluate permitted-connection formulas against source/target lengths. Select individual or combined 3D surfaces and move n/m markers. | Source and target lengths affect attention types differently; the third dimension is connection count. | FLOPs, measured runtime, memory allocation or equivalent generation workloads. Encoder-only does not generate the target. |
| RAG: `rag_architecture_pipeline.png`, `rag_information_flow.gif`, `retrieval_and_reranking.html` | Independent lexical count vectors give cosine retrieval; an explicit pair rule changes candidate ordering. Vary displayed candidate count. Generation and validation remain conceptual. | Explains document-vector reuse and why joint scoring is normally applied to a selected candidate set; separates cross-encoder scoring from encoder-decoder cross-attention. | A trained bi-encoder/cross-encoder, semantic retrieval quality, a working LLM, validated answers or production scalability measurements. |

## Numerical contracts and simplifications

All attention probabilities use the same educational NumPy core. For each
query, it computes projected Q/K/V, $QK^\top/\sqrt{d_k}$, additive masking,
stable row-wise softmax and multiplication by V. Self-attention comparisons
share input representations and projections; cross-attention changes the
origin of Q versus K/V. Probability scales are fixed to $[0,1]$.
Mask transitions explicitly show raw scores, binary visibility, masked scores
and normalized probabilities. Raw logits remain inspectable even when their
positions are subsequently masked.

NumPy and functional PyTorch SDPA use Boolean **True = permitted**. PyTorch
Transformer modules use Boolean **True = blocked**. The adapter explicitly
inverts the convention. Fully blocked rows are rejected by the educational
core; unsupported rectangular cached causal masks require explicit offsets
and are not silently accepted. The cache experiment's last query sees only
the available prefix, so its rectangular all-visible mask has no future keys.

The MLM fixture deliberately selects three of six positions and forces one
mask/random/unchanged case each to make the distinction visible. It is not
a sample from BERT's original 15% selection and 80/10/10 replacement process.
Only selected prediction positions contribute to its displayed mean loss.
The CLM fixture uses shifted inputs/labels and a causal mask. Removing that
mask permits position $i$ to read the input token at $i+1$, which is its label;
the untrained toy loss need not improve. Training rows are computed in parallel;
the separate generation example selects unknown tokens sequentially.
Span corruption uses two manually selected nonadjacent spans, unique sentinels
and a shifted reconstruction target; no tokenizer, NSP or pretraining runs.

The cache experiment uses one layer/head, key/value width 8 and float64.
Inputs include identical absolute sinusoidal positions in both paths.
At each step the uncached path recomputes all prefix K/V; the cached path
projects only new entries. Query computation and attention-score work are
not included in the projected-token count. The token stream is fixed rather
than generated by an LM. Logical cache storage is $2Ld_kb$ bytes for prefix
length $L$, width $d_k$ and $b$ bytes per element. The separate calculator uses:

$$
\text{KV bytes}=2\,B\,N_{\mathrm{layers}}\,N_{\mathrm{KVheads}}\,d_{\mathrm{head}}\,L\,b.
$$

The complexity surfaces count one instance of each relevant attention type:

$$
C_e=n^2,\qquad C_d=\frac{(n+m)(n+m+1)}{2},\qquad
C_{ed}=n^2+\frac{m(m+1)}{2}+nm.
$$

Depth, FFNs, hidden widths, cache policies and hardware are intentionally
separate. Triangular visibility does not guarantee that dense kernels perform
half the work.

RAG uses six code-defined documents and query `bank loan approved`. Its toy
pair rule is $0.4\times\text{cosine}+0.4[\text{bank AND loan}]+0.2[\text{approved OR approval}]$.
All six toy scores are precomputed to support the offline candidate selector;
only the selected candidates are displayed/reranked. This is not a timing
experiment or a claim of learned relevance. There are no external datasets.

## Validation and dated runs

Run focused checks without touching previews or historical outputs:

```sh
python scripts/validate_repo.py tests --scope 05-transformers-llms/71-encoder-decoder-models
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

The tests in [tests/test_visual_lab.py](tests/test_visual_lab.py) check masks,
shapes, weighted values, shared projections, causal invariance, actual-layer
trace parity, target shifts/loss support, sentinels, shared PCA, cache equality,
theoretical formulas, seeded generation and the declared retrieval rule.
They also compare actual plotted heatmap arrays and embedded HTML payloads to
computed values, decode a real multi-frame GIF, export real PNG/HTML files in
a temporary directory, and verify refusal to overwrite an existing run.

The generator validates 15 numerical assertions before export. It then
decodes every PNG and every GIF frame, checks that GIFs have multiple distinct
frames, and checks interactive HTML for embedded Plotly plus computed data.
These structural checks do not replace inspection of labels, browser controls
or scientific interpretation. New run details and browser scope are recorded
below after execution. The analytical seed-71 record in [notes.md](notes.md)
remains unchanged and separate from this seed-271 visual protocol.

### Executed visual record — 2026-11-25

**Hypothesis:** causal prefixes are protected from future-input changes;
visible source information can reach conditional targets; identical positional
inputs and fixed projections give equivalent cached and recomputed attention.
All interpretations below remain **author review required**.

The final full quick run used:

```sh
python 05-transformers-llms/71-encoder-decoder-models/visualizations.py --quick --output-dir 05-transformers-llms/71-encoder-decoder-models/outputs/visual_20261125_quick_v3
```

Its manifest records UTC `2026-11-25T20:51:58.573787+00:00`, seed 271,
Python 3.11.14, NumPy 2.4.6, PyTorch 2.15.0+cpu, Matplotlib 3.11.2, Plotly 6.9.0
and Pillow 12.3.0 on Windows. Numerical models use CPU float64, one thread,
width 16, four attention heads, FFN width 48, one layer per stack, Post-LN,
zero dropout, evaluation/inference modes and no optimizer updates.
Six English source tokens and seven decoder-input positions include shared
synthetic embeddings and absolute sinusoidal positional information. A
separate random output head generates short untrained greedy continuations.
These settings differ from the historical analytical seed-71 run.

**Observed results:**

| Check or experiment | Result | Interpretation boundary |
|---|---|---|
| Exports | 12 PNGs, 11 GIFs, 8 standalone HTMLs; zero export failures | Successful encoding/decoding is not scientific review. |
| GIF inspection | Respectively 16, 4, 7, 6, 11, 3, 7, 7, 5, 8 and 7 frames in architecture, masks, cross-attention, perturbation, geometry, MLM, CLM, spans, generation, cache and RAG | All frames decoded; inspected representative frames assess local readability, not every possible viewer. |
| Generator checks | 15/15 passed with tolerance $10^{-12}$ | Small deterministic fixtures, not exhaustive numerical coverage. |
| Source change → earlier causal states | Maximum L2 difference exactly 0 | Causal visibility is correctly enforced for this intervention. |
| Target change → earlier conditional target states | Maximum L2 difference exactly 0 | Source memory stays fixed in this separate control. |
| Source change → bidirectional states | L2 `[1.387226, 1.022269, 0.612996, 1.970372, 0.729146, 4.455310]` | Visible paths allowed changes; no universal positive lower bound. |
| Source change → conditional target states | L2 `[0.524192, 0.459333, 0.539022, 0.534057, 0.491103, 0.558657, 0.444474]` | Source sensitivity does not demonstrate translation or source fidelity. |
| Manual operation traces versus official PyTorch layers | Maximum absolute difference exactly 0 for all three stacks | Supports the displayed Post-LN decomposition at these inputs and parameters. |
| Cached versus prefix-recomputed attention | Maximum absolute difference $2.220446\times10^{-16}$ | One-layer single-head numerical mechanism, not a complete cached language model. |
| K/V projection count at length 10 after prompt length 3 | 104 projected token-vectors uncached versus 20 cached; logical float64 cache 1,280 bytes | Counts projections of K and V separately; no timing or allocator measurement. |
| Shared PCA | First three variance fractions approximately 0.261542, 0.159890, 0.147806; sum 56.92% | Variance of the combined fitted fixture, not semantic structure or full-distance preservation. |
| Toy MLM mean selected-position NLL | 2.415810 nats | Random toy head, dense forced corruption fixture; not a language-model benchmark. |
| Toy retrieval/reranking | Retrieved D0 → D3 → D1 → D2; reranked D0 → D3 → D2 → D1 | Result of the declared lexical/pair rule; no ground-truth relevance evaluation. |

The complete numerical arrays and per-file metadata remain in the ignored
run directory. Earlier `quick_v1` and `quick_v2` galleries are retained as
development records; the final record above uses the corrected layout and
declared approval-term pair rule. They are not pooled into a benchmark.
An additional `--section attention --quick --no-gif` layout check produced six
nonanimated artifacts and intentionally skipped two GIFs. The non-quick
full-resolution mode and `--show` viewer launching were not executed.

The final scoped suite passed **67 tests** across two isolated files (43
analytical, 24 visual), without skips or warnings. Real export tests write to
temporary directories and do not regenerate the four public previews.
Repository checks also passed: 516 Python files parsed, 1,919 local links
checked across 319 files, compatibility of 81 installed packages checked with
`uv --offline --no-cache pip check --python` against the stable interpreter,
and `git diff --check`. These are local checks, not the complete numerical CI
suite. The EN/PT root READMEs and module index were left byte-for-byte
unchanged during this visual phase, as requested pending author review.

### Browser validation — 2026-11-25

Playwright CLI 0.1.24 with Headless Chrome 158.0.8127.37 exercised all eight
HTMLs at 1440×1000 over a loopback-only HTTP server. The 152 scenarios included
all 19 attention query/family combinations, seven dedicated cross queries,
state/family selection in both PCA pages, actual mouse rotation in both 3D
views, greedy/sampling rollouts at all four temperatures, cache formula parity
and invalid inputs, individual/combined complexity surfaces and six candidate
counts. Rendered probabilities were compared to embedded numerical payloads;
cache and connection formulas were checked against independent arithmetic.

The completed audit had **no page/JavaScript errors or external requests**.
Three Canvas2D readback-performance warnings were observed. The loopback
wrapper answered favicon requests locally; an earlier first load had a
favicon-only HTTP 404. Initial sandboxed Chrome launches failed before page
load because its graphics process could not start. The completed review used
an isolated headless process outside that execution sandbox, without changing
global browser settings. Its renderer reported Intel HD Graphics 630 through
ANGLE/Direct3D11; numerical experiments themselves remained explicitly CPU.

The CLI blocks navigation to `file:` URLs, so direct file-protocol opening was
not browser-tested. Each HTML embeds its library and data and performs no
runtime fetches; the audit verified the local standalone content over loopback.
This does not establish software-WebGL compatibility, native-GPU performance,
mobile/cross-browser/accessibility coverage, GitHub rendering, production
readiness or scientific/author approval. Browser logs, screenshots and the
inspection harness remain ignored local working material.

## Preview policy

Complete HTMLs, GIFs, PNGs and numerical records stay ignored under `outputs/`.
A maximum of four inspected PNG/GIF previews may be copied intentionally to
`assets/` and referenced from the topic README. Preview curation and successful
rendering do not approve the interpretations on the author's behalf.

The inspected set copied without alteration from the final quick run is
`architecture_comparison.png`, `attention_probabilities.png`,
`causal_invariance.png` and `hidden_representation_evolution.gif`. The
[topic README](README.md#selected-visual-previews) references these deliberate
Git candidates in `assets/`; scoped ignore rules exclude other assets.
