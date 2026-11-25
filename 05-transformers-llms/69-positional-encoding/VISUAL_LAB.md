# Positional Encoding — offline visual learning lab

[Topic overview](README.md) · [Mathematical notes](notes.md)

The lab makes thirteen positional questions inspectable through actual arrays,
rotations, projections and attention calculations. All content is deterministic
and synthetic. There are zero training updates, no pretrained models, no paid
services and no external requests. The random absolute-position table is
**initialization**, not a learned language-model representation.

## Run and explore

From the repository root, use Python 3.11+ in an isolated environment with the
canonical dependencies in [pyproject.toml](../../pyproject.toml). The visual
generator uses the shared runtime; `.[dev,deep-learning]` also supplies pytest
and the independent PyTorch numerical checks. No dependency changes or
installations are needed in the documented local reference environment.

```powershell
python 05-transformers-llms/69-positional-encoding/visualizations.py --list
python 05-transformers-llms/69-positional-encoding/visualizations.py --quick
python 05-transformers-llms/69-positional-encoding/visualizations.py --all
python 05-transformers-llms/69-positional-encoding/visualizations.py --demo rope --show
python 05-transformers-llms/69-positional-encoding/visualizations.py --demo sinusoidal --format html --save
python 05-transformers-llms/69-positional-encoding/visualizations.py --demo permutation --demo alibi
python 05-transformers-llms/69-positional-encoding/visualizations.py --demo extrapolation --seq-len 128 --dim 32 --base 10000
```

No arguments lists demonstrations without generating files or opening windows.
`--quick` defaults to HTML/PNG only, caps DPI at 75 and animation frames at eight;
`--quick --format all` explicitly includes short GIFs. `--show` opens one gallery,
not a window for every figure. `--save` is an explicit synonym for the existing
export behavior; all selected formats are saved regardless of that flag.

Parameters include `--seq-len` (8–512), `--dim` (even, 4–128), `--base` (2–1e8),
`--frames` (4–80), `--fps` (1–12), `--seed`, `--dpi` (60–160), `--output-dir` and
repeatable `--format html|png|gif|all`. The extrapolation scene samples contexts
up to 4,096 positions independently of the ordinary `--seq-len` limit.

Every execution creates a fresh `outputs/visual/run-<UTC timestamp>/` folder.
Open its `index.html`. Pages share **local** `plotly.min.js` and
`visual_controls.js`; keep the export folder together. No server or CDN is
needed, including when opening pages through `file://`. The gallery includes
GIF/PNG links and `report.json`; `frames/` retains three inspected-state
candidates per animation for close reading. These exports remain ignored.

## Demonstrations and legitimate observations

Run each row with `--demo <name>`. Names are also shown by `--list`.

| Demonstration | Computation and controls | Main generated artifacts | Observation and boundary |
|---|---|---|---|
| `permutation` | Same NumPy content/QKV projections; original, swapped dog/cat, and inverse-aligned outputs; joint token/coordinate control | `permutation_equivariance.gif`, `attention_with_without_position.png`, `permutation_interactive.html` | Unmasked content-only attention is equivariant; fixed slots generally change the aligned result. Animation transports labels schematically; matrices show actual discrete endpoints, not interpolated language tokens. |
| `sinusoidal` | Paired sine/cosine coordinates; editable length, width, base, pair and position; synchronized curves/cursor/vector bars | `sinusoidal_frequencies.gif`, `sinusoidal_frequencies.html` | Frequencies span scales; wavelength is tokens/cycle and angular frequency is radians/token. No semantic inference. |
| `construction` | Progressive position columns and widths 8/16/32/64/128 using the same diverging scale | `positional_encoding_construction.gif`, `positional_encoding_heatmap.png`, `positional_dimension_comparison.html` | Blank columns are unconstructed. All heatmaps use [-1,+1], without independent normalization. |
| `embedding-space` | Fixed PCA basis per width and exact selectable original dimensions; prefix/position controls | `positional_embedding_space_3d.html`, `original_dimensions_3d.html`, `positional_embedding_trajectory.gif` | Hover carries original norm, previous-position cosine/L2 and projected coordinates. Reported PCA variance is retained variance, not distance fidelity. |
| `addition` | Exact 2D/3D content-plus-position arrows, repeated content at different positions, different content at one position; shared high-dimensional linear basis | `token_position_vector_addition.gif`, `position_aware_embeddings_3d.html` | $(x+e)B=xB+eB$. The PCA-derived vector basis does not subtract a mean; changing positions preserves width, not vector norm or content recoverability. |
| `similarity` | Original-dimensional cosine matrix, moving reference row, signed-offset curve and bounded 3D surface | `positional_similarity_matrix.png`, `relative_distance_similarity.gif`, `positional_similarity_surface_3d.html` | Pure sinusoidal dot products depend on displacement. Curves can oscillate; these are not final attention scores. |
| `absolute-relative` | Same six-slot pattern at 5–10 and a common shift up to +20; seeded absolute table and scalar relative bias | `absolute_vs_relative_positions.gif`, `positional_methods_comparison.png`, `absolute_vs_relative_interactive.html` | Relative biases survive common shifts; absolute vectors change. Pure sinusoidal Gram identities also survive, but additive content attention need not. No table optimization. |
| `rope` | Exact Q/K pair rotations; query/key positions, frequency, original angles, magnitudes and shared shift; separate independent-key-motion and common-shift animations | `rope_rotation.gif`, `rope_relative_shift_invariance.gif`, `rope_interactive_geometry.html` | Norms and fixed-content common-shift dots are preserved; independent displacement changes scores. Absolute vectors rotate throughout. |
| `rope-multifrequency` | Up to four exact displayed pair planes, all pair-dot contributions and a fixed-basis PCA trajectory | `rope_multifrequency_rotation.gif`, `rope_attention_contributions.html`, `rope_embedding_trajectory_3d.html` | Every pair contributes to the full unscaled dot, then the attention score divides by $\sqrt d$. RoPE is not inherently one 3D rotation. |
| `alibi` | Positive head slopes; flat logits then a distant content boost; causal bias/logit/probability heatmaps and a distance-by-slope probability surface | `alibi_distance_penalty.gif`, `alibi_attention_heatmaps.png`, `alibi_interactive.html`, `alibi_attention_surface_3d.html` | Bias is added after content scaling. Future cells are masked; finite penalties can be overcome by content. The surface holds 32 visible keys fixed and uses flat content. |
| `comparison` | Six mechanisms with the same synthetic input/projections and causal visibility; query/method controls, logits, probabilities, entropy and expected distance | `attention_mechanisms_comparison.png`, `attention_mechanisms_interactive.html` | Additive methods act before Q/K/V; RoPE after Q/K projection; biases after scaling. Distribution differences do not rank trained-model quality. |
| `extrapolation` | Hypothetical 128-position boundary; sampled 128/256/512/1024/4096 contexts, table bounds, rotary float32/64 measurements and coordinate interpolation | `position_extrapolation.html`, `rope_numerical_precision.png`, `position_interpolation.gif` | Separates representability, phase precision, architecture/runtime capacity and trained generalization. No model was trained; interpolation is illustrative, not an evaluated context-extension method. |
| `rag` | Editable prompt blocks with one actual synthetic fact, evidence span and fixed-size evidence movement; inspectable prompt text | `rag_evidence_position.html` | Display units are whitespace words, not model tokens. Exact match, grounding and citation correctness are proposed metrics; no LLM or accuracy curve is evaluated. |

The main question for RoPE is whether moving both position IDs preserves the
interaction **with original Q/K vectors fixed**. Compare that control with moving
only the key. Real hidden states can also depend on surrounding content, masks
and previous layers.

## Geometry, masking and bounded computation

PCA fits once on positions `0..max(128, seq_len)-1` for each width. Trajectory
prefixes reuse the full-window mean and basis; independently fitted widths are
not a shared coordinate system. The exact-coordinate page performs no PCA.
Vector addition instead uses a single PCA-derived linear basis without affine
centering; subtracting the mean separately from every summand would break the
displayed addition identity.
At position zero, the previous-position hover fields use the display convention
cosine 1 and distance 0; there is no actual predecessor for that first point.

The comparison uses a deterministic single-head educational forward, with no
dropout and identical causal masks across methods. The permutation experiment
is deliberately unmasked. Values are rotated in neither rotary comparison.
ALiBi slopes and the relative scalar bias are explicit teaching choices, not a
reproduction of a trained checkpoint or the original head-slope schedule.

The report distinguishes scene populations: six permutation tokens, at most
64 tokens in static comparison, at most 128 in the report's comparison, and an
editable HTML comparison initially limited to 32. HTML content fixtures cache
128 rows. Absolute tables are seeded random arrays; their random-number stream
offset depends on fixture length. Comparisons are paired **within** each scene,
not numerical claims that every scene uses identical table rows.

Interactive positional surfaces are bounded at 64×64 samples. Extrapolation
uses 96 sampled positions and selected query rows, never a dense 4096×4096
attention matrix. Its numerical precision panel displays recorded NumPy
computations; live browser geometry uses JavaScript double precision. Phase,
trigonometry and dot all use the named dtype in the precision check, so float32
also rounds the original content vectors.

## Implementation and verification entry points

- [visualizations.py](visualizations.py): CLI, dated run folders, provenance,
  hypotheses/results, export decoding and gallery.
- [visual_core.py](visual_core.py): fixtures, stable softmax, attention, PCA,
  rotary contributions and precision measurements; reuses [from_scratch.py](from_scratch.py).
- [visual_figures.py](visual_figures.py): Matplotlib/Pillow scientific exports.
- [visual_html.py](visual_html.py) and [visual_controls.js](visual_controls.js):
  offline pages and actual responsive control calculations.
- [tests/test_visual_core.py](tests/test_visual_core.py): independent complex-number
  PyTorch parity, inverse rotations, projection/mask properties and invalid inputs.
- [tests/test_visual_exports.py](tests/test_visual_exports.py): real HTML/PNG/GIF
  export tests in temporary directories. Tests never overwrite public previews.

```powershell
python scripts/validate_repo.py tests --scope 05-transformers-llms/69-positional-encoding
python scripts/validate_repo.py syntax
python scripts/validate_repo.py links
git diff --check
```

The numerical and export tests do not launch a browser or approve scientific
interpretations. Browser interactions and author review are separate evidence.

## Visual execution record — 2026-11-25

Codex executed the complete local export on Windows CPU using Python 3.11.14,
NumPy 2.4.6, Matplotlib 3.11.2, Plotly 6.9.0, scikit-learn 1.9.1 and Pillow 12.3.0.
The interpreter was `.local/environments/n09-full-2026-11-13/Scripts/python.exe`.

```powershell
python 05-transformers-llms/69-positional-encoding/visualizations.py --all --seq-len 48 --dim 16 --base 10000 --frames 24 --fps 6 --seed 269 --dpi 90
```

The ignored full-run folder is `outputs/visual/run-20261125T162413688347Z/`.
All 13 demonstrations executed: 16 interactive HTML pages plus the gallery,
12 GIFs, seven primary PNGs and 36 PNG inspection frames. The generator decoded
all raster frames and resolved local HTML dependencies for 72 exports. Two
shared local JavaScript files support the HTML pages. This record does not
replace the earlier seed-69 analytical evidence in [notes](notes.md).

A separate presentation-only rerun of `--demo permutation --format png --format gif`
with the same numerical configuration is retained at
`outputs/visual/run-20261125T163028560581Z/`. It shortens panel titles to avoid
cropping; equations, fixtures and measured results are unchanged. The original
full-run artifacts and report remain preserved. Use this later folder for the
revised permutation raster exports; the full-run HTML controls are unchanged.

The permutation scene uses six identifiable token occurrences, shared
projections, no dropout and no causal mask. Repeated occurrences of “the” share
the same content embedding. The global permutation is `[0,4,2,3,1,5]`.
Attention comparison scenes instead use a common causal mask across all six
methods. The detailed scene configurations, random streams, source hashes,
hypotheses, results, interpretation candidates and limitations are retained in
the ignored `report.json`.

| Hypothesis / measured control | Result | Interpretation candidate — author review required | Limitation |
|---|---:|---|---|
| Content-only unmasked attention is equivariant | Aligned maximum error $2.2204\times10^{-16}$ | Consistent with row equivariance | Synthetic untrained weights; no language quality measure |
| Keeping positional slots fixed generally changes aligned outputs | Maximum difference $0.827887$ | Slot assignment changes this fixture's attention | Difference magnitude is not model quality |
| Permuting content and coordinates together preserves the operator's symmetry | Maximum error $4.4409\times10^{-16}$ | Consistent with the joint-coordinate control | No position-dependent mask in this experiment |
| A shared uncentered linear basis preserves vector addition | Maximum error $4.4409\times10^{-16}$ | The displayed vector construction respects addition | Projection can still discard information |
| All causal attention rows normalize | Largest row-sum error $2.2204\times10^{-16}$ | Numerically consistent probabilities | Does not validate a trained attention mechanism |
| A 3D PCA projection retains only part of positional variance | Sum of explained ratios $0.485551$ | Original-dimensional hover metrics remain necessary | Fit on positions 0–127, width 16; no cross-width coordinate comparison |

For fixed seed-271 Q/K vectors and relative offset five, phase/trigonometry/dot
construction in the named NumPy dtype produced:

| Absolute origin | float32 absolute dot error | float64 absolute dot error | float32 represented offset |
|---:|---:|---:|---:|
| 128 | $7.4158\times10^{-8}$ | $3.6082\times10^{-15}$ | 5 |
| 4,096 | $3.7015\times10^{-5}$ | $3.6693\times10^{-14}$ | 5 |
| 1,000,000 | $0.012421$ | $3.3192\times10^{-11}$ | 5 |
| 100,000,000 | $1.647380$ | $5.1679\times10^{-10}$ | 8 |
| 1,000,000,000 | $0.775620$ | $1.9370\times10^{-8}$ | 0 |

The reference is the relative-form float64 dot $-0.15542945300571193$.
**Interpretation candidate — author review required:** computing absolute
coordinates and phases in low precision can lose intended displacements;
relative algebra alone does not remove that numerical constraint. Float32 also
rounds content in this protocol. This is not a benchmark of a model's mixed
precision kernels or million-token reasoning.

ALiBi used 32 keys and query 24, with zero content logits for the isolated
penalty check. Expected past distance was 10.964315, 7.272695, 1.541401 and
0.581977 tokens for supplied slopes 0.02, 0.1, 0.5 and 1.0 respectively.
**Interpretation candidate — author review required:** stronger slopes favor
recent keys in this flat-content fixture. The separate +12 distant-content
logit scene illustrates competition, not real relevance retrieval. No paper
slope schedule, optimizer or trained model was evaluated.

## Selected public previews

| Preview | Bytes | Reason to keep |
|---|---:|---|
| [RoPE common shift](assets/rope_relative_shift_invariance.gif) | 497,906 | Absolute vector motion and constant relative interaction require multiple states |
| [Sinusoidal widths](assets/positional_encoding_heatmap.png) | 98,529 | Five widths with the same [-1,+1] color scale |
| [Matched attention mechanisms](assets/attention_mechanisms_comparison.png) | 101,600 | Same projections and causal visibility; probability scale [0,1] in every panel |

These three inspected candidates total 698,035 bytes. Only the rotary GIF is
inline in the topic README; full galleries, alternate exports, inspection
frames and run records remain ignored. Curation is not author approval.

## Verification record

The focused repository runner executed **85 passing cases in three isolated
files**: 49 original analytical cases, 33 visual numerical cases and three
export/CLI cases. There were no skips, failures or test warnings in this final
run. Real export checks decoded PNGs and GIF frames and resolved the local HTML
resources in temporary directories. The numerical oracle uses PyTorch
2.15.0+cpu and pytest 9.2.0.

Repository checks parsed 499 Python files, checked 1,848 local links in 309
Markdown files, passed `node --check` for the interaction script and found 81
compatible installed packages with the explicit-interpreter offline `uv pip check`.
`git diff --check` reported no whitespace errors; Git emitted its informational
LF-to-CRLF notice for the module index.

Chrome 158 on Windows ran 127 final checks across thirteen demonstration pages,
including range controls through the keyboard, selectors, play/pause, masks,
normalization and six browser-versus-NumPy probability comparisons. The largest
probability difference was $5.5511\times10^{-17}$ for the inspected query row.
There were no JavaScript/console errors or external HTTP(S) requests. Five
Canvas2D readback-performance warnings were emitted; they are recorded locally.
The three auxiliary HTML filenames share the same controls; they were resource
checked but not independently included in this thirteen-page interaction count.

A separate four-check geometry review exercised actual 3D camera dragging,
verified the original-dimensional hover fields, and tested a 430×932 mobile
viewport plus keyboard operation. Desktop viewport was 1500×1100. The inspected
renderer identified Intel HD Graphics 630 via ANGLE/Direct3D11. These are
functional local-flow checks, not GPU benchmarks, cross-browser coverage,
exhaustive accessibility review, GitHub rendering or production verification.
Browser logs and screenshots remain ignored; successful checks do not approve
the interpretations on the author's behalf.

All thirteen core demonstrations were generated; no core experiment was skipped.
Optional language-model evaluation, absolute-table optimization, bfloat16,
checkpoint adaptation and trained context-extension studies were not run.
The RAG page is an executed prompt-construction tool, not an executed model
quality evaluation. Interpretation candidates remain pending author review.
