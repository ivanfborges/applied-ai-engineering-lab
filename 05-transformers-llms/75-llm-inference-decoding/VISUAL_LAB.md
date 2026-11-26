# Interactive LLM decoding visual laboratory

This lab answers nine concrete questions using synthetic logits, categorical
sampling, inspectable conditional transitions and explicit cache assumptions.
All core experiments run on CPU without credentials, downloads or APIs.
**Interpretation candidates require author review.** The original analytical
[execution record](notes.md#execution-record--2026-11-26) remains a separate
seed-42, five-token study; these visual experiments use seed 275 and ten tokens.

## Run and explore

From the repository root, with stable Python 3.11+:

```sh
python -m pip install -e ".[dev]"
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --all
python -m streamlit run 05-transformers-llms/75-llm-inference-decoding/interactive_demo.py --server.address 127.0.0.1
```

NumPy, Matplotlib, Plotly, Pillow and Streamlit already belong to the root
[dependency contract](../../pyproject.toml); pytest is in `dev`. There is no
topic-specific requirements file or new model framework. Streamlit keeps live
controls in Python so the analytical decoder is reused instead of translated
into a second JavaScript implementation. The two Plotly 3D exports are standalone
offline HTML files with their runtime bundled; the live playground needs the
local Python process.

```sh
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --list
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --temperature
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --top-k --top-p
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --interactions
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --generation --beam-width 2
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --stopping --reproducibility
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --sampling --cache
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --all --quick
```

The full profile uses 32 temperature/top-p sweep frames, a 31×31 interaction
grid per top-k condition and 10,000 sampling trials per policy. `--quick` uses
10 sweep frames and 13×13 grids; it retains all strategies, stopping cases,
16-run reproducibility protocols and the default 10,000 samples. Some animations
have a natural number of states: top-k has 10, tree search has 24, generation
has 3, and cache accounting has 8. Temperature includes an additional greedy
frame. No export opens a browser automatically.

Each execution creates a new ignored `outputs/visual-<UTC timestamp>/` with
PNGs, GIFs, HTMLs, an `index.html` gallery, per-group experiment JSONs and a
manifest of configurations, versions and SHA-256 hashes. Existing output
directories are rejected. To render saved measurements without rerunning them:

```sh
python 05-transformers-llms/75-llm-inference-decoding/visualizations.py --all --reuse-run PATH_TO_EXISTING_RUN --output-dir PATH_TO_NEW_RUN
```

Reuse checks measurement hashes, preserves their original experiment dates
and records a separate rendering date. Core files:

| File | Responsibility |
|---|---|
| [from_scratch.py](from_scratch.py) | Canonical decoding distribution, entropy, sampling and generation |
| [visual_core.py](visual_core.py) | Stage inspection, grids, exhaustive/beam search, RNG protocols and cache calculations |
| [visual_render.py](visual_render.py) | Headless Matplotlib/Pillow figures and independent flat-tile Plotly surfaces |
| [visualizations.py](visualizations.py) | Single batch entry point, provenance, selection and offline gallery |
| [interactive_demo.py](interactive_demo.py) | Seven live exploration sections and reset/preset controls |
| [tests/test_visual_core.py](tests/test_visual_core.py) | Independent numerical and protocol checks |
| [tests/test_visual_rendering.py](tests/test_visual_rendering.py) | Actual temporary PNG/GIF/HTML exports and flat-tile validation |
| [tests/test_interactive_demo.py](tests/test_interactive_demo.py) | AppTest controls, reset, all sections and zero budget |

## Questions and visual outputs

| Group | Technical question | Regenerable outputs |
|---|---|---|
| Temperature | How do fixed logits become more or less concentrated? | `temperature_softmax.gif`, `temperature_entropy.png` |
| Top-k | What disappears and what must be renormalized before drawing? | `top_k_sampling.gif` |
| Top-p | Why does one threshold retain different candidate counts? | `top_p_sampling.gif`, `top_k_vs_top_p.png` |
| Interactions | How does preceding scaling/filtering change a nucleus? | Four temperature/top-p heatmaps for entropy, candidates, maximum probability and effective vocabulary; `decoding_entropy_landscape.html`, `decoding_candidate_landscape.html` |
| Generation | When does local token selection miss the best complete path? | `autoregressive_generation_tree.gif`, `greedy_vs_beam_search.png`, `autoregressive_decoding_loop.gif` |
| Stopping | How do EOS, caps and multi-token stops affect visible text? | `generation_stopping_conditions.gif`, `structured_output_truncation.png` |
| Reproducibility | What changes when a seed is reset or an RNG advances? | `seed_reproducibility.png`, `decoding_sequence_comparison.gif`, `sequence_lengths_and_agreement.png` |
| Sampling | How far are finite empirical frequencies from a fixed policy? | `sampling_convergence.gif`, `theoretical_vs_empirical_sampling.png` |
| Cache | Which K/V work can be reused, and what still grows? | `prefill_decode_kv_cache.gif`, `inference_complexity.png` |

Blue denotes original/reference quantities, green retained mass or reused K/V,
gray excluded candidates or padding, and orange selected tokens or newly
computed quantities. Reproducibility matrices use blue for A, orange for B,
green for EOS and gray for padding; their legends explicitly state this mapping.

## Distribution contract and live controls

The ten-token vocabulary and three logit vectors are explicitly defined in
`visual_core.PRESETS`. The transformation order is
**Temperature → Softmax → Top-k → renormalization → Top-p → Sampling**.
The analytical `decoding_distribution` computes every stage. Stable sorting
uses descending original logits; lower vocabulary IDs win ties. Top-p includes
the first threshold-crossing token, rather than retaining a percentage of tokens.

The playground exposes distribution preset, T=0–3, k=1–10, p=0.05–1, seed and
100–50,000 trials. It shows logits, scaled logits, softmax, raw retained top-k
mass, raw nucleus mass after k normalization, final probabilities, an exact
value table, empirical frequencies and the first sampled token. At k=10 no
top-k candidates are removed. Reset restores all six distribution controls.
T=0 bypasses division and produces the defined index-based greedy choice.

Metrics are Shannon entropy $H$ in bits, $2^H$ effective vocabulary size,
maximum probability, positive numerical support and mass retained from the
**original T=1 distribution**. This original mass differs from temperature-
adjusted or post-top-k mass. Positive support can be smaller than nominal k
after floating-point underflow. Different inference engines can use different
ordering, tie policies or sampling flags; no library parity is claimed.

The heatmaps display computed values with no smoothing. The 3D Mesh3d surfaces
contain independent constant-height tiles, each centered on one evaluated grid
point with edges at neighboring midpoints. Hover reports the actual computed
T, p, k, entropy, support and maximum probability. No triangles bridge
different tile heights. A top-k dropdown compares disabled, 3 and 6. Tile
extent is a nearest-grid display convention; it does not locate exact
continuous threshold boundaries. This is parameter space, not physical geometry
or a real-model benchmark.

## Sequence search and stopping assumptions

The binary A/B model starts with probabilities (0.6,0.4). The next step uses
(0.55,0.45) after A and (0.95,0.05) after B; later steps depend on the last
symbol. The depth-3 experiment enumerates all eight complete paths and compares
greedy, full sampling, T=1.7, k=1, p=0.7 and configurable-width beam search.
Top-k=1 deliberately exposes its equivalence to greedy on this binary example.
The tree reports conditional model probabilities and joint model log-probability.
The loop additionally reports transformed-policy log-probability, which can
differ from the original model score. Alternative nodes remain visible.

The search experiment is fixed length, so it has no EOS, length penalty or
length normalization. Beam width 1 is greedy; larger beams can still prune
optimal paths in general. Exact enumeration is the reference here, not a
scalable production algorithm. Maximizing model likelihood does not maximize
truth, usefulness, grounding or human preference.

Stopping uses a separate handwritten fragment model producing `{"ok": true}`.
A delimiter variant emits `<EN` and `D>` before EOS. The same analytical
`generate` loop handles EOS, budget exhaustion and accumulated-text stop
matching. The matched stop and EOS are excluded from returned text, but sampled
IDs count toward budget. Cross-token stop completion can remove previously
accumulated suffix text; this is non-streaming and cannot retract already
delivered bytes. A budget of two fragments leaves invalid JSON; three fragments
can be syntactically complete while ending with `length`. Syntax, finish reason
and semantic completeness are separate observations. The loop does not implement
real tokenizer decoding or enforce a total prompt-plus-output context window.

## Reproducibility and sampling metrics

Four protocols generate 16 variable-length A/B/EOS runs each with a cap of
12 sampled IDs: greedy, fresh seed 275 per run, seeds 275–290, and one shared
seed-275 Generator advanced between runs. The last-token-dependent model makes
EOS increasingly likely with context length. EOS is represented explicitly.

Exact agreement averages equality over all unordered run pairs. Token agreement
is the mean pairwise fraction of equal token IDs up to each pair's longest
sequence, counting EOS and treating missing positions as mismatches; double
padding is excluded. Unique sequence counts and actual sampled-length
distributions are also reported. These metrics do not assume different seeds
must produce different outputs. Controlled NumPy reproduction cannot establish
distributed or hosted-model reproducibility: precision, revisions, kernels,
hardware, batching and nondeterministic operations can change outputs.

The six single-step sampling configurations are T=0.4/2, k=2/6 and p=0.5/0.9
on the moderate preset. Each resets the RNG to the same seed and uses 10,000
draws, with nested prefixes for the convergence animation. Total variation is
$\operatorname{TV}(\hat p,p)=\frac12\sum_i|\hat p_i-p_i|$. Finite-sample error
need not decrease at every observed prefix. A single seed and coupled starting
streams across configurations do not support a general variance/rate estimate.

## Prefill, decode and cache accounting

The diagram uses prompt length $P=6$ and $G=8$ output predictions. Prefill
evaluates causal prompt attention and predicts the first output token.
For prediction $j+1$, feed output token $j$; the context lengths are 6–13,
not 6–14. Newly computed K/V positions are $(6,1,\ldots,1)$ with cache,
versus $(6,7,\ldots,13)$ without cache. Old queries are not cached.

For batch $B$, layers $L$, KV heads $h_{kv}$, head dimension $d_h$, sequence
length $S$ and bytes per stored scalar $b$, retained K/V storage is

$$
M_{KV}=2BLh_{kv}d_hSb.
$$

The visual assumes $B=1,L=2,h_{kv}=2,d_h=8,b=4$. Counts of projected source
positions omit GEMM constants and feed-forward/output layers. Attention-pair
counts assume an implementation that computes the causal lower triangle:
prefill $P(P+1)/2$, then $P+j$ for each new query. Recomputing all visible
causal rows uses $S(S+1)/2$ pairs per step. Dense masked kernels may perform
more work. These counts are theoretical, not latency measurements.

A separate actual NumPy check compares full-prefix versus incrementally cached
single-head attention in float64 using fixed input vectors and shared random projections.
It is a one-layer equality check, not a multilayer Transformer or a language
generation benchmark. Caching avoids recomputing previous K/V projections;
each new query still reads the growing cache.

Time to first token includes queueing, prompt prefill and initial selection;
time per later output token depends on model execution, memory traffic and
serving conditions. Prompt length affects prefill and cache size; output length
adds sequential decode steps and grows K/V storage. No timing or cost estimate
is derived from the count plots.

## Execution and review record

### Runs and configurations

Executed by Codex on **2026-11-26** using Python 3.11.14, NumPy 2.4.6,
Matplotlib 3.11.2, Plotly 6.9.0, Pillow 12.3.0 and Windows 10. The local
playground was verified with Streamlit 1.67.0. Commands used the explicit
stable interpreter at
`.local/environments/n09-full-2026-11-13/Scripts/python.exe`.

- Full calculation/export: `visualizations.py --all --seed 275 --samples 10000`,
  started at 10:58:51 UTC and completed at 11:03:35 UTC (07:58–08:03 in
  America/Sao_Paulo). The run lives in
  `outputs/visual-20261126T105851897852Z/` and generated **24 artifacts**:
  nine GIFs, twelve PNGs, two standalone HTML landscapes and one HTML gallery.
  Nine JSON experiment records and the manifest are additional provenance.
- Presentation correction: `visualizations.py --generation --interactions
  --reuse-run 05-transformers-llms/75-llm-inference-decoding/outputs/visual-20261126T105851897852Z
  --output-dir 05-transformers-llms/75-llm-inference-decoding/outputs/presentation-20261126`,
  completed at 11:04:39 UTC. It generated ten render/gallery artifacts from
  saved, hash-verified measurements. The change widened the static tree,
  separated node labels from edges and used integer candidate colorbar ticks.
  **No experiment was rerun for this correction.** Original records and dates
  remain intact; both directories stay ignored.

The full profile evaluated **2,883** interaction configurations
(31×31×3), each through the analytical decoder. Temperature exports contain
33 frames including greedy; top-k has 10, top-p 32, the tree 24, the generation
loop 3, stopping 5, sequence comparison 11, convergence 32 and cache 8.
Frame counts were read from the generated GIFs. Parameter-sweep frames and
nested sampling prefixes are distinct kinds of experiments.

### Observations and interpretation candidates

The hypotheses and limitations for every group are in their JSON records and
in the sections above. Selected observed results:

| Group | Actual result | Interpretation candidate and boundary |
|---|---|---|
| Temperature | At T=0.1: H=0.140962 bits, max p=0.981089, effective size=1.102640. At T=3: H=3.261075, max p=0.154858, effective size=9.586971. | The fixed moderate preset becomes less concentrated; this is token uncertainty, not correctness. |
| Sequence search | Greedy AAA: log P=-1.331806, P=0.264. Exact best BAA and width-2 beam: log P=-1.190728, P=0.304. | Local argmax can miss the best joint path on this depth-3 tree; no open-ended quality result follows. |
| Stopping | EOS returned complete JSON after four sampled IDs. Cap 2 returned invalid JSON with `length`; cap 3 returned valid JSON with `length`. Split `<EN`/`D>` stop returned the complete JSON after five sampled IDs with `stop_sequence`. | Finish reason, syntactic validity and visible text length carry different information; semantic correctness was not evaluated. |
| Cache | Maximum full-prefix versus cached attention absolute discrepancy: 2.220446049250313e-16. | One-layer fixed-input numerical agreement supports the stated algebraic check, not performance or a full Transformer implementation. |

All 16 runs in the greedy and reset-seed protocols were pairwise identical.
The remaining protocols had the following observations:

| Protocol | Unique sequences / 16 | Pairwise exact agreement | Mean pairwise token agreement | Sampled length range including EOS |
|---|---:|---:|---:|---:|
| Greedy | 1 | 1.000000 | 1.000000 | 7–7 |
| Reset same seed | 1 | 1.000000 | 1.000000 | 5–5 |
| Different seeds | 15 | 0.008333 | 0.218365 | 1–11 |
| Shared RNG | 16 | 0.000000 | 0.252407 | 1–9 |

**Interpretation candidate:** resetting or advancing an RNG defines different
reproducibility protocols. A duplicate under distinct seeds is permitted.
**Limitation:** these are 16 synthetic runs per protocol; no hosted/distributed,
cross-version or language-quality reproducibility claim follows.

At 10,000 categorical draws per configuration, actual total variation errors
were:

| Configuration | Total variation distance |
|---|---:|
| T=0.4 | 0.010468 |
| T=2 | 0.008688 |
| k=2 | 0.004388 |
| k=6 | 0.007962 |
| p=0.5 | 0.004388 |
| p=0.9 | 0.010152 |

**Interpretation candidate:** the displayed empirical policies approximate the
specified categorical distributions with nonzero finite-sample discrepancies.
**Limitation:** the one-seed nested-prefix design does not estimate confidence
intervals, a universal error bound or a monotonic convergence rate. Neither
filtering nor sampling measurements establish factual accuracy.

### Preview selection

Three inspected assets were copied byte-for-byte from the saved renders into
`assets/` and explicitly unignored. They are Git candidates, with no commit
created. The selection answers three different questions and totals
**1,087,974 bytes (about 1.04 MiB)**, within the repository preview budgets.
No re-encoding, timing change or numerical change was applied.

| Public preview | Bytes | Source and reason |
|---|---:|---|
| [temperature_softmax.gif](assets/temperature_softmax.gif) | 873,108 | Original full run; all 33 synchronized probability states, including greedy |
| [temperature_top_p_candidate_heatmap.png](assets/temperature_top_p_candidate_heatmap.png) | 68,496 | Presentation run; discrete filter interactions with integer ticks |
| [greedy_vs_beam_search.png](assets/greedy_vs_beam_search.png) | 146,370 | Presentation run; exact local/global likelihood counterexample and readable alternative paths |

All other images, full HTMLs, galleries, experiment records, contact sheets,
browser screenshots and intermediate files remain ignored. The offline HTMLs
bundle Plotly and are intended for local exploration; they are not public
README assets. The batch exporter never overwrites `assets/` previews.

### Validation performed

- `python scripts/validate_repo.py tests --scope 05-transformers-llms/75-llm-inference-decoding`:
  **99 passed**, no skips or warnings, across four isolated files. This includes
  59 analytical cases, 34 numerical/protocol cases, two AppTests and four
  rendering/surface cases. Three cases actually export PNG/GIF/HTML into
  temporary directories; the fourth checks surface tile geometry and hover data.
- The full batch and the hash-checked presentation replay completed successfully.
  Representative beginning/middle/end frames from all nine GIFs and selected
  PNGs were inspected locally. Public preview byte hashes match their sources.
- A separate localhost Streamlit runtime was driven through its versioned agent
  API: all seven sections, greedy controls, reset, beam width 1 and zero budget
  returned ready without application exceptions. This is distinct from AppTest.
- Headless Chrome with software WebGL completed **22 browser checks**: both
  HTML grids and actual top-k dropdowns, camera drag/reset, rendered probability
  charts, temperature zero, presets/reset, seed/trial inputs, top-k/top-p sliders,
  all live sections and zero budget. There were **zero JavaScript/console errors
  and zero external HTTP requests**. Seven performance warnings were observed:
  three Canvas2D readback messages and four software-WebGL ReadPixels/stall
  messages. These checks establish only the inspected local desktop flows.

The cached Playwright CLI package was unavailable offline; browser review used
the already installed Playwright core with local Chrome, without installing
packages. No GPU, cross-browser/mobile or accessibility audit was performed.
Repository parsing passed for **555 Python files**, internal links passed for
**2,054 links in 339 files**, and `git diff --check` passed. Dependency
compatibility passed for **89 packages** with
`uv --no-cache --offline pip check --python .local/environments/n09-full-2026-11-13/Scripts/python.exe`;
the selected interpreter has no `pip` module. The temporary localhost runtime
was stopped after inspection. A final playground color adjustment keeps all
original/scaled logits blue, including negative logits, so gray consistently
means excluded probability mass rather than a negative score.

All interpretations above remain **author review required**. Final publication
and EN/PT root README evidence updates remain pending human review. The
analytical phase's existing navigation changes and dated records are preserved.

No optional pretrained-model extension is implemented or executed. No silent
synthetic fallback is present. GitHub math rendering, cross-browser/mobile,
accessibility and production behavior need separate review.
