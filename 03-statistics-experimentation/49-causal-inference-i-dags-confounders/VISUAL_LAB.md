# Day 49 — Offline Causal Inference Laboratory

The lab asks how assumptions about paths change the quantity a regression
estimates. Eight lessons use the same six-variable SCM and exogenous draws.
All observations are synthetic. $U$ and paired potential outcomes are visible
only as simulator oracles. **Interpretations require author review.**

## Windows execution

Use stable Python 3.11+ to match the repository contract. From the repository
root, activate an isolated environment and install the canonical dependencies:

```powershell
python -m pip install -e ".[dev]"
python -m pip check
Set-Location 03-statistics-experimentation/49-causal-inference-i-dags-confounders
python visualize_causal_inference.py --all --fast
```

The generator prints absolute paths for every output. Open `index.html` from
the generated directory in Chrome, Edge or another browser with WebGL support.
Each lesson is also a standalone HTML file with its own inline Plotly bundle
and data. The dashboard links and embeds the lessons locally. Installation
may require internet access; generating and opening the lab do not.

Focused commands from this topic folder:

```powershell
python visualize_causal_inference.py --dashboard --fast
python visualize_causal_inference.py --gifs --fast
python visualize_causal_inference.py --module collider --all --seed 81 --fast
python visualize_causal_inference.py --all --seed 49 --repetitions 60 --output-dir outputs/larger-study
python -m pytest -q tests
```

`--dashboard` writes the HTML lessons and navigation page. `--gifs` writes
the three animations and two PNGs for the relevant modules. `--all` does both;
omitting all three flags also generates everything. `--module` restricts the
scope. `--output-dir` must be empty or new; the automatic default adds a numeric
suffix to preserve previous runs. No tracked preview is overwritten.

| Configuration | Interactive tape length | Offline seeds | Repetitions/size/seed | Sizes | Requested GIF frames |
|---|---:|---:|---:|---|---:|
| `--fast` | 4,000 | 4 | 12 | 200, 600, 1,800 | 18 |
| Default | 8,000 | 4 | 24 | 200, 600, 1,800, 5,400 | 30 |

`--repetitions` accepts 8–200. Larger settings are optional, not previously
measured results. BLAS work is bounded to one thread where those environment
variables are not already configured. Generation is serial, uses the Agg
renderer and does not need a GPU. Displayed points are thinned deterministically;
estimates use all cases in the current sample or selection.

## Follow the learning sequence

| Lesson / HTML filename | Question and useful interaction | Mathematical boundary |
|---|---|---|
| `dag.html` | Choose a control set and one of four paths; inspect its state and explanation | Exact eight-edge graph; original sufficient backdoor criterion, not general identification |
| `confounding.html` | Reveal C, inspect distributions and target-weighted contrasts | Separate marginal association, within-stratum differences, standardized estimate and known effect |
| `collider.html` | Change S noise, interval/threshold, sample prefix and seed | Selection on S and continuous adjustment for S are distinct operations; insufficient overlap/rank gives an unavailable estimate |
| `mediation.html` | Change $A\to M$ and $M\to Y$; animate both paths | $1.3+\alpha\beta$ is the total effect in this additive SCM; zero coefficients disable those mechanisms |
| `intervention.html` | Set A=0 or A=1 and compare synchronized distributions | Observational subsets versus paired interventions over every case; downstream M is regenerated |
| `geometry.html` | Rotate/zoom, toggle groups, strata, expectations and contrasts | Binary A/C stay binary; only displayed observations are jittered, never expected markers or numerical fits |
| `adjustment.html` | Change $C\to Y$ and $A\to M$, compare OLS and residuals | FWL equals joint OLS algebraically; it cannot establish exchangeability |
| `sample-size.html` | Switch sampling ranges/SD and the seeded simulation bank | Empirical 10th–90th percentiles or ±1 SD describe sampling spread, not causal confidence intervals |

Each module includes **Question**, **Visual explanation**, **Interaction**,
**Mathematical connection**, and **Senior-level takeaway**. Plausible examples
include GenAI adoption, RAG routing, targeted retention and feedback collection;
their arrows require validation in any real application. The baseline deliberately
assigns AI more often to complex cases. Another product might favor simple cases,
which would require a different assignment mechanism and potentially reverse bias.

## Shared equations and live controls

The baseline equations and all eight edges are specified in the
[topic README](README.md). Each embedded tape stores $C$, conditionally
assigned $A$, and independent normal draws for $U,\varepsilon_M,
\varepsilon_S,\varepsilon_Y$. Those normal draws are also independent of $C$
and the assignment randomness. Four NumPy tapes start at the CLI seed and end
at seed+3. The $A$ draws follow $P(A=1\mid C)=0.2+0.6C$ and are not independent of $C$.
Standard deviations for the baseline M/Y disturbances are 1 and for S are 0.5.

Parameter changes regenerate the structural equations over the same tape,
so unrelated Monte Carlo changes do not obscure the chosen mechanism. Each
lesson starts from baseline parameters; changes are local to that lesson.
The sample-size slider takes a prefix of the full tape and does not redraw
assignment. Changing a browser seed selects another embedded NumPy tape;
arbitrary new seeds require regenerating the HTML with `--seed`.

With $alpha=A\to M$, $eta=M\to Y$, $gamma=C\to Y$ and S-noise SD $sigma$:

$$
\tau_{total}=1.3+\alpha\beta,\qquad
\tau_{naive}=\tau_{total}+0.6\gamma,\qquad
\tau_{C,S}=\tau_{total}-\frac{1.2}{1+\sigma^2}.
$$

These are population properties of the specified independent-disturbance SCM.
Baseline values are 2.0, 2.9 and 1.04; the direct path is 1.3. The baseline
conditional means are $E[Y\mid A,C]=2A+1.5C$. Natural observational means are
0.3 and 3.2, whereas intervention means are 0.75 and 2.75.

The browser recomputes finite-sample QR regressions and standardization from
the embedded data. FWL residualization removes the intercept and linear C
from both A and Y. Python uses NumPy least squares and checks against the
original FWL routine and scikit-learn. Small finite-sample differences between
standardization and constant-effect OLS are expected.

The repeated-size module uses fresh NumPy samples for each size and repetition,
separately from the interactive prefix. For browser seed $s$, its seed offset
is $10000s+100000+1000i+r$, with zero-based size index $i$ and repetition $r$.
The generator records mean, SD, MCSE and empirical percentiles in `run.json`.
These precomputed curves do not refit when unrelated controls in another lesson
change. The default profile and larger repetition studies remain unexecuted
unless a separate run is recorded below.

## Generated artifacts and publication

Every selected module produces a standalone HTML file; `index.html` provides
navigation. The `assets` subdirectory contains:

- `confounding_explained.gif`: observed association → reveal C → stratify →
  compare within strata → standardize with target weights → compare with truth.
- `collider_bias_explained.gif`: narrow an S interval over a fixed population;
  inspect induced A–U dependence and the selected sample count.
- `mediation_explained.gif`: trace direct/indirect paths and combine their
  structural contributions. Packet motion is schematic, not elapsed time.
- `causal_dag.png` and `treatment_effect_comparison.png`: local preview candidates.

Animations use fixed axis limits, deterministic frames and readable annotations.
Identical consecutive GIF frames may be combined by Pillow while preserving
their total display duration. Frames are rendered in memory and not exported
as dozens of PNG files. `run.json` records configuration, versions, computed
results, paths and elapsed generation time. Existing records are preserved.

All output stays under ignored `outputs/`. The author has not selected public
previews; there are no Markdown links to ignored exports. Open the generated
paths printed by the CLI to inspect them locally. The root READMEs and module
index were not changed by this visual phase.

## Executed visual study — 2026-11-12

**Hypothesis:** the same baseline SCM should produce coefficients near its
analytical limits, collider selection should induce A–U dependence, and larger
samples should reduce sampling spread without repairing structural bias.
Browser controls should reproduce the corresponding Python calculations.

The first `--all --fast` export used seed 49, four 4,000-case tapes, 12
repetitions for each of three sample sizes and 18 requested GIF frames.
It completed in 17.91 seconds in the local Windows environment. Its ignored
directory `outputs/visual-2026-11-12-fast` is preserved as the initial run.
Image inspection led to a separate final export with clearer 3D layout,
path colors, GIF labels and total-effect path highlighting.

**Final run:** `outputs/final-2026-11-12-fast`, generated in 19.11 seconds with
the same configuration. The exact command from the repository root was:

```powershell
.local\environments\day46\Scripts\python.exe 03-statistics-experimentation/49-causal-inference-i-dags-confounders/visualize_causal_inference.py --all --fast --output-dir 03-statistics-experimentation/49-causal-inference-i-dags-confounders/outputs/final-2026-11-12-fast
```

Environment: Windows, Python 3.11.14, NumPy 2.4.6, pandas 3.1.1, Matplotlib
3.11.2, Plotly 6.9.0, NetworkX 3.6.1, Pillow 12.3.0, scikit-learn 1.9.1,
pytest 9.2.0. The output occupies 48.74 MiB, including nine HTMLs, three GIFs,
two PNGs and `run.json`. Inline Plotly bundles make the individual HTMLs
portable at the cost of repeated library bytes.

**Measured baseline, 4,000 cases / seed 49:**

| Quantity | Computed value | Analytical population reference |
|---|---:|---:|
| Naive coefficient | 2.911520 | 2.90 |
| C-adjusted coefficient / FWL | 2.095608 | 2.00 |
| C,M-adjusted coefficient | 1.345661 | 1.30 |
| C,S-adjusted coefficient | 1.091279 | 1.04 |
| Standardized over empirical C | 2.097263 | 2.00 |
| Paired intervention mean, A=0 | 0.721343 | 0.75 |
| Paired intervention mean, A=1 | 2.721343 | 2.75 |

The paired contrast is exactly 2.0 up to floating-point rounding. With S-noise
SD 0.5 and the selection rule $|S-0.5|\le0.35$, 908 cases remained. The
full-population mean U difference by A was +0.034390, versus -0.743161 in the
selected subset. C-adjusted OLS on that selected subset was 1.072586. These
finite-sample measurements are separate from the historical 50,000-case
console study in [notes.md](notes.md#executed-synthetic-study--2026-11-12).

**Repeated-size evidence, displayed seed bank 49:** each cell below summarizes
12 independent samples. Other embedded seed banks are retained in `run.json`.

| Controls | Mean at n=200 | SD at n=200 | Mean at n=1,800 | SD at n=1,800 |
|---|---:|---:|---:|---:|
| None | 2.956462 | 0.260098 | 2.919229 | 0.066740 |
| C | 2.003885 | 0.347118 | 2.027379 | 0.120980 |
| C,S | 0.965109 | 0.254132 | 1.015047 | 0.121936 |

**Interpretation candidates — author review required:** the repeated-size
results are consistent with shrinking sampling spread around different
population limits. In the 4,000-case baseline, C adjustment was still about
0.096 above total ATE; it did not recover the exact effect in that sample.
Selection induced a sizable negative A–U mean contrast, while continuous
collider adjustment also departed from the total effect. The twelve-repeat
SDs, quantiles and MCSEs are themselves noisy; these results do not establish
behavior outside the specified SCM.

**Validation actually executed:**

- 70 topic tests passed, without skips or warnings. The original 45 tests
  remain intact; 25 visual tests add external graph agreement, a collider
  descendant, modified-equation checks, an 80,000-case convergence check with
  a 0.05 absolute tolerance, repeated-simulation reproducibility and temporary
  HTML/GIF rendering.
- Offline Chrome through Playwright CLI exercised 55 scenarios: 35 numerical
  parameter/sample/seed cases compared with independent Python expectations
  at absolute tolerance $10^{-8}$; 16 adjustment/path combinations; keyboard
  slider operation, real 3D drag/zoom, group toggling and animation Play/Pause.
  Plotted coefficients, structural bars, observation/intervention means and
  binary category tooltips were checked. No JavaScript/console errors or
  HTTP(S) requests occurred. Five Plotly Canvas2D readback-performance warnings
  were emitted; these were not numerical failures.
- Both PNGs and initial/middle/final GIF frames were inspected. The stored
  confounding GIF has 8 distinct frames after identical-frame merging; collider
  and mediation have 18 each. All have 10.8-second duration. The final 3D
  chart was checked to fit its container.
- Repository syntax parsed 332 Python files; 1,084 internal links in 225
  Markdown files passed. Dependency compatibility, JavaScript syntax and
  `git diff --check` passed. The full repository test suite and Linux CI were
  not run.

## Scope and review

Only a known linear additive SCM is exercised. There is no demonstration of
identification on real data, treatment-version consistency, interference,
measurement error, hidden treatment confounding or general mediation
identification. Graph roles depend on the path and estimand. The original
backdoor criterion is sufficient; failure does not prove that every alternative
identification strategy fails.

The six-node DAG remains fixed in the explorer. The general separation helper
also handles collider descendants, which have dedicated tests even though no
new node is added to this exact study graph. NetworkX independently checks
all 240 endpoint/conditioning combinations for the study DAG.

Browser validation is scoped to the tested Chromium configuration. Canvas/WebGL
performance, alternate browsers, mobile presentation, GitHub rendering and
larger profiles need separate evidence. A working export and passing tests do
not establish author approval or a product rollout decision.

Implementation references: [NetworkX d-separation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.d_separation.is_d_separator.html)
and [Plotly offline HTML export](https://plotly.com/python/interactive-html-export/).
