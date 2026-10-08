# Day 31 SVM visual laboratory

[visualize_svm.py](visualize_svm.py) generates static geometry, two looping
parameter animations, and two offline 3D explorers. Every visual answers a
specific question. All data are synthetic and all displayed measurements are
computed during the run. Study scripts and visual experiments use separate
configurations; their metrics should not be merged.

## Run

Use the existing dependencies declared in [pyproject.toml](../../pyproject.toml).
No runtime dependency was added; no Kaleido, browser export tool, or network
connection is required by the generator.

From the repository root:

```bash
python 01-classical-machine-learning/31-support-vector-machine/visualize_svm.py
```

Or from this topic directory:

```bash
python visualize_svm.py
python visualize_svm.py --skip-gifs
python visualize_svm.py --show
```

`--show` opens the two HTML explorers and canonical PNG in the default browser.
Default execution is headless. `--skip-gifs` generates the remaining assets;
any previously generated GIFs are left untouched and excluded from that run's
console list and manifest. Output paths are relative to this script, and the
`outputs/` directory is created automatically. Reruns overwrite generated
files with the same names.

## Generated assets and their questions

All filenames below are relative to `outputs/`. They are **regenerable
artifacts**, currently ignored rather than public preview links.

| Output | Conceptual question |
|---|---|
| `01_maximum_margin.png` | Where are the decision hyperplane, score ±1 lines, geometric margin, and supporting points? |
| `02_possible_hyperplanes.png` | Why can valid zero-error separators have different nearest-point distances? |
| `03_support_vectors.png` | Which nonzero dual terms contribute to the same fixed fitted score? |
| `04_hinge_loss.png` | When is hinge loss positive, and how does its exact zero region differ from logistic loss? |
| `05_C_effect.gif` | How does increasing the violation penalty change one overlapping linear fit? |
| `06_soft_margin_violations.png` | Which correctly classified points still violate the margin? |
| `07_kernel_feature_space.html` | Can an explicit squared-radius feature enable a planar separator? Rotate, zoom, and hover. |
| `07_kernel_feature_space.png` | Static 2D/3D companion to the explicit educational lift, made with matplotlib. |
| `08_rbf_similarity.png` | How does gamma set the distance scale at which similarity decays? |
| `09_gamma_effect.gif` | How does locality change the boundary, training/validation scores, and supporting points? |
| `10_C_gamma_grid.png` | How do violation cost and locality interact on the same training rows? |
| `11_C_gamma_cv_heatmap.png` | Which configurations validate under fold-local scaling on this synthetic sample? |
| `12_scaling_effect.png` | How does a feature multiplied by 1000 change RBF distance geometry? |
| `13_decision_function_3d.html` | How does the zero-plane intersection of a curved score surface define the 2D boundary? |
| `14_support_vector_count.png` | How many kernel terms are retained for different gamma settings? |
| `15_kernel_prediction_process.png` | How do all weighted kernel terms and the intercept produce one actual prediction? |

`visual_lab_results.json` records generator parameters, split rules, versions,
all sweep and CV measurements, hypotheses, interpretation candidates,
limitations, sizes, and author-review status. It also stays ignored.
Validation contact sheets, extracted GIF frames, browser screenshots, and
check records under `outputs/_validation/` are intermediate artifacts.

## Mathematical and evaluation boundaries

- The canonical linear fit uses high `C` on separable blobs. The script checks
  near-unit minimum functional margin and absence of multipliers at the `C`
  bound before calling it a hard-margin-equivalent solution. Equal distance
  units are preserved on both axes. The total margin is `2 / ||w||`.
- Candidate hyperplanes are constructed from positive class-projection gaps;
  their validity and geometric distances are checked rather than assumed.
- The support-vector comparison uses the **same fitted model** and reconstructs
  its score from the dual terms. It does not remove rows or refit preprocessing.
- Signed margins classify points as `m >= 1`, `0 < m < 1`, or `m <= 0`.
  At exactly zero the point lies on the boundary; prediction depends on the
  tie rule. Finite solver tolerances can place support points extremely close
  to either side of margin one.
- Increasing `C` raises the penalty on violations relative to the norm term.
  The linear C animation does not claim universal monotonic margin widths.
- The explicit lift `(x1, x2, x1²+x2²)` is an educational feature-map analogy.
  It is not the RBF feature mapping. The 3D separating plane is learned from
  the squared-radius coordinate of these synthetic circles.
- RBF similarity and decision scores are not probabilities. Nonlinear score
  contours are not equal-distance margins in the original input space.
- Gamma is numeric in the locality sweep. The scaling comparison instead
  uses `gamma="scale"` in both pipelines, so its fitted gamma also changes.
  This is a pipeline comparison, not a fixed-gamma causal ablation.
- Moons use 120 training and 60 descriptive validation rows. The heatmap runs
  five shuffled stratified folds on training rows only, with scaling fitted
  inside every fold. No independent final test or latency benchmark is run.

## Executed evidence — interpretation pending author review

The lab was executed on 2026-10-07 with the existing environment: Python
3.11.0rc2, NumPy 2.4.6, scikit-learn 1.9.0, matplotlib 3.11.0, Plotly 6.9.0,
and Pillow 12.3.0. The installed Python prerelease is recorded accurately;
these checks do not cover every supported version combination.

**Maximum-margin hypothesis.** Separable data permit different zero-error
boundaries with different geometric margins. Configuration: 60 blobs, centers
`[-0.2,-2]` and `[0.2,2]`, standard deviation 0.42, seed 42; linear SVC,
`C=10000`, tolerance `1e-9`. The computed full margin was 2.484603 feature
units, with 3 support vectors (5% of rows). Candidate nearest-point distances
were 0.557756, 0.588968, and 0.496921; the SVM half-width was approximately
1.242302. The dual reconstruction maximum error was about `8.88e-16`.
Interpretation candidate: this controlled geometry illustrates why separation
alone does not select the widest corridor. Limitation: one separable generator
and a numerical high-C solution.

**Locality hypothesis.** At fixed violation cost, local RBF similarity changes
training fit and held-out behavior. Configuration: `make_moons(n_samples=180,
noise=0.24, random_state=42)`, stratified 120/60 split with seed 42,
training-fitted StandardScaler, `C=10`, tolerance `1e-6`.

| gamma | Training accuracy | Descriptive validation accuracy | Support vectors |
|---:|---:|---:|---:|
| 0.01 | 0.875000 | 0.883333 | 50 |
| 1 | 0.966667 | 0.883333 | 27 |
| 100 | 1.000000 | 0.833333 | 118 |

Interpretation candidate: the most local displayed fit achieved perfect
training classification while scoring lower on validation in this run.
Limitations: one seed, 60 validation rows, multiple inspected configurations,
and no universal optimal gamma or monotonic support-count conclusion.
The full nine-setting record is regenerated in JSON.

**Joint validation hypothesis.** `C` and gamma interact and need fold-local
validation. Configuration: values `[0.01,0.1,1,10,100]` for each parameter,
five shuffled stratified training folds with seed 42, accuracy scoring, and
scaling inside each fold. The first maximum was `C=10`, `gamma=1`, with mean
CV accuracy 0.950000 and fold standard deviation 0.031180. Interpretation
candidate: inspect this high-scoring region as evidence for the current
synthetic sample. Limitations: selection optimism, overlapping training folds,
small data, and no independent final test; fold standard deviation is not a
confidence interval.

**Scaling hypothesis.** Unequal units can dominate the RBF distance.
Configuration: multiply the same moons feature 2 by 1000; compare unscaled
SVC and a training-fitted StandardScaler pipeline, both `C=10`,
`gamma="scale"`, tolerance `1e-6`. Validation accuracy was 0.850000 without
scaling and 0.883333 with scaling. Effective gamma was approximately
`2.911595e-6` versus 0.5. Interpretation candidate: compare the measured
boundaries after changing feature geometry. Limitations: one split; the
scaling rule also changes gamma, so this does not isolate a fixed-gamma effect.

The circle lift and the measured prediction diagram have their own complete
hypothesis/configuration/result/interpretation/limitation entries in the run
record. No result is presented as the author's approved conclusion.

## Validation performed

- Executed the complete generator from both the repository root and topic
  directory. Two complete runs produced byte-identical hashes for all 16
  visual assets and the JSON record in the same installed environment.
- Ran all Day 31 tests: 51 passed. Visual checks cover the RBF distance scale,
  margin categories, valid candidate lines, radial separation, binary dual
  reconstruction, deterministic data/splits, fold-local scaling, nonempty
  output generation, figure closure, and distinct looping GIF frames.
- Inspected all static plots via a contact sheet and selected full-resolution
  images; corrected crowded annotations and repeated grid labels.
- Both GIFs have nine distinct frames and `loop=0` (continuous looping).
- Both offline HTMLs rendered in headless Chrome using the already installed
  Playwright tooling. Actual mouse rotation and wheel zoom changed each 3D
  camera; no page errors or external network requests were observed.
- Chrome reported a Canvas2D readback performance advisory, with no rendering
  failure. Browser tooling was used only for validation and was not added as
  a generator dependency.

Run the numerical/rendering tests from the repository root:

```bash
python -m pytest -q 01-classical-machine-learning/31-support-vector-machine/tests
```

## Preview recommendations

Recommended candidates after author review:

| Candidate | Measured size | Reason |
|---|---:|---|
| `01_maximum_margin.png` | about 107 KiB | Canonical boundary, margins, and supporting observations |
| `09_gamma_effect.gif` | about 415 KiB | Broad-to-local similarity made visible through the learned boundary |
| `10_C_gamma_grid.png` | about 296 KiB | Joint parameter behavior shown without a benchmark claim |

The C GIF is about 311 KiB; neither GIF is large in this run, but review looping
pace and redundancy before publishing. Keep both self-contained HTMLs local
(about 4.7 and 5.0 MiB), along with the remaining supplementary plots and
validation intermediates. Bundled Plotly.js accounts for most HTML size.

Nothing is intentionally unignored by this change. Add image links only after
selecting previews and deliberately unignoring those files; otherwise links
would point at private local outputs. The HTML files are useful for local
rotation/zoom, rather than GitHub social previews.

## Study navigation

Return to the [topic overview](README.md) for the runnable study examples and
to [notes.md](notes.md) for their separate mathematical and experiment records.

Offline export background:
[Plotly HTML export documentation](https://plotly.com/python/interactive-html-export/).
