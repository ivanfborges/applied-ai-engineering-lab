# Day 32 visual learning companion

Run from the repository root with the shared dependencies already installed:

```bash
python 01-classical-machine-learning/32-feature-engineering/visual_feature_engineering.py
```

Or run `python visual_feature_engineering.py` from this topic directory.
Outputs always go to `visuals/` beside the script unless you pass
`--output-dir PATH`. No GPU, ffmpeg, credentials, or runtime network calls are
required. PNGs use 150 dpi; GIFs use FuncAnimation and PillowWriter at 100 dpi.

Plotly is optional for the HTML view: a missing import skips that view while
retaining its static 2D companion. The shared dependency setup includes Plotly.
Its JavaScript is embedded in the HTML, so the generated file can be opened
locally without internet access. Rotate, zoom with the wheel, and hover over
points. See [Plotly's HTML export documentation](https://plotly.com/python/interactive-html-export/)
and [PillowWriter](https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.PillowWriter.html).

## Start here

1. Inspect 01 and 02 for the difference between display axis ranges and the
   actual Euclidean metric.
2. Inspect 06, open 07 in your browser, then inspect 11: a plane in lifted
   coordinates can correspond to a curved boundary in the original space.
3. Inspect 09 and 10 to explain how an event history becomes a fixed-size
   feature vector, and why an event must both occur and become available
   before its prediction cutoff.

## Generated sequence

Filenames below are local regenerable outputs, not links to ignored artifacts.

| File in `visuals/` | Question it answers |
|---|---|
| `01_scaling_before_after.png` | Why can income units dominate age variation? |
| `02_scaling_distance.gif` | How do four nearest-neighbor identities change under the fitted metric? |
| `03_categorical_encoding.png` | Which ordinal assumptions disappear with one-hot encoding? |
| `04_binning.png` | What do continuous ages become with four versus eight intervals? |
| `05_binning_resolution.gif` | How do width, occupancy, and resolution differ across uniform and quantile bins? |
| `06_interaction_2d.png` | Why is noisy opposite-quadrant structure difficult for one straight boundary? |
| `07_interaction_3d.html` | What does the fitted decision plane look like after adding x1*x2? |
| `08_polynomial_features.png` | How can a nonlinear feature still enter a score linearly in its coefficient? |
| `09_aggregation_timeline.png` | How do windowed events become six features? |
| `10_point_in_time_leakage.gif` | What information was unavailable at the original prediction timestamp? |
| `11_model_inductive_bias.png` | How do a linear score, an interaction score, and a bounded tree partition the same inputs? |
| `12_feature_engineering_pipeline.png` | Where are transforms fitted, and where are they reused? |
| `13_preprocessing_leakage.png` | How does fitting a scaler on all rows expose held-out distribution information? |

## Interpretation boundaries

Every dataset is synthetic and defined in the generator. The root seed is 42;
independent constructions use small offsets from that seed. The scaling example
fits 70 of 100 rows. The bins fit 320 training ages and display 80 held-out
ages. The classifiers fit the same 315 training observations and show 105
held-out observations.

The interaction target is `1[x1*x2 + Gaussian(0,0.3) > 0]`. Its classes are
noisy, so the HTML does not claim perfect separation. The plane uses fitted
logistic coefficients converted back from standardized features to raw lifted
coordinates. The linear models use fixed C=1; the tree uses depth 4 and minimum
leaf size 8. Boundaries illustrate inductive bias; no accuracy, timing, or
benchmark ranking is reported.

A readable raw scatter can hide a distance imbalance through independent axis
ranges. View 01 also shows equal numerical units; view 02 computes neighbors in
the actual raw or training-standardized units. This is a model-dependent metric
choice, not evidence that scaling always improves prediction.

Bin IDs are interval labels. A model using the numeric IDs directly imposes a
different assumption from one-hot bin indicators. The counts displayed above
intervals come from training ages; quantile ties can prevent equal counts.

The event feature contract is `[T-30d,T)` with `available_at<T`. The 7-day
count has the analogous 7-day window. Amounts and merchants are filtered before
aggregation. The recency feature is restricted to eligible events in the
30-day window; absent history gives zero count/mean/max and missing recency.
Timestamp equality is excluded by this lab's explicit convention. The red
leaking count and all-data scaler are deliberately invalid examples.

## Executed generation and checks

The complete generator ran locally on 2026-10-07 and produced all 13 requested
filenames. GIF durations were verified as 8, 9, and 11 seconds with infinite
loops. The HTML rendered in headless Chrome with software graphics and
external hostname resolution disabled; its JavaScript is embedded.
Numerical observations in this construction:

- Raw neighbor IDs: 1, 2, 3, 48; standardized IDs: 37, 6, 5, 4.
- Available 30-day history: 6 events; 7-day count: 2; average amount: $95.83;
  maximum: $180; distinct merchants: 3; windowed recency: 1 day.
- The deliberately invalid count uses 10 events, including unavailable ones.

**Hypothesis:** feature maps change numerical geometry and representable
boundaries; temporal filtering determines admissible historical information.
**Result:** the generated sequence exposes these changes under the configuration
above. **Interpretation candidate — pending author review:** use the visual
differences to explain those mechanisms, without inferring generalization or
business value. **Limitations:** deliberately constructed synthetic data,
fixed model settings, and no measured predictive improvement.

An ignored `outputs/visual_lab_record.json` stores the generation configuration,
versions, numerical observations, artifact inventory, interpretation candidate,
and limitations. Inspection frames stay ignored under `outputs/inspection/`.

The focused checks cover neighbor identities, train-only means/scales and bin
boundaries, the algebraic decision plane, temporal endpoints and late arrivals,
no-history behavior, invalid inputs, an absent-Plotly fallback, and Pillow GIF
timing/looping. Run:

```bash
python -m pytest -q -W error 01-classical-machine-learning/32-feature-engineering/tests
```

## Public preview selection

Only three small representative outputs are intentionally unignored and
referenced by the topic README:

- [Scaling geometry](visuals/01_scaling_before_after.png).
- [Interaction in the original space](visuals/06_interaction_2d.png).
- [Point-in-time storyboard](visuals/10_point_in_time_leakage.gif).

Other PNGs/GIFs remain regenerable. The approximately 4.9 MB self-contained HTML
stays ignored because it embeds the Plotly runtime. No generated file has been
committed or published.
