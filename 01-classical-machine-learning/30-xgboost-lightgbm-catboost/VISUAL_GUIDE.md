# Day 30 — Boosting visual companion

[visualize_day30.py](visualize_day30.py) connects the study's mechanisms to
thirteen views. All data are synthetic. Some views are deliberately simplified
diagrams; measured model experiments do not establish a general library ranking.

## Generate locally

From the repository root, using the canonical dependency bounds:

```bash
python -m pip install -e ".[dev,boosting]"
python 01-classical-machine-learning/30-xgboost-lightgbm-catboost/visualize_day30.py --only gradient_hessian regularization
python 01-classical-machine-learning/30-xgboost-lightgbm-catboost/visualize_day30.py --all
```

The selected-view command is a small starting point. The full command also
trains library models and renders GIFs; it is not required for reading the study.
No dataset download or browser is required. The 3D HTML bundles Plotly locally.
Missing optional libraries skip only dependent views; unexpected failures
produce a nonzero exit code. Install from the repository root, not ad hoc
unbounded package versions.

Generated images, animations, HTML and JSON evidence live in ignored `visuals/`.
Re-running a selection overwrites its matching artifacts. Records include
configuration, results, limitations and interpretation candidates; author review
is still required. These are different experiments from [example.py](example.py).

## Questions and boundaries

| CLI selection | Question | Important boundary |
|---|---|---|
| `boosting` | How do successive trees correct residuals? | In-sample synthetic regression illustration |
| `gradient_hessian` | What do loss slope and curvature mean? | Curvature is not statistical confidence |
| `split_gain` | Which split improves the regularized local objective? | Approximate second-order gain, not a full library implementation |
| `regularization` | How do L2 and gamma change leaf scores and acceptance? | A controlled numerical mechanism, not tuned model performance |
| `tree_growth` | How do depthwise and best-first expansion differ? | Illustrative gains and unequal leaf budgets |
| `histogram` | What changes when thresholds are binned? | Candidate reduction is not a runtime benchmark |
| `goss` | Why reweight sampled small-gradient observations? | Simplified sampling illustration |
| `efb` | How can exclusive sparse features share storage? | Simplified bin codes, not ordinary ordinal encoding |
| `catboost_ordered` | Which labels may enter a row's target statistic? | Prefix-based illustration with a fixed prior |
| `symmetric_tree` | What does repeating a split at each depth imply? | Structural illustration, not a capacity-equivalence claim |
| `decision_surface` | How do shallow models partition a feature plane? | Fixed synthetic models and scores; no tuning contest |
| `capacity` | How does capacity affect training/validation loss? | One synthetic split, no universal optimum |
| `calibration` | Can ranking stay fixed while probability quality changes? | Bin support matters; Brier/log loss do not isolate calibration |

## Validate and publish deliberately

```bash
python -m pytest -q 01-classical-machine-learning/30-xgboost-lightgbm-catboost/tests/test_visualize_day30.py
```

Tests cover selected numerical contracts and temporary-file rendering, not
inspection of every exported frame or browser interaction. Before publishing
a preview, inspect it, review its interpretation, deliberately unignore that
specific file and only then add a Markdown link. There are no public preview
links here to ignored local outputs.
