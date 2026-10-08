# Random Forest: diversity, OOB evaluation, and importance

A random forest combines trees trained on bootstrap samples. At each split, a tree considers a random subset of features. These choices make trees less alike, so aggregating their predictions can reduce the instability of a single tree. For classification, scikit-learn averages tree class probabilities and predicts the class with the highest mean probability; majority vote is another possible rule.

This study uses **seeded synthetic tabular data** with a known nonlinear rule, independent label noise, and unrelated noise features. It examines bootstrap sampling, feature randomness, out-of-bag (OOB) predictions, and two feature-importance measures. It makes no claim about real-world performance or causal effects.

## Files and execution

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Variance and bootstrap mathematics, OOB boundaries, importance caveats, and the executed example record |
| [example.py](example.py) | scikit-learn forest with OOB, held-out accuracy, impurity importance, and held-out permutation importance |
| [from_scratch.py](from_scratch.py) | Educational bootstrap ensemble of randomized decision stumps with explicit OOB voting |
| [tests/test_random_forest.py](tests/test_random_forest.py) | Check sampling, feature selection, OOB aggregation, repeatability, and invalid inputs |
| [visualizations.py](visualizations.py) | Generate 16 conceptual, measured, animated, and optional 3D views from synthetic data |
| [VISUAL_GUIDE.md](VISUAL_GUIDE.md) | Visual questions, execution modes, evidence boundaries, and results marked for author review |
| [tests/test_visualizations.py](tests/test_visualizations.py) | Check visual math and one-view CLI generation |
| [interview_questions.md](interview_questions.md) | Questions about the mechanisms and evaluation pitfalls |

From the repository root, install the shared dependencies in [pyproject.toml](../../pyproject.toml), then run:

    python -m pip install -e ".[dev]"
    python 01-classical-machine-learning/27-random-forest/example.py
    python 01-classical-machine-learning/27-random-forest/from_scratch.py
    python -m pytest -q 01-classical-machine-learning/27-random-forest/tests

The scratch model uses one split per tree to expose sampling and voting. A production forest grows recursive trees and samples candidate features at **every** split. The scripts are demonstrations, not equivalent implementations.

## Visual lab

Run the complete local lab from the repository root:

    python 01-classical-machine-learning/27-random-forest/visualizations.py --all

Use --static, --animations, --interactive, or --only 07_correlation_variance to narrow a run. See the [visual guide](VISUAL_GUIDE.md) for the question answered by each view and the measured results awaiting author review. The 14 PNGs, GIF, and optional self-contained HTML are generated under outputs/visualizations/ and remain ignored, regenerable local artifacts; no generated asset is linked as a public preview.

## Key takeaways

- A bootstrap sample has the same number of draws as the training set, but only about 63.2% of rows are unique for large samples. About 36.8% of original rows are OOB for a given tree.
- OOB predictions use only trees that did not train on that row. OOB estimates are useful when rows are exchangeable, but they do not repair group, time, target, or preprocessing leakage.
- Impurity importance measures training split use. Permutation importance measures the change in an evaluation score when a feature is shuffled. Both describe model behavior and can mislead with correlated features.
- More trees can stabilize aggregation while increasing compute and memory. `max_features` trades tree strength against similarity among trees.

The [notes](notes.md#executed-synthetic-example) record the exact configuration and output from the run, with the interpretation marked for author review.
