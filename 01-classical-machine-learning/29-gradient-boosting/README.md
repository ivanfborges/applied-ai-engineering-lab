# Day 29 — Gradient Boosting Intuition

How can a sequence of weak models learn a nonlinear function? Gradient boosting
starts with a simple prediction and repeatedly adds a learner fitted to the
negative gradient of the current loss. Under squared error, that gradient
signal is the residual of the **whole current ensemble**:

$$
F_0(x)=\bar y_{\mathrm{train}},\qquad
r_{im}=y_i-F_{m-1}(x_i),\qquad
F_m(x)=F_{m-1}(x)+\eta h_m(x).
$$

Each tree predicts a correction, which can be positive or negative. The final
prediction is a sum, rather than an average of independently fitted trees.
This study connects weak learners, residuals, additive modeling, shrinkage,
and validation-based capacity selection.

Tree boosting is useful for structured prediction problems involving thresholds
and nonlinear effects, such as demand estimation or request routing from
retrieval scores and metadata. Those applications require representative labels,
features available at decision time, and evaluation against operational costs;
the synthetic example here does not validate an application.

## What to inspect

| File | Purpose |
|---|---|
| [notes.md](notes.md) | Functional gradients, leaf updates, classification distinction, trade-offs, and executed experiment records |
| [example.py](example.py) | scikit-learn learning-rate/stage comparison with separate train, validation, and test rows |
| [from_scratch.py](from_scratch.py) | NumPy one-dimensional regression stump and squared-error boosting loop |
| [tests/test_gradient_boosting.py](tests/test_gradient_boosting.py) | Split oracle, closed-form shrinkage, residual targets, refitting, controlled library agreement, and selection checks |
| [interview_questions.md](interview_questions.md) | Conceptual, mathematical, validation, and deployment questions |
| [references.md](references.md) | Original research and official documentation |

The runnable examples focus on regression. The notes explain why binary
classification uses loss gradients in logit space. Modern boosting frameworks
belong to Day 30 and are not required here.

## Run from the repository root

Use the shared dependencies in [pyproject.toml](../../pyproject.toml):

```bash
python -m pip install -e ".[dev]"
python 01-classical-machine-learning/29-gradient-boosting/example.py
python 01-classical-machine-learning/29-gradient-boosting/from_scratch.py
python -m pytest -q 01-classical-machine-learning/29-gradient-boosting/tests
```

The library example prints train/validation RMSE at selected stages, scans
three learning rates and stages 0–300, then evaluates only the selected prefix
on test rows. All observations are synthetic, generated in the script; no
external dataset is downloaded.

Its configuration, environment, full measured trajectories, and review status
are written to `outputs/experiment.json` relative to the script. This generated
file is ignored and overwritten on reruns. The intentional public result
summary is in the [executed experiment record](notes.md#executed-learning-rate-and-stage-experiment).
No generated assets are selected for publication.

## Key takeaways and evidence boundary

- Ordinary residual fitting is the squared-error case of loss-gradient fitting.
  Other losses need their own gradients and possibly different leaf updates.
- Learning rate, stage count, and tree capacity interact. A small learning rate
  can leave substantial error at a fixed tree budget.
- Training error can decrease while held-out error worsens. Stage selection
  belongs on validation data; test data supplies a final evaluation.
- In the executed synthetic run, validation selected learning rate 0.1 and
  stage 234; selected test RMSE was 0.374100 versus 1.869013 for the training-mean
  baseline. Interpretation remains **pending author review**.
- The scratch model handles one finite numeric feature, unweighted observations,
  and squared error. It lacks interaction modeling, sampling, alternative
  losses, early stopping, and efficient split search. It is an educational
  implementation rather than a production library replacement.

Prerequisites: [decision trees](../26-decision-trees/),
[Random Forest](../27-random-forest/), and
[gradient descent](../../00-foundations/06-gradient-descent-from-scratch/).

## Visual laboratory

[README_VISUALS.md](README_VISUALS.md) describes the separate Python laboratory
for inspecting residual targets, additive updates, shrinkage, and validation
behavior. It generates 12 PNGs, two GIFs, and two offline HTML explorers under
ignored outputs. All experimental interpretations and preview selections
remain pending author review.

~~~bash
python 01-classical-machine-learning/29-gradient-boosting/visual_lab.py --headless
~~~
