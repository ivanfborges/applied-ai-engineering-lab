# Day 25 visual companion

The generator answers one sequence of questions: how priors and word likelihoods become class scores, why log scores and smoothing are needed, and why a useful class decision can coexist with a misleading posterior magnitude. All training text, count data, and toy probabilities are **synthetic and code-defined**. No external dataset, network call, or benchmark is involved.

Install the shared dependencies from the repository root, then generate everything or one view:

    python -m pip install -e ".[dev]"
    python 01-classical-machine-learning/25-naive-bayes/visualizations.py --all
    python 01-classical-machine-learning/25-naive-bayes/visualizations.py --view token-evidence
    python 01-classical-machine-learning/25-naive-bayes/visualizations.py --view posterior-3d

The script writes to its own `outputs/` directory by default and creates it when needed. Use `--output-dir PATH` for a different local destination. Run the numerical checks with:

    python -m pytest -q 01-classical-machine-learning/25-naive-bayes/tests

## View map and output policy

| Visual question | CLI view | Output and type | Policy |
|---|---|---|---|
| How does one token update a class prior? | `bayes-update` | [01_bayes_update.png](outputs/01_bayes_update.png), PNG | Public preview |
| What can violate conditional independence within one class? | `conditional-independence` | `02_conditional_independence.png`, PNG | Regenerable |
| Which words add evidence for each class? | `token-evidence` | [03_token_evidence.png](outputs/03_token_evidence.png), PNG | Public preview |
| How do log scores accumulate token by token? | `evidence-accumulation` | [04_log_evidence_accumulation.gif](outputs/04_log_evidence_accumulation.gif), GIF | Public preview |
| When does a raw probability product underflow? | `log-underflow` | `04b_log_underflow.png`, PNG | Regenerable |
| How does alpha change a zero token likelihood? | `smoothing` | `05_laplace_smoothing.gif`, GIF | Regenerable |
| Why is the two-count decision boundary a line? | `decision-boundary` | `06_decision_boundary.png`, PNG | Regenerable |
| How does only the prior move that boundary? | `prior-shift` | `07_prior_shift.gif`, GIF | Regenerable |
| How do fitted NB contrasts differ from logistic coefficients? | `nb-vs-logistic` | `08_nb_vs_logistic.png`, PNG | Regenerable |
| What does the score-difference plane look like in 3D? | `posterior-3d` | `09_posterior_surface.html`, interactive HTML | Regenerable local file |
| How can perfectly copied evidence overstate confidence? | `correlated-evidence` | [10_correlated_evidence.png](outputs/10_correlated_evidence.png), PNG | Public preview |

The four linked previews were inspected and explicitly unignored in the repository. Other generated outputs, including the roughly 5.9 MB self-contained Plotly HTML, remain ignored. The 3D file is a **class-score difference surface**, despite its historical filename; its zero plane is the decision boundary. It is not a calibration surface. No contact-sheet review images are publication assets.

## Executed observations and limits

Every interpretation below is a **candidate for author review**, not an author-endorsed conclusion. The outputs illustrate mechanisms under their stated configurations.

| View | Hypothesis and executed configuration | Generated result | Interpretation candidate for author review | Limit |
|---|---|---|---|---|
| Bayes update | A single fitted token likelihood should move a balanced prior. Eight synthetic training tickets, unigram counts, alpha 1; observe one `invoice` token draw. | Priors 0.5/0.5; likelihoods 0.0732 billing and 0.0244 access; normalized billing posterior 0.75. | The token updates the class belief through prior times likelihood. | One-token toy event; no empirical probability calibration test. |
| Conditional independence | A hidden within-class subtype can make word-presence events dependent. Seed 25, 1,200 synthetic billing rows, three conditionally linked binary words. | Observed invoice/payment co-occurrence was about 0.30; product of their marginal rates was about 0.21. | The factorization discards this within-class dependence. | Single synthetic generator and finite sample; no causal claim about words. |
| Token evidence | Billing and access words should have opposite fitted log contrasts. Same eight tickets and fitted Multinomial NB. | `invoice` had billing-minus-access contrast +1.099; `password` had -1.099. | Each occurrence can be read as an additive push on log odds. | Vocabulary and examples were constructed to be lexical and separable. |
| Accumulation | Adding known words should change class log scores by their fitted terms. The fixed document was `invoice payment amount incorrect`; five frames include prior only. | Final joint log scores were -15.547 access and -11.558 billing; billing had the larger score. | The animation makes the class choice emerge from cumulative terms. | A hand-written query, no held-out prediction metric. |
| Underflow | Repeated multiplication should lose float64 range before summing logs does. Multiply 0.05 for 350 terms on this runtime. | The raw product first equaled zero at factor 249; the accumulated log score stayed finite. | Log space preserves the ordering information after raw underflow. | The threshold is specific to this factor and floating-point runtime. |
| Smoothing | Positive alpha should give unseen in-vocabulary words a nonzero class probability. Fixed counts [8, 5, 3, 0, 0], alpha 0 through 5. | For `login`, probability changed from 0 at alpha 0 to 0.0476 at alpha 1 and 0.1220 at alpha 5. | Smoothing avoids a zero likelihood and pulls the distribution toward uniformity. | Counts are stipulated, not fitted; alpha 0 is only a diagnostic frame. |
| Boundary and 3D surface | Multinomial log-score difference should be affine in two counts. Seed 25, 28 Poisson count rows per class, alpha 1. | Fitted difference was approximately 0.00 + 0.793 × billing count − 0.884 × access count; the zero contour was a line. | The 3D plane and 2D line are two views of the same decision rule. | Count points are discrete; continuous surfaces extend the score formula for explanation. |
| Prior shift | Changing only class priors should shift that fixed boundary. Same fitted token likelihoods, billing prior 0.1 to 0.9. | At count vector (2, 2), score difference changed from -2.379 at prior 0.1 to +2.016 at prior 0.9. | A prior change alone can reverse a prediction. | This is a controlled prior override, not evidence of actual production shift. |
| Copied evidence | Treating copies of one latent event as independent should exaggerate posterior magnitude. Prior 0.5; event likelihoods 0.75/0.25; one to four perfect copies. | Correct posterior stayed 0.75; naive multiplication rose to 0.9878 at four copies. | Correlated features can produce overconfident model posteriors. | A deliberately extreme toy case; it does not prove that every correlation harms calibration. |
| NB versus logistic | The same counts can produce class-aligned but differently defined weights. Same eight tickets and vocabulary; Multinomial NB alpha 1; LogisticRegression default regularization, max_iter 1000, seed 25. | For `invoice`, NB contrast was +1.099 and logistic coefficient +0.449; for `password`, -1.099 and -0.451. | The signs align in this toy data while the numerical objects and scales differ. | No held-out model comparison or claim of relative quality. |

The text model uses `CountVectorizer` fitted only on the synthetic training texts. The two-feature count model uses seeded Poisson draws; plotted points can overlap, and the synthetic generators were chosen to expose concepts. The score surfaces are derived from `log P(class) + Σ count × log P(token | class)`; the 3D zero plane is where the two class scores tie. Real deployment probabilities require separate calibration evaluation on representative labeled data.
