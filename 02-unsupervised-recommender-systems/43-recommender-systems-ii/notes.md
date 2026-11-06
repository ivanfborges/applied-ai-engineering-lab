# Objectives, embeddings and retrieval boundaries

## Low rank is an assumption about shared structure

For $m$ users and $n$ items, a $d$-factor model scores

$$
s_{ui}=p_u^\top q_i,\qquad P\in\mathbb{R}^{m\times d},\quad Q\in\mathbb{R}^{n\times d}.
$$

There are $(m+n)d$ embedding parameters rather than one independent parameter
for every pair. Shared factors can transfer evidence across overlapping user
histories. Sparsity, changing intent and context can defeat this assumption.
Bias terms can capture global/user/item effects in rating models; the educational
BPR implementation here has no bias terms.

For explicit ratings observed at pairs $\Omega$, one possible objective is

$$
\sum_{(u,i)\in\Omega}(r_{ui}-p_u^\top q_i)^2
+\lambda(\lVert P\rVert_F^2+\lVert Q\rVert_F^2).
$$

Missing ratings do not enter that sum. By contrast, the uncentered SVD baseline
approximates the **whole binary matrix**, including its zeros. If
$X\approx U_d\Sigma_dV_d^\top$, scikit-learn's transformed user coordinates
correspond to $XV_d$ and the reconstructed scores are
$(XV_d)V_d^\top$. They are low-rank reconstructions, not probabilities or an
observed-only matrix-completion solution. Randomized SVD is approximate.

For implicit counts $r_{ui}\geq0$, confidence-weighted factorization instead
defines preference $y_{ui}=\mathbb{1}(r_{ui}>0)$ and confidence
$c_{ui}=1+\alpha r_{ui}$, then minimizes

$$
\sum_{u,i}c_{ui}(y_{ui}-p_u^\top q_i)^2
+\lambda(\lVert P\rVert_F^2+\lVert Q\rVert_F^2).
$$

ALS alternates conditional user and item least-squares solutions. Unobserved
entries still contribute at lower confidence: they are not verified dislikes.
This is the distinction in the [Hu–Koren–Volinsky paper](https://yifanhu.net/PUB/cf.pdf).
Implicit ALS is explained here but not implemented.

## BPR: a ranking loss applied to factorization

Construct a triplet $(u,i,j)$ from an observed positive $i$ and an unobserved
item $j$ for user $u$. Let $\Delta=p_u^\top(q_i-q_j)$. Our per-triplet loss is

$$
\ell=\log(1+e^{-\Delta})
+\frac{\lambda}{2}(\lVert p_u\rVert_2^2+\lVert q_i\rVert_2^2+\lVert q_j\rVert_2^2).
$$

With $g=\sigma(-\Delta)$, its gradients are

$$
\nabla_{p_u}\ell=-g(q_i-q_j)+\lambda p_u,\qquad
\nabla_{q_i}\ell=-gp_u+\lambda q_i,\qquad
\nabla_{q_j}\ell=gp_u+\lambda q_j.
$$

Each update subtracts the gradient using **copies of all three old vectors**.
Using the already-updated user vector to update item vectors changes the step.
`logaddexp(0, -margin)` and `exp(-logaddexp(0, margin))` avoid direct exponentiation
of a large positive argument. Finite-difference tests check the derivatives;
an exact one-step test checks the optimizer's simultaneous updates.

The [original BPR work](https://proceedings.mlr.press/r7/rendle09a.html) separates
the ranking criterion from the scoring model. Here we use dot-product MF,
shuffle all observed positives once per epoch and sample one unseen item
uniformly per visit. This is a teaching schedule, not an exact reproduction
of the paper's bootstrap optimizer. Regularization is applied to sampled
vectors per step: frequent users/items receive more accumulated shrinkage.
It is not an unbiased gradient of a separately added, once-per-epoch global
Frobenius penalty without frequency correction.

Unseen pools are computed from training data, so sampling terminates even for
users with one eligible item. Saturated users and completely empty training
matrices raise errors. Empty-history users get a zero factor as an explicit
uninformative policy. A column with no positives may receive negative-only
updates; that does not make it an informed cold-item representation.
The separate appended-item probe therefore uses an explicit zero-score fallback.

Held-out future positives can be sampled as training negatives because their
labels are not available then. Excluding them using future knowledge would
leak labels. This also illustrates the false-negative assumption in implicit
sampling. Do not interpret a decreasing pairwise training loss as a guarantee
of increasing held-out NDCG, or BPR as direct NDCG optimization.

## What an embedding score does and does not mean

A common orthogonal change of basis applied to user and item vectors preserves
their dot products and L2 norms. Individual collaborative axes therefore have
no uniquely identified genre meaning. The one-hot metadata axes here are
explicit genres; learned collaborative factors are different representations.

Dot product depends on vector norms; cosine divides by their norms. Switching
between them can change rankings. Text embeddings trained for semantic
similarity and collaborative factors trained on interactions need separate
task evaluation. Neither is automatically a click probability.

A two-tower model computes user/context and item vectors from features rather
than only looking up trained IDs. Item vectors can be precomputed for retrieval.
Feature-based towers can support new items if their inputs transfer; new users
still require useful context, onboarding or a fallback. Neural towers are a
possible extension here, not executed code.

## Ranking objectives and hybrid choices

| Objective | Training signal | Main caution |
|---|---|---|
| Pointwise | One pair's label, e.g. logistic click loss | Logged clicks depend on exposure; negative sampling can invalidate probability calibration |
| Pairwise | A preferred pair ordering, e.g. BPR | Sampling changes which comparisons matter; it is not direct NDCG optimization |
| Listwise | A whole list's labels/order through a list-level loss or surrogate | Reliable query/user groups and relevance labels are needed; many losses only approximate metric goals |

For content scoring, average the one-hot features of **observed history only**
and compute profile/item cosine. An empty profile has score zero in the library
used here. The synthetic favorite genre is hidden and is not used to build it.

Our hybrid policy is

$$
s^{\mathrm{hybrid}}_{ui}=\beta z_u(s^{\mathrm{BPR}}_{ui})
+(1-\beta)z_u(s^{\mathrm{content}}_{ui}),\qquad \beta\in[0,1].
$$

For each user and source, $z_u$ subtracts the mean and divides by the standard
deviation over eligible items. A standard deviation at most $10^{-12}$ maps
to zeros. Seen items never participate. The catalog must retain at least one
eligible item per user. This normalization is **not probability calibration**.

Normalize before candidate restriction so that changing retrieval budgets does
not also change score scales. Using the full catalog is feasible here because
it is tiny; a real two-stage system would need a stable train-fitted or otherwise
efficient serving transformation. No speedup is measured by this example.
The weight is selected on a separate global validation round; the model is then
refit on development history with that weight frozen. A zero selected weight is
allowed and should be reported without hiding it behind a hybrid label.

Candidate union and score fusion solve different problems. Union admits items
from either source; fusion decides their order. Feature-level hybrids instead
feed component scores, context and metadata into a learned ranker. Such a ranker
would need its own point-in-time feature/label construction, chronological
training and validation; it is not included here.

## Retrieval versus final evaluation

For target set $T_u$, candidates $C_u$ and ranked output $L_u\subseteq C_u$:

$$
\operatorname{Recall@K}_u=\frac{|T_u\cap L_u[:K]|}{|T_u|}
\leq\frac{|T_u\cap C_u|}{|T_u|}
=\operatorname{CandidateRecall}_u.
$$

This bound needs the same targets and denominator. Computing ranking metrics
only on users with a retrieved positive hides retrieval failures. Here every
user has one target, so a hit at position $r$ contributes $1$ to recall and
$1/\log_2(r+1)$ to NDCG (otherwise zero). The ideal DCG is one. The implementation
is intentionally scoped to this case, not general graded/listwise evaluation.

A larger candidate set contains the smaller one under fixed source rankings,
so candidate recall cannot decrease. Final top-K metrics need not increase:
new candidates can displace previously ranked targets. A restricted catalog
can even improve observed NDCG by discarding competitors. Full-catalog NDCG
is a reference here, not a mathematical upper bound on candidate-limited NDCG.

At each cutoff, histories, popularity, profiles and factors use only past
rounds across **all users**. This avoids the cross-user future leakage possible
with unsynchronized leave-last-one-out histories. The generator is stationary,
so chronological validity alone does not make this a realistic drift test.
Sampling distinct items also deliberately removes repeated-consumption tasks.

## Operational trade-offs and common mistakes

Use exact retrieval for small catalogs; only adopt ANN after profiling a real
latency/catalog constraint. Measure ANN recall against exact retrieval, version
the embedding/index pair and enforce availability/permission filters consistently.
A feature-rich ranker may be worthwhile after retrieval has adequate coverage;
freshness/diversity rules then need their own evaluation and product decisions.

Do not infer satisfaction, revenue, fairness or causal preference from offline
hits. Logged exposure, position bias and feedback loops change the observed
positive distribution. Uniform synthetic sampling supplies none of that evidence.
More factors can overfit sparse histories; more candidate sources also increase
operational cost and calibration responsibilities. Start with baselines and
measure the intended cohort before adding components.

## Suggested extensions — not executed

- Repeat across prespecified seeds with paired uncertainty estimates; do not
  choose seeds because the hybrid wins.
- Select factor count, epochs and sampling policy using validation data, then
  evaluate a newly reserved test period. The current test has already been read.
- Add a genuine new-item/new-user cohort and measure its relevance separately;
  the existing metadata probe only demonstrates candidate admission.
- Introduce exposure logs, preference drift or multi-interest metadata and
  examine which generator assumptions change the baseline ordering.
- Compare implicit ALS or a pointwise feature ranker only with a specific
  evaluation question and leakage-safe feature construction.
