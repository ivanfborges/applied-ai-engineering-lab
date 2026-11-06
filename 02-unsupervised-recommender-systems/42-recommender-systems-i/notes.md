# Collaborative filtering: evidence, support and fallback

## The matrix is only partially observed

Let $R\in\mathbb{R}^{m\times n}$ denote users by items, with observed pairs
$\Omega$. Only entries in $\Omega$ are ratings. The code stores unobserved entries
as `NaN` and rejects infinities. It accepts explicit ratings on a declared scale
(default 1–5), including fractional values. A zero is invalid on this scale;
another scale requires different `rating_bounds`.

The demonstration has seven user rows and seven catalog columns. One user and
one item have no training observations. Keeping their IDs in the matrix tests
the distinction between a known identifier and learned preference evidence.
An entirely unknown user can instead be passed as `None`. Unknown item indices
raise an error; represent a known new catalog item with an empty column.

Ratings are not sampled uniformly from all possible pairs. The user chooses
what to consume and rate, and the platform influences what is exposed. A sparse
matrix is therefore more than a storage issue: the observation mechanism affects
what an offline result can mean.

## Similarity and prediction use different means

For $I_{uv}$, the set of items rated by both users, define $n_{uv}=|I_{uv}|$ and
the two overlap-local means $\bar r_{u,uv}$ and $\bar r_{v,uv}$. The implemented
similarity is

$$
s(u,v)=\frac{\sum_{i\in I_{uv}}(r_{ui}-\bar r_{u,uv})(r_{vi}-\bar r_{v,uv})}
{\sqrt{\sum_{i\in I_{uv}}(r_{ui}-\bar r_{u,uv})^2}
 \sqrt{\sum_{i\in I_{uv}}(r_{vi}-\bar r_{v,uv})^2}},
\qquad
\widetilde{s}(u,v)=\frac{n_{uv}}{n_{uv}+\beta}s(u,v).
$$

The function returns zero for insufficient overlap or zero variance. It requires
`min_common >= 2`; the example uses three because two nonconstant paired ratings
always have correlation magnitude one. `shrink >= 0` controls a heuristic pull
toward zero. This is not a probability or a calibrated confidence interval.

For each target item, select up to `max_neighbors` users who rated it and have
positive similarity to the target user. Exclude the user itself. Stable row-ID
ordering resolves tied similarities. With $w_{uv}=\max(\widetilde{s}(u,v),0)$,

$$
\widehat r_{ui}=\bar r_u+
\frac{\sum_{v\in N_u(i)}w_{uv}(r_{vi}-\bar r_v)}
{\sum_{v\in N_u(i)}w_{uv}}.
$$

Here $\bar r_u$ and $\bar r_v$ are **full training-row means**, not the
overlap-local means used for correlation. The weighted deviation is added to
the target baseline, then clipped to the rating bounds. Negative neighbors are
discarded, so opposing tastes cannot act as inverted evidence in this policy.
The neighbor set is item-specific: a highly similar user who never rated the
target item contributes nothing.

For example, a target row `[1, 2, 3, NaN]` has mean 2. Neighbors
`[1, 2, 3, 4]` and `[2, 3, 4, 5]` have means 2.5 and 3.5 and equal positive
overlap correlation. Both target-item deviations are 1.5, yielding 3.5.

Shrinkage need not change a prediction. If all participating neighbors have the
same overlap count, their weights receive the same multiplicative factor, which
cancels in this normalized average. Different overlap counts can change relative
weights and neighbor selection. A larger beta alone does not guarantee that the
prediction moves closer to the target user's mean.

Cosine on zero-filled centered vectors is a
different approximation. Even after centering, its vector norms include each
user's nonshared coordinates; overlap Pearson recenters and normalizes on the
common items only. The implementation consistently uses the latter.

## Fallback and candidate policy

When there is no usable neighborhood, predict the training item mean if it
exists, otherwise the training user mean if it exists, otherwise the global
training mean. The returned `source` identifies `neighborhood`, `item_mean`,
`user_mean` or `global_mean`; `neighbors` counts actual contributors.

| Situation | Policy | Evidence boundary |
|---|---|---|
| Warm user/item, no positive supported neighbors | Item mean | Item-level aggregate, no personalization |
| New user, observed item | Item mean | Same list for every history-free user |
| Warm user, unobserved item | User mean | No evidence about this item |
| New user and unobserved item | Global mean | No specific user or item evidence |
| No ratings anywhere | Reject fit | No data-derived baseline exists |

Item mean is a rating-level baseline, **not a popularity count**. A single highly
rated item can outrank a widely rated one. The example deliberately keeps this
weakness visible rather than adding an unvalidated Bayesian ranking prior.

`recommend` excludes the user's observed items and all columns without training
ratings. An optional boolean eligibility mask can further restrict the catalog.
Scores sort descending with item-ID ties, returning fewer than $K$ when needed.
Candidate availability is assumed fixed at the training cutoff; the code does
not model inventory or permission changes. A new item requires a content,
editorial or controlled-exploration route to receive exposure. Giving it a
generic fallback score does not solve discovery.

## Feedback semantics and neighboring approaches

Explicit feedback directly records an expressed preference, such as a rating.
Implicit feedback records behavior: a view or purchase supplies uncertain
evidence affected by exposure, position and context. Event counts are not
interchangeable with star ratings.

Hu, Koren and Volinsky separate an implicit preference target from confidence:

$$
z_{ui}=\mathbb{1}[n_{ui}>0],\qquad c_{ui}=1+\alpha n_{ui},
$$

$$
\mathcal L=\sum_{u,i}c_{ui}(z_{ui}-p_u^\top q_i)^2+
\lambda\left(\sum_u\|p_u\|_2^2+\sum_i\|q_i\|_2^2\right).
$$

For an unseen pair this particular model sets $z_{ui}=0,c_{ui}=1$; that is a
low-confidence modeling target, not a verified dislike. The all-pairs weighted
objective differs from fitting only observed explicit ratings. This study
explains the distinction without implementing implicit ALS. See the
[original paper](https://yifanhu.net/PUB/cf.pdf).

User-based CF compares histories across rows. Item-based CF instead estimates
relationships across columns and can precompute item neighborhoods; catalog
churn and co-rating support still matter. The
[Sarwar et al. paper](https://files.grouplens.org/papers/www10_sarwar.pdf)
describes item-based algorithms; it is background, not a claim that they always
win or an implementation supplied here.

Content-based models can use metadata for a new item. Hybrid models combine
such evidence with behavior. Factorization learns lower-dimensional user/item
representations, but isolated IDs still lack collaborative evidence. Neither
semantic embeddings nor a vector index automatically learns user preferences.
Google's [collaborative-filtering lesson](https://developers.google.com/machine-learning/recommendation/collaborative/basics)
provides an embedding-based introduction. Factorization is developed in Day 43.

## Evaluation boundaries

MAE and RMSE measure numerical rating error on the held-out observed pairs:

$$
\operatorname{MAE}=\frac{1}{|T|}\sum_{(u,i)\in T}|r_{ui}-\widehat r_{ui}|,
\qquad
\operatorname{RMSE}=\sqrt{\frac{1}{|T|}\sum_{(u,i)\in T}(r_{ui}-\widehat r_{ui})^2}.
$$

Top-K recommendation instead asks whether relevant eligible items appear high
in a list. Its evaluation must define relevance, candidate universe, seen-item
exclusion and any negative sampling; sampled-candidate scores cannot be freely
compared to full-catalog scores. Day 44 develops ranking metrics. No ranking
metric is measured here.

The example assigns artificial time 1 to 30 historical ratings and times
100–107 to eight future labels, one event per user–item pair. A global cutoff of
99 hides all future labels simultaneously. Means, similarities, support counts
and candidate eligibility use the training snapshot only. The model stays
frozen throughout evaluation, including for users who receive multiple future
ratings. This is not a replay that updates after each event. Parameters are
fixed for teaching; no tuning is performed on these test labels.

Impressions can help distinguish unexposed pairs from exposed nonresponses,
but do not eliminate position or selection effects. Offline errors on selected
ratings do not establish causal product gains. For enterprise document discovery,
collaborative signals may complement query relevance, while eligibility must
respect the requesting user's permissions at serving time. No such service or
cloud architecture is implemented here.

## Executed experiment — November 6, 2026

**Hypothesis:** the deliberately grouped synthetic preferences allow supported
positive neighbors to reduce warm-pair rating error relative to the item-mean
fallback baseline; cold users/items remain reliant on fallback.

**Execution:** Codex ran `example.py` on November 6, 2026, at
`10:44:42 UTC`, using Windows x64, Python 3.11.14 and NumPy 2.4.6.
There is no random generator or seed: all values and artificial timestamps are
declared in `synthetic_snapshot`. Configuration: 7×7 matrix, 30 train ratings,
8 future ratings, cutoff 99, minimum overlap 3, shrinkage 5, maximum 3 positive
neighbors and clipping to [1, 5]. Both policies use the exact same test pairs.

| Segment | Pairs | MAE, neighborhood + fallback | RMSE | MAE, fallback only | Neighborhood-supported predictions |
|---|---:|---:|---:|---:|---:|
| All | 8 | 0.738605 | 0.834823 | 1.577083 | 4 |
| Warm user and item | 4 | 0.510543 | 0.650192 | 2.187500 | 4 |
| New user only | 2 | 1.000000 | 1.000000 | 1.000000 | 0 |
| New item only | 1 | 0.666667 | 0.666667 | 0.666667 | 0 |
| Both cold | 1 | 1.200000 | 1.200000 | 1.200000 | 0 |

The training-only recommendation demonstration ranks Ana's unseen supported
items as Mystery (4.539442, two neighbors), Art (2.166667, one) and Garden
(1.166667, one). A history-free user receives Art (4.0), Garden (3.75) and
SciFi (3.0), all from item means. These lists are not a separate ranking-quality
experiment. NewItem is excluded from both lists because it has no training
ratings, although its future rating errors are included in the table.

**Interpretation candidate — author review required:** the observed warm-pair
error reduction is consistent with the hand-built preference groups. Cold-start
errors are identical under the two policies because those predictions use the
same fallback ladder. Returning a value for every pair must not be mistaken for
full collaborative support (here only 4/8 predictions).

**Limitations:** eight selected test labels, only four warm pairs, intentionally
favorable groups, fixed catalog and artificial times, no exposure process,
no uncertainty estimate or tuning/validation partition. This is an inspectable
mechanics demonstration, not evidence for general superiority, scalability,
fairness, ranking improvements or commercial benefit. The experiment ran;
personal author approval of its interpretation remains pending.

## Further experiments — not executed

- Hide increasing fractions of training ratings with fixed seeds; track both
  errors and neighborhood-support rate by history length.
- Compare unshrunk and shrunk similarities across unequal overlaps. Include an
  equal-overlap control where normalized predictions should stay unchanged.
- Compare raw cosine, centered cosine and overlap Pearson using an unchanged
  split, without choosing a winner from the final test labels.
- Add content metadata and a controlled new-item candidate route; evaluate it
  separately from warm-item recommendations.
- Use a documented public interaction dataset with real event/availability
  times, validation for policy selection and a clearly defined ranking universe.
