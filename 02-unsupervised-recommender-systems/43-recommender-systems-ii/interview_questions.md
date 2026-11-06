# Interview discussion

**How do SVD, explicit matrix completion and implicit ALS differ?**

SVD approximates the matrix supplied to it, including zero-filled entries.
Observed-only rating factorization fits available ratings. Implicit ALS fits
binary preference indicators with confidence weights over observed and
unobserved pairs. The representations may all be low rank, but the data
assumptions and losses differ. This study executes SVD and BPR, not ALS.

**Why use BPR if the product is judged by NDCG?**

BPR supplies a tractable pairwise surrogate that prefers observed items over
sampled unseen items. It does not directly maximize NDCG or model all missing
items as verified negatives. I would validate the sampler, regularization and
serving candidate population rather than assuming objective alignment ensures
better held-out ranking. The default run here did not favor BPR.

**What is the sign of the user-vector gradient?**

For $\Delta=p_u^\top(q_i-q_j)$, the data-loss derivative is
$-\sigma(-\Delta)(q_i-q_j)$. Subtracting it moves the user vector toward the
positive/negative difference. Item updates use the old user vector; the
finite-difference and exact-step tests cover this detail. Regularization adds
$\lambda p_u$ under this topic's half-squared-norm convention.

**Does a collaborative axis represent a genre?**

There is no unique semantic axis: common orthogonal rotations preserve dot
products and L2 norms. Genre metadata and learned interaction embeddings are
different features. Dot product and cosine also differ when norms vary.

**Can a hybrid fix low candidate recall?**

Score fusion cannot rank an absent item. A union with a complementary source
can admit it, then fusion orders the union. Measure candidate recall and final
quality on the same users and targets; do not remove retrieval misses from
the ranking denominator. More candidates need not improve final NDCG.

**What did the experiment establish about hybrid value?**

Validation selected collaborative weight zero. The selected full-catalog
policy therefore matches content in this synthetic run. That is evidence about
this fixed setup, not proof that content always wins or that hybrids never help.
The generator directly ties behavior to genre metadata. The interpretation
remains **author review required**.

**How would you handle a new item or user?**

New items can enter a metadata candidate source without collaborative history,
if those features transfer to preference. A new user needs onboarding, context
or a declared popularity/segment fallback. Evaluate each cold cohort separately.
This study has a constructed new-item admission probe, not a cold-start quality
evaluation or an evaluated new-user population.

**When would a separate learned ranker be justified?**

When candidate coverage is sufficient and useful context or cross-features
cannot be expressed by the retrieval score. Construct training labels and
point-in-time features separately, hold out later periods and compare against
simple fusion, content and popularity. Include latency, freshness, coverage
and failure behavior in the decision. Offline NDCG alone does not establish
product impact.
