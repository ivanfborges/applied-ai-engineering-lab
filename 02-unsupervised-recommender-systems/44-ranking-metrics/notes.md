# Ranking evaluation: what is the denominator?

## Define the evaluation unit first

A ranking is an ordered list of unique item IDs for a query or user. Its
relevance judgments must be independent of its predicted scores. Here, every
item in a declared universe has a nonnegative grade; the returned list may
contain any subset of those IDs. Unreturned relevant items remain in recall
and ideal-gain denominators.

In search, an item might be a document. In RAG, it might be a chunk, a source,
or an evidence unit. Counting five overlapping chunks as five independent
answers can reward redundancy. This implementation rejects identical IDs;
it cannot detect semantically duplicate passages with different IDs.

## Five metrics, five questions

Let $R_q$ be the number of relevant items in the judged universe for query $q$,
$r_i=\mathbf{1}[g_i>0]$ at returned rank $i$, and $r_i=0$ for missing positions.
Ranks start at one and $K$ is a positive integer.

$$
P@K(q)=\frac{\sum_{i=1}^{K}r_i}{K},\qquad
\operatorname{Recall}@K(q)=\frac{\sum_{i=1}^{K}r_i}{R_q}.
$$

At fixed K these ignore internal order. Increasing K cannot decrease recall,
but precision can move in either direction. A short list is not given a
smaller precision denominator; a single relevant result at K=5 has P@5=0.2.
This convention makes failure to fill the budget visible.

$$
AP@K(q)=\frac{1}{R_q}\sum_{i=1}^{K}P@i(q)r_i,\qquad
MAP@K=\frac{1}{Q}\sum_q AP@K(q).
$$

AP rewards high precision where relevant items occur. Missing relevant items
contribute nothing to the numerator. With this **total-relevant denominator**,
AP@K is nondecreasing in K and is no larger than Recall@K. Even an ideal
top-K list has AP@K below one when $R_q>K$.

The alternative `capped` denominator is $\min(R_q,K)$. With four relevant
items and two relevant results at ranks 1 and 2, total AP@2 is 0.5 and capped
AP@2 is 1.0. These are different conventions, not conflicting arithmetic.
The nondecreasing-in-K property need not hold with the capped denominator.
The [Spark implementation](https://raw.githubusercontent.com/apache/spark/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala)
uses the capped denominator for truncated AP. Record it before comparing MAP.

For first relevant rank $h_q$:

$$
RR@K(q)=
\begin{cases}
1/h_q & h_q\leq K,\\
0 & \text{no hit within }K,
\end{cases}
\qquad MRR@K=\frac{1}{Q}\sum_q RR@K(q).
$$

RR cannot decrease as K grows for a fixed list: it is zero until the first
hit, then constant. Moving later relevant items does not affect it. A single
rank-one hit gives RR=1 even if every other relevant item is missing. This
is useful for one-answer lookup and insufficient for multi-evidence retrieval.

$$
DCG@K(q)=\sum_{i=1}^{K}\frac{G(g_i)}{\log_2(i+1)},\qquad
NDCG@K(q)=\frac{DCG@K(q)}{IDCG@K(q)}.
$$

Use $G(g)=2^g-1$ by default, or $G(g)=g$ for linear gain. IDCG sorts **all
judged universe grades**, not just returned grades, in descending order before
truncating at K. Nonnegative gains and positive ideal gain give scores in
$[0,1]$, subject to floating-point rounding. NDCG need not be monotonic in K:
both its numerator and denominator can change.

Exponential gains for grades 0, 1, 2 and 3 are 0, 1, 3 and 7. This models a
larger utility gap at higher grades. Neither gain nor logarithmic discount is
a verified user-behavior model for this example. The core uses `expm1` for
small positive grades, rejects gain overflow, and scales gains by a common
positive maximum before summing; that scale cancels in NDCG.

## Worked query and executable configuration

All data below are synthetic, with fully known judgments. No model produced
the lists. The initial order for both queries is `a,b,c,d,e,f`.

| Query | a | b | c | d | e | f | Reordered list |
|---|---:|---:|---:|---:|---:|---:|---|
| policy | 3 | 0 | 2 | 1 | 0 | 2 | a,c,d,b,e,f |
| returns | 0 | 0 | 2 | 1 | 3 | 1 | e,c,d,a,b,f |

For `policy`, K=5 contains relevant ranks 1, 3 and 4; $R_q=4$:

$$
P@5=3/5,\qquad \operatorname{Recall}@5=3/4,\qquad
AP@5=\frac{1+2/3+3/4}{4}=0.604166\ldots,\qquad RR@5=1.
$$

$$
DCG@5=7+\frac{3}{\log_2(4)}+\frac{1}{\log_2(5)},\qquad
IDCG@5=7+\frac{3}{\log_2(3)}+\frac{3}{\log_2(4)}+\frac{1}{\log_2(5)}.
$$

Their ratio is approximately 0.825122. The second query has AP@5 =
$(1/3+2/4+3/5)/4=0.358333\ldots$ and RR@5 = $1/3$. Equal-query averages give
MAP@5 = 0.48125 and MRR@5 = $2/3$. These exact fixtures define this study;
they are not borrowed benchmark observations.

The executed [README record](README.md) includes order changes, a K=1..6
table in the saved JSON, two AP conventions, two gain choices and a fixed
candidate probe. Each probe preserves the assumptions it is intended to
isolate. In particular, reordering leaves each query's top-five membership
unchanged, while the candidate probe deliberately restricts the universe.

## Candidate retrieval versus reranking

For the first query, pool `b,c,d,e` contains only two of four relevant items.
Its recall is 0.5. Ordering it by the known grades yields `c,d,b,e`; this
oracle uses labels deliberately to expose the pool's ceiling. A real model
must not use evaluation labels to generate predictions.

At K=3, evaluating only against pool labels gives ideal ordering and NDCG=1.
Against all six original judgments, missing grades 3 and 2 reduce NDCG to
0.349370. Candidate-conditioned recall is 1 while universe recall is 0.5.
Both calculations are internally consistent, but describe different tasks.

For final K no larger than pool size, end-to-end recall cannot exceed pool
recall. RR has no corresponding bound: one retrieved relevant item at rank
one can give RR=1 even with poor pool recall. Report candidate recovery and
final ordering against the same declared universe alongside any conditional
reranking score. Never infer corpus-level recall from retrieved positives alone.

## Aggregation, incomplete labels and no-answer queries

Equal-query means give each query the same weight. Micro recall instead divides
total hits by total relevant items, giving queries with more positives greater
weight. Traffic-weighted means answer a different question again; this core
implements only equal-query aggregation. Keep the query population identical
across systems, including explicit empty results for failures.

When $R_q=0$, recall and AP have zero denominators and ideal DCG is zero.
The default raises. The explicit `zero` convention returns zero for all
metrics and retains the query in the mean. It does not mean abstention failed;
no-answer queries need their own supported-answer/abstention evaluation.
Silent exclusion changes the population and is deliberately unsupported.

Real relevance labels are often sparse, subjective or exposure-biased.
Rejecting an unjudged returned ID prevents silently treating it as irrelevant,
but the evaluator cannot verify that the supplied universe is exhaustive.
With incomplete qrels, describe recall as recovery of known positives. Record
annotation coverage, rubric, adjudication, evidence identity and sampling.
Clicks alone are not unbiased relevance labels.

The API evaluates an existing order and accepts no scores. Resolve ties before
calling it, for example by item ID after descending model score, or choose a
tie-aware evaluator. scikit-learn averages over score ties by default. The
parity tests use distinct scores and transform labels for exponential gain.
Its classifier `average_precision_score` can agree with untruncated per-query
AP on a complete list with distinct scores; pooling all query-document pairs
does not compute grouped MAP, and truncation/ties can change the equivalence.

## Applied choices and unrun extensions

For one-answer lookup, inspect MRR with no-answer behavior. For multi-source
RAG, inspect evidence coverage and graded ordering at the actual context
budget, plus citation support and answer quality. For recommendations, inspect
relevance alongside diversity, coverage and outcomes. These are evaluation
design considerations, not measured product improvements here.

**Suggested, not executed:** evaluate real, independently judged queries;
segment by intent and history; vary judgment completeness or semantic duplicate
policies; compare paired per-query differences with a justified query-level
bootstrap; measure latency and downstream grounding. Keep tuning and final
evaluation queries separate, respecting time, user and document boundaries.
Two deliberately constructed queries cannot support uncertainty or significance
claims. Interpretation candidates in the execution record remain **author
review required**.
