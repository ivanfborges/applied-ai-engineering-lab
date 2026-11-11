# Ranking metrics: interview questions

**Why can P@5 and Recall@5 stay constant while MAP@5 improves?**
Reordering the same top-five items preserves hit counts and denominators.
AP also uses the precision at each relevant rank, so earlier relevant results
can improve it. MRR changes only if the first relevant position changes.

**Does NDCG@5 = 1 mean all relevant evidence was found?**
No. It means maximum discounted gain at that cutoff within the declared
judgment universe. Relevant items may remain below K, and a restricted
candidate-universe evaluation may omit important items entirely.

**Why do two MAP@K implementations disagree?**
Check query membership, relevance threshold, truncation, denominator R versus
min(R,K), short-list handling, duplicates, ties, no-positive policy and weights.
This example exposes both AP denominators and defaults to R. MAP is a mean
of query-level AP, not one AP over pooled documents from unrelated queries.

**What does a perfect reranker fail to repair?**
Missing candidates. Reordering a fixed pool cannot recover items outside it.
Measure candidate recall against independent judgments and final top-K quality
against the same universe. An oracle using labels diagnoses a ceiling; it is
not evidence of learned predictive performance.

**How would you handle queries with no relevant answer?**
Declare a scoring policy and report the population. This core raises by
default or explicitly retains zero-scored queries. Separately measure whether
the application abstains correctly and avoids unsupported answers. Excluding
these queries silently can change the meaning of an aggregate score.

**Why does scikit-learn give a different NDCG?**
Its direct gains are the supplied relevance values. To match exponential gain,
supply 2**grade - 1 as relevance, use the same full universe and cutoff, and
align score-tie handling. Predicted scores determine order; they are not
ground-truth relevance. The tests verify these aligned configurations.

**Would a 0.01 NDCG improvement justify deployment?**
That number alone is insufficient. Compare paired query outcomes, annotation
quality, uncertainty and regressions by segment. Also inspect latency, costs,
guardrails and downstream outcomes. None of those measurements is established
by this synthetic example.
