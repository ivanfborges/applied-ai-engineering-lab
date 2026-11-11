# References and implementation conventions

Primary references checked on November 6, 2026. External documentation may
evolve; the executed record uses scikit-learn 1.9.1. No public dataset is used:
all item IDs, grades and orders are defined synthetically in [example.py](example.py).

- Järvelin and Kekäläinen (2002), [Cumulated gain-based evaluation of IR
  techniques](https://doi.org/10.1145/582415.582418), ACM TOIS 20(4), 422–446.
  Foundational reference for cumulative and discounted graded-relevance
  evaluation. [Author institution's publication record](https://researchportal.tuni.fi/en/publications/cumulated-gain-based-evaluation-of-ir-techniques/)
  confirms the bibliographic details; this study does not claim to reproduce
  the paper's experiments.
- Apache Spark, [ranking evaluation guide](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#ranking-systems)
  and [RankingMetrics source](https://raw.githubusercontent.com/apache/spark/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala).
  Documents fixed-K precision for short lists; the source makes the
  min(number of relevant items, K) truncated-AP denominator explicit. Spark is
  a convention reference, not an installed dependency or executed comparison.
- scikit-learn, [ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).
  Reference for direct relevance gain, ideal normalization, score-induced
  ordering and tie averaging. The executable parity check passes transformed
  labels for exponential gain and distinct scores to avoid tie ambiguity.
- scikit-learn, [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
  Threshold-based classifier AP. One test checks agreement with complete-list
  per-query AP under aligned binary labels and unique scores; this API does
  not by itself provide query-grouped MAP@K.

The [topic notes](notes.md) specify every implemented formula and policy.
The [previous study](../43-recommender-systems-ii/) connects these choices to
candidate retrieval and implicit-feedback recommendations.
