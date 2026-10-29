# K-Means interview questions

Use [the code and executed record](README.md) to substantiate the answers;
mathematical details are in [notes.md](notes.md).

1. **Why does the update use the mean?**
   With assignments fixed, the derivative of summed squared distances vanishes
   at the arithmetic mean. The positive-definite Hessian makes it the unique
   minimizer for a nonempty cluster. Replacing squared Euclidean distance by
   another dissimilarity can require a different update.

2. **Does decreasing inertia prove convergence to the best clustering?**
   No. Exact Lloyd steps do not increase the objective, but can settle at a local
   solution. Initialization and restarts matter. Stable assignments, tolerance
   stopping and an iteration cap also have different guarantees; inspect the
   final centers/labels contract.

3. **Why not choose the K with the lowest inertia?**
   Optimal inertia is non-increasing with K, eventually reaching zero with a
   center per distinct observation. Look for diminishing returns, inspect
   silhouette and profiles, assess stability, and validate the intended use.
   An elbow is a heuristic and may be absent.

4. **Is standardized clustering always better?**
   No. Standardization imposes inverse-variance feature weights; noise may gain
   influence, and correlated variables still repeat information. In this study's
   artificial scale mismatch it recovered the generator at K=3; that observation
   does not make it universally preferable. Compare partitions in a justified
   common geometry rather than comparing differently scaled inertia values.

5. **What does a high silhouette establish?**
   Cohesion and separation under a particular distance and population. It favors
   compact groups, needs at least two occupied clusters, and says nothing by
   itself about semantic meaning or downstream benefit. The raw and standardized
   sweeps here have different silhouette maxima.

6. **How do you compare runs if label zero becomes label two?**
   Compare partitions using ARI on the same evaluation rows, or match centers
   and profiles in a common representation. Direct equality of raw IDs is not
   appropriate. Multiple seeds and resampling answer different stability questions;
   neither study is included in the current experiment.

7. **What happens to empty clusters and outliers?**
   Empty clusters have no mean and need a policy. The teaching loop retains
   their previous centers, potentially leaving fewer occupied groups than K.
   Outliers can move means and consume centers because the objective squares
   distances. K-Means always assigns them; distance is not an anomaly probability.

8. **Does normalizing embeddings make K-Means a cosine clusterer?**
   Pairwise Euclidean and cosine rankings agree for unit vectors, but arithmetic
   centroids need not have unit or equal norms. Ordinary K-Means is therefore
   generally different from spherical K-Means. Evaluate semantics and retrieval
   separately rather than assuming the partition improves RAG.

9. **Where can unsupervised leakage occur?**
   A scaler or clusterer fitted using validation rows, future events or unavailable
   features can contaminate a downstream prediction evaluation. Fit all learned
   transforms on the training population and inside supervised folds as needed.
   The full-corpus example here makes only exploratory claims.

10. **How would you use and monitor this in a real system?**
    Clarify whether the purpose is segmentation, compression or coarse candidates.
    Validate profiles and downstream outcomes, preserve the preprocessing/model
    bundle, and monitor schemas, feature distributions, cluster sizes and
    distance profiles. For large corpora, evaluate batching and memory explicitly.
    Cluster IDs need matching across retraining. Grouping alone is not a
    personalized recommender.
