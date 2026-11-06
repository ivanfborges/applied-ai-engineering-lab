# Interview questions — Recommender Systems I

### Why does a missing entry differ from a low rating?

A low rating is observed evidence of expressed dissatisfaction on a defined
scale. An absent entry can reflect no exposure, no consumption or no decision
to rate. The example preserves an observation mask and estimates no rating
from a fabricated zero. In an implicit model, zero may be a deliberately weak
target; its meaning follows the model's assumptions, not a proven dislike.

### What does high Pearson correlation establish with two co-rated items?

Very little reliability. With two nonconstant paired values, correlation has
magnitude one. Require sufficient support, examine overlap counts and consider
shrinkage. None of these controls turns correlation into a probability that a
recommendation will succeed.

### Why use different means for similarity and prediction?

Overlap-local means measure relative agreement on the same items. Full training
user means provide the baseline for the rating predictor. Confusing them changes
the algorithm. Replacing missing centered values with zero and calculating
cosine is also a different operation, with different normalization support.

### Does increasing shrinkage always make predicted ratings more conservative?

No. Prediction divides a weighted sum by its weight sum. If all neighbor weights
receive the same shrinkage factor, it cancels. Unequal overlap counts can change
relative influence or the selected neighborhood. Minimum overlap and the rule
for falling back are additional decisions.

### How does this implementation handle cold start?

It reports training item means for a new user. For a known new item it uses the
warm user's mean, or the global mean if both are cold. These are fallback
values, not collaborative personalization. The list generator excludes items
with no training ratings. Discovering new items needs another evidence or
exploration mechanism; new-platform cold start may require curated/content
recommendations because no aggregate rating baseline exists yet.

### Why can rating RMSE improve without better recommendations?

Rating error aggregates numerical discrepancies on selected evaluated pairs.
Recommendation depends on relative order near the top of an eligible catalog,
often for implicit engagement rather than expressed ratings. Define a ranking
protocol and product guardrails separately. The toy experiment measures neither
ranking quality nor online satisfaction.

### How would you prevent temporal leakage?

Specify a global prediction-time boundary and information-availability rules.
Build every mean, similarity, count and candidate statistic from earlier data
only. Keep policy tuning on validation data and reserve final test labels. In
this example all future ratings are hidden together; changing their values
cannot change any fitted state or prediction. Real delayed events and catalog
changes need additional handling.

### What would change at production scale?

The dense user-pair matrix uses quadratic user memory, and its naive fitting
cost grows with users squared times items. Consider sparse supported-neighbor
search, precomputed item neighborhoods or learned retrieval representations
when justified. Separate retrieval from scoring and re-ranking, enforce current
eligibility and monitor latency, freshness, concentration and segment quality.
Choose using actual data and measurements; this study implements none of those
serving components.

### Can collaborative document discovery replace semantic retrieval?

Behavioral affinities can supply personalization, while semantic retrieval
answers query relevance. Their objectives differ. Combine them only with a
clear evaluation and permission-aware candidate handling. A document being
popular among similar users never grants access to it.
