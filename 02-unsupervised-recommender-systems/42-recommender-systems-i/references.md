# References and provenance

Primary sources consulted on November 6, 2026:

- Google for Developers, [Collaborative filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics).
  Feedback matrices and a conceptual introduction to collaborative embeddings.
- Badrul Sarwar, George Karypis, Joseph Konstan and John Riedl (2001),
  [Item-Based Collaborative Filtering Recommendation Algorithms](https://files.grouplens.org/papers/www10_sarwar.pdf).
  Item similarity, neighborhood aggregation and the motivation for item-based CF.
- Yifan Hu, Yehuda Koren and Chris Volinsky (2008),
  [Collaborative Filtering for Implicit Feedback Datasets](https://yifanhu.net/PUB/cf.pdf).
  Preference/confidence separation and a weighted implicit-factor objective.
  The implicit model is discussed, not implemented in this study.
- Plotly, [Interactive HTML export](https://plotly.com/python/interactive-html-export/).
  Embedded JavaScript for standalone offline exploration; consulted for the visual lab.

The shrinkage rule and fallback ladder in this topic are explicit educational
design choices, not claims to reproduce a particular production library.

## Data provenance

All ratings, names, item labels and timestamps in [example.py](example.py) are
hand-authored synthetic teaching data. There are no real users, downloaded
ratings, external dataset files or observed exposure logs. Artificial preference
groups are intentionally easy to inspect and do not represent a sampled
population. No benchmark or online-performance claims follow from them.

The separate [visual lab](VISUAL_LAB.md) uses seed-42 synthetic preference groups
and disjoint observation/test masks from [recommender_visual_core.py](recommender_visual_core.py).
Its complete latent preference table is generator-known teaching ground truth,
not observed real-user behavior. Exposure and event-count simulations are
separately declared mechanisms, not logs collected from a service.
