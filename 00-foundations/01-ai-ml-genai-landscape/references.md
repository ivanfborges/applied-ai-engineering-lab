# References and Further Reading

## Foundational Papers

- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Introduces the Transformer architecture and scaled dot-product attention.
- Lewis et al. (2020), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). Foundational formulation combining parametric generation with retrieved non-parametric memory.
- Sculley et al. (2015), [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). Explains why production ML complexity extends far beyond model code.
- Sutton and Barto (2018), [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/the-book-2nd.html). Standard reference for states, actions, rewards, policies, and value functions.

## Official and Authoritative Documentation

- scikit-learn, [Working With Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html). Text feature extraction, classifiers, pipelines, and evaluation.
- scikit-learn, [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). API used by `example.py`.
- scikit-learn, [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Classifier used by `example.py`.
- Google, [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml). Practical guidance on baselines, pipelines, objectives, and production iteration.
- NIST, [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). A risk-oriented framework for trustworthy AI design and governance.

## Books

- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022). System-level treatment of data, deployment, distribution shift, monitoring, and continual learning.
- Martin Kleppmann, *Designing Data-Intensive Applications* (O'Reilly, 2017). Foundations for reliable data systems underlying production AI.
- Christopher M. Bishop and Hugh Bishop, *Deep Learning: Foundations and Concepts* (Springer, 2024). Mathematical and conceptual deep-learning reference.

## Notes About This Module

The datasets in `example.py` and `from_scratch.py` are synthetic and embedded directly in the scripts. No public dataset is used, and no performance benchmark is claimed.
