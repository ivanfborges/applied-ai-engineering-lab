# References

Original research:

- Chen and Guestrin (2016), [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754): regularized tree objectives and scalable split search.
- Ke et al. (2017), [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html): GOSS and EFB. Its benchmark claims belong to that paper, not this study.
- Prokhorenkova et al. (2018), [CatBoost: unbiased boosting with categorical features](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), with an [expanded paper](https://arxiv.org/abs/1810.11363): ordered statistics and prediction shift.

Official documentation used to review implementation choices:

- XGBoost: [objective derivation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html), [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [scikit-learn interface and early stopping](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html), and [categorical features](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).
- LightGBM: [features and growth strategy](https://lightgbm.readthedocs.io/en/stable/Features.html), [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [categorical support](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), [classifier API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html), and [early-stopping callback](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html).
- CatBoost: [training parameters and defaults](https://catboost.ai/docs/en/references/training-parameters/common) and [fit API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
- scikit-learn: [synthetic classification generator](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) and [probability calibration concepts](https://scikit-learn.org/stable/modules/calibration.html).

All data in the implemented examples are generated synthetically. The generator
reference documents an API, not a public dataset source. Stable documentation
can change; executed package versions are recorded in the experiment summaries
and ignored JSON records.
