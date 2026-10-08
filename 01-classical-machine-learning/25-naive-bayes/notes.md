# Naive Bayes: model and decisions

## From Bayes' rule to a class score

For class $c$ and an observed document $x$,

$$
P(c\mid x)=\frac{P(x\mid c)P(c)}{P(x)}.
$$

The evidence $P(x)$ is common to every candidate class for the same document, so choosing the most probable class only needs $P(x\mid c)P(c)$. The prior $P(c)$ can be estimated as the fraction of training documents in class $c$. If production prevalence differs from the training sample, this empirical prior may be inappropriate.

A generic Naive Bayes model simplifies the class-conditional feature distribution by assuming conditional independence. For text, the Multinomial variant has a more precise story: given a class and document length, it treats token occurrences as draws from a class-specific vocabulary distribution. The count features themselves are constrained to sum to the document length, so describing the counts as literally independent is imprecise.

Let $x_j$ be the count of vocabulary token $j$, and let $N_{jc}$ be its total training count in class $c$. With vocabulary size $V$ and additive smoothing $\alpha>0$,

$$
\theta_{jc}=\frac{N_{jc}+\alpha}{\sum_{k=1}^{V}N_{kc}+\alpha V}.
$$

The multinomial coefficient depends on the document counts but not on the candidate class. It therefore drops out of the class comparison:

$$
S_c(x)=\log P(c)+\sum_{j=1}^{V}x_j\log\theta_{jc},
\qquad \hat c=\arg\max_c S_c(x).
$$

A log turns a product of tiny likelihoods into a sum and avoids underflow. For two classes, subtracting their scores exposes each token's contribution: $x_j\log(\theta_{jA}/\theta_{jB})$. This class contrast is more useful than listing a class's most frequent words, since common words can be frequent in both classes.

## What smoothing does

Without smoothing, a vocabulary token with $N_{jc}=0$ would contribute zero likelihood for any document containing it. Setting $\alpha=1$ is Laplace smoothing; any positive $\alpha$ is additive smoothing. Larger values pull each class's token distribution toward uniformity, so the smoothing strength belongs in model selection on development data. It does **not** create a probability for arbitrary words outside the fitted vocabulary: the vectorizer ignores those words by default.

If every query token is outside the fitted vocabulary, the vector is all zeros. The score is then just the class log prior. That is a useful diagnostic for vocabulary drift, not semantic understanding.

## Choosing a variant

| Variant | Input and likelihood | Distinctive behavior |
|---|---|---|
| Multinomial | Nonnegative token counts; a class-specific token distribution | Repetition adds evidence |
| Bernoulli | Binary feature presence | Absence also contributes to the likelihood |
| Gaussian | Continuous features with class-specific univariate normal likelihoods | Requires a plausible continuous-feature model |
| Complement | Text statistics formed from classes other than the target class | Candidate to evaluate under imbalance; not an automatic fix |

TF-IDF is nonnegative and can be used with library Multinomial NB, but its weights are not literal token counts. Count vectors give the cleanest match to the derivation here. A logistic-regression text baseline uses the same representation while fitting a conditional boundary directly; it does not share the NB likelihood assumption. Neither model should be declared better without a task-specific comparison.

## Applied boundaries and common mistakes

- **Correlated evidence:** Related words can push the posterior to extreme values. Ranking quality and probability calibration are separate checks.
- **Compositional meaning:** Unigrams lose order and interactions such as negation. Bigrams can expose some phrases, at the cost of a larger, sparser vocabulary.
- **Leakage:** Fit vocabulary, token statistics, and any preprocessing choices on training folds only. A pipeline keeps vectorization inside each fit.
- **Imbalance:** Empirical priors follow the training mix; accuracy alone can hide poor minority-class recall. Choose the evaluation metric from the real decision cost.
- **Preprocessing:** Lowercasing, stop-word removal, token rules, and vocabulary cutoffs can remove signal. The example lowercases and uses unigram counts; these are demonstration choices, not tuned defaults.
- **Operations:** Watch unknown-token rates and input vocabulary changes. Review class prevalence and predicted-class distribution as labels arrive; measure calibration if probabilities control a decision.

The [first-principles code](from_scratch.py) intentionally uses dense matrices and positive smoothing, with no sparse optimization, calibration, custom prior, or incremental update. Its finite log scores and parity checks show the arithmetic; they do not validate any deployment claim.

## Suggested experiments

These have **not been run as comparative studies** for this topic:

1. Tune $\alpha$ within training folds and inspect both minority-class recall and confidence.
2. Compare unigram counts with unigram-plus-bigram counts, preserving the same validation split.
3. Compare count and TF-IDF representations, explicitly noting the change in model interpretation.
4. Under an intentionally imbalanced dataset, compare Multinomial and Complement NB using minority-class precision, recall, and F1.
5. Compare NB with logistic regression on the same fixed features; evaluate ranking and calibration separately.

For any later experiment, record the hypothesis, exact data and split, preprocessing, metric, observed result, interpretation candidate **for author review**, and limits.

## Executed mechanism check

- **Hypothesis:** With this vocabulary, billing terms should favor the billing class and access terms should favor the access class; the count implementation should match library arithmetic under the same smoothing and priors.
- **Configuration:** Eight code-defined synthetic training tickets, four per class; lowercase unigram counts; empirical priors; alpha = 1.0; two hand-written query tickets. The scratch parity test used a separate four-row, three-feature synthetic count matrix.
- **Result:** The example printed `billing` for "refund the wrong payment" and `access` for "password reset for account". The billing-minus-access log-likelihood contrasts were +1.099 for `invoice` and -1.099 for `password`. The focused parity test matched library class priors, feature log probabilities, and predictions; all 14 focused tests passed.
- **Interpretation candidate — for author review:** The signs of the token contrasts and query predictions are consistent with the intended evidence-accumulation mechanism.
- **Limit:** These queries were written to contain familiar training vocabulary. There is no held-out accuracy estimate, robustness study, or calibration assessment, and this check does not establish real-world ticket-routing performance.