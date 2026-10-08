# Naive Bayes interview questions

## What is the "naive" assumption?

Generic Naive Bayes factorizes class-conditional feature likelihoods. In Multinomial text NB, the working model treats token occurrences as class-specific draws given document length; token-count coordinates are constrained by their sum. Natural-language dependencies routinely violate the simplified model. The relative scores can still be useful for classification, but the predicted probabilities may be too extreme.

## Derive the Multinomial NB decision rule

Start with $P(c\mid x)\propto P(x\mid c)P(c)$. For token counts $x_j$ and class token probabilities $\theta_{jc}$, $P(x\mid c)$ is a class-independent multinomial coefficient times $\prod_j\theta_{jc}^{x_j}$. Taking logs and dropping that common coefficient gives $\log P(c)+\sum_j x_j\log\theta_{jc}$. Choose the class with the largest score.

## Why are smoothing and log space separate choices?

Smoothing makes a known vocabulary token's class likelihood positive even when its training count in that class is zero. Log space prevents products of many small positive numbers from underflowing. Logs alone cannot repair a zero likelihood.

## What happens when a new word appears at inference?

A word absent from the fitted vocabulary is normally omitted by the vectorizer. A word in the vocabulary but absent from one class has a nonzero smoothed likelihood. If no query words are known, the class prior alone decides.

## When would you choose Bernoulli or Complement NB?

Bernoulli NB is appropriate when binary presence and explicit absence are the intended evidence. Complement NB is a candidate for imbalanced text tasks. Test it on the relevant validation metric; imbalance alone does not establish superiority.

## How would you compare NB to logistic regression fairly?

Fit the same text preprocessing separately inside each training fold, tune model settings without touching the final test set, and use metrics tied to the decision cost. Report minority-class behavior where relevant. Compare probability calibration separately from class ranking or thresholded decisions.

## What would you monitor in a deployed ticket router?

Track unknown-token rates, vocabulary drift, class prevalence when labels arrive, predicted-class mix, and delayed error metrics. If a score is used as confidence, assess calibration on representative labeled data and document the decision threshold. A high model posterior alone does not certify a ticket is safe to route automatically.
