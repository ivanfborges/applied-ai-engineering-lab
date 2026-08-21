# Senior Interview Questions

## 1. What is the difference between probability and likelihood?

With parameters fixed, \(p(x\mid\theta)\) is a probability model over possible
observations. Once the data are fixed, the same expression considered as a
function of \(\theta\) is a likelihood. A likelihood ranks how well candidate
parameters explain the observations; it is not generally normalized over the
parameter space.

## 2. Why optimize log-likelihood?

For independent observations, likelihood multiplies per-observation terms.
Logarithms turn the product into a sum, reduce underflow risk, and simplify
differentiation. Because logarithm is strictly increasing, the maximizer is
unchanged.

## 3. How do MLE and MAP differ?

MLE maximizes \(p(D\mid\theta)\). MAP maximizes
\(p(D\mid\theta)p(\theta)\), so it combines data fit with a parameter prior.
Both return point estimates. MAP uses a Bayesian posterior but does not retain
the full posterior distribution.

## 4. Why is cross-entropy related to maximum likelihood?

For Bernoulli or categorical output models, the negative log-probability
assigned to observed labels is binary or multiclass cross-entropy. Minimizing
that loss is conditional MLE, assuming the probabilistic output model and data
factorization are appropriate.

## 5. Under what assumption does squared error correspond to MLE?

Assume outcomes equal the model's conditional mean plus independent Gaussian
noise with fixed variance. The Gaussian NLL then differs from summed squared
error only by constants and positive scaling. Heteroscedastic or heavy-tailed
errors require a different likelihood to preserve that interpretation.

## 6. How does a Gaussian prior produce L2 regularization?

The negative log-density of a zero-mean Gaussian prior is proportional to
\(\lVert w\rVert_2^2/(2\sigma_w^2)\). Adding it to NLL yields the negative
log-posterior. The correspondence requires clarity about which parameters have
the prior and whether the complete objective is summed or averaged.

## 7. Why can sum-versus-mean reduction change the MAP interpretation?

Dividing the complete negative log-posterior by \(n\) preserves its optimizer.
Dividing only the data NLL while keeping the same penalty coefficient makes the
prior \(n\) times stronger relative to the likelihood. Dataset and batch
scaling therefore affect the implied prior unless coefficients are adjusted.

## 8. Is AdamW weight decay equivalent to Gaussian-prior MAP?

Not automatically. An explicit quadratic term in the optimized objective has
a direct Gaussian-prior interpretation. AdamW decouples shrinkage from the
adaptive loss gradient, changing optimizer dynamics; it is not universally
the optimizer for that same explicit MAP objective.

## 9. When might MAP be preferable to MLE?

MAP can stabilize weakly identified or high-variance parameters when data are
limited and the prior is defensible. The trade-off is prior-dependent bias.
Prior sensitivity and validation are essential because a strong incorrect
prior can dominate limited evidence.

## 10. Why is MAP not full Bayesian inference?

MAP reports only the posterior mode. Two posteriors can share a mode while
having very different width, skewness, correlation, or multimodality. Full
Bayesian inference retains the posterior and integrates it for uncertainty and
prediction.

## 11. Why is MAP not invariant to reparameterization?

MAP chooses the mode of a density. Under a nonlinear transformation, density
values receive a Jacobian factor, so transforming the original mode need not
produce the transformed density's mode. This is one reason not to treat a
posterior mode as a complete posterior summary.

## 12. How does MLE connect to language-model pretraining?

An autoregressive model factorizes a sequence into conditional next-token
probabilities. Summing the negative log-probability of each observed next token
is token-level cross-entropy and therefore a conditional MLE-style objective.
This statistical framing does not, by itself, guarantee factuality,
calibration, alignment, or downstream utility.

## 13. What limitation does more data not automatically fix?

Model misspecification. MLE and MAP select parameters inside the chosen family.
If that family cannot represent the data-generating mechanism, more samples
can increase confidence in the best available approximation without making the
model structurally correct.

## 14. What would you inspect before giving a regularized model a MAP interpretation?

Identify the exact likelihood, prior family, parameters covered by the prior,
objective reduction, coefficient scaling, optimizer update, and treatment of
the intercept. Then check whether batching or dataset-size changes alter the
relative prior strength and whether the prior is defensible for the application.
