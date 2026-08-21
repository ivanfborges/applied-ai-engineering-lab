# Technical Notes

## From probability to likelihood

For a model \(p(x \mid \theta)\), fixing \(\theta\) produces a probability
distribution over possible observations. After observing data \(D\), the same
expression can be viewed as a function of candidate parameters:

\[
L(\theta; D) = p(D \mid \theta).
\]

Likelihood is not generally normalized over \(\theta\), so it is not a
frequentist probability distribution for the parameter. For IID observations,

\[
L(\theta;D)=\prod_{i=1}^{n}p(x_i\mid\theta), \qquad
\ell(\theta;D)=\sum_{i=1}^{n}\log p(x_i\mid\theta).
\]

Because logarithm is strictly increasing, likelihood and log-likelihood have
the same maximizers. The sum is also numerically safer and easier to
differentiate. MLE can therefore be written as

\[
\hat\theta_{\mathrm{MLE}}
=\arg\max_\theta \ell(\theta;D)
=\arg\min_\theta -\ell(\theta;D).
\]

The gradient \(\nabla_\theta\ell(\theta)\) is the score. Setting it to zero
characterizes an interior optimum only when differentiability and other
regularity conditions hold; it does not guarantee a unique global maximum.

## Bernoulli MLE and Beta-Bernoulli MAP

For \(k\) successes in \(n\) independent Bernoulli trials,

\[
\ell(p)=k\log p+(n-k)\log(1-p).
\]

At an interior optimum, differentiating and solving gives

\[
\hat p_{\mathrm{MLE}}=\frac{k}{n}.
\]

With a \(\operatorname{Beta}(\alpha,\beta)\) prior,

\[
p\mid D \sim
\operatorname{Beta}(\alpha+k,\beta+n-k).
\]

When both posterior shape parameters exceed one, the unique interior mode is

\[
\hat p_{\mathrm{MAP}}=
\frac{k+\alpha-1}{n+\alpha+\beta-2}.
\]

The formula is not universally valid at the boundary. A Beta distribution can
be monotone, uniform, or U-shaped, so code must handle boundary and non-unique
modes explicitly. The posterior mean,

\[
\mathbb E[p\mid D]=\frac{k+\alpha}{n+\alpha+\beta},
\]

is a different estimator and should not be mislabeled as MAP.

## MAP as a regularized objective

Bayes' theorem gives

\[
p(\theta\mid D)\propto p(D\mid\theta)p(\theta).
\]

Consequently,

\[
\hat\theta_{\mathrm{MAP}}
=\arg\min_\theta\left[-\log p(D\mid\theta)-\log p(\theta)\right].
\]

For a zero-mean isotropic Gaussian prior on selected weights,

\[
w\sim\mathcal N(0,\sigma_w^2I), \qquad
-\log p(w)=\frac{\lVert w\rVert_2^2}{2\sigma_w^2}+C.
\]

A Laplace prior similarly produces an L1-type penalty. This correspondence is
about an explicit probabilistic objective. Decoupled optimizer weight decay
changes update dynamics and should not automatically be described as the same
Gaussian-prior MAP problem.

### Sum-versus-mean scaling

The exact summed negative log-posterior is

\[
J_{\mathrm{sum}}(w)=
\sum_{i=1}^{n}\operatorname{NLL}_i(w)
+\frac{\lVert w\rVert_2^2}{2\sigma_w^2}.
\]

Dividing the complete objective by \(n\) preserves its optimizer:

\[
J_{\mathrm{mean}}(w)=
\frac{1}{n}\sum_{i=1}^{n}\operatorname{NLL}_i(w)
+\frac{\lVert w\rVert_2^2}{2n\sigma_w^2}.
\]

Using mean NLL but leaving the penalty unscaled instead changes the prior's
strength relative to the dataset. This is a frequent source of inconsistent
regularization when training code changes reduction, dataset size, or batching.

## Common ML losses as likelihoods

### Gaussian regression

Assuming independent residuals with fixed variance,

\[
y_i\mid x_i,\theta\sim
\mathcal N(f_\theta(x_i),\sigma^2),
\]

the NLL differs from the sum of squared residuals only by positive scaling and
constants independent of \(\theta\). Squared error therefore follows from a
homoscedastic Gaussian noise model. Heavy tails, heteroscedasticity, or an
incorrect conditional mean challenge that interpretation.

### Bernoulli and categorical outputs

For binary labels with \(p_i=p_\theta(y_i=1\mid x_i)\),

\[
-\ell(\theta)=-\sum_i
\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],
\]

which is binary cross-entropy. For categorical labels, the observed-class NLL
is \(-\sum_i\log p_\theta(y_i\mid x_i)\), the usual multiclass
cross-entropy. Autoregressive language-model pretraining applies the same
conditional-likelihood idea across next-token predictions.

MLE is not identical to empirical risk minimization in general. The two align
when the per-example loss is a negative log-likelihood; ranking, focal, and
some contrastive objectives may modify or depart from that form.

## What MLE and MAP do not provide

- **Full posterior uncertainty:** MAP retains one posterior mode, not credible
  intervals, correlations, multimodality, or posterior predictive integration.
- **Model correctness:** both methods optimize within the specified family.
  More data can identify the best approximation within a misspecified family
  without making the family correct.
- **Causal validity:** likelihood optimization does not turn observational
  prediction into an intervention effect.
- **Optimization guarantees:** neural objectives can contain symmetries, flat
  directions, saddle points, and many equivalent parameterizations.
- **Design validity:** leakage, selection bias, dependence, and invalid splits
  remain threats.

MAP modes also depend on parameterization because transformed densities acquire
a Jacobian factor. A prior that is uniform in \(p\) is not uniform in
\(\log(p/(1-p))\). Treating the posterior mode as a uniquely privileged summary
can therefore hide an important modeling choice.

## Further experiments

The visual lab executes deterministic single-path demonstrations of symmetric
prior strength, a misspecified Beta(20, 2) prior, and logistic coefficient
shrinkage across prior scales. Those computed observations are recorded in the
topic README. The following extensions remain proposed and unrun:

1. Repeat the prior-strength and misspecified-prior demonstrations across many
   sample paths to quantify estimator variability rather than relying on one
   seed.
2. Evaluate several logistic prior scales on repeated train/validation draws
   and report variation rather than selecting a conclusion from one split.
3. Contrast summed NLL plus a fixed penalty with mean NLL plus the same numeric
   penalty, then restore equivalence by scaling the complete objective.
4. Compare a posterior mode with posterior intervals or posterior predictive
   quantities in a small conjugate model.
