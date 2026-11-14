# Propensity scores: identification before adjustment

## From an observational contrast to a causal question

Define eligible units, assignment time, treatment versions, baseline features,
outcome horizon and the target population before fitting a model. For a
selective assistant rollout, baseline task complexity may affect both adoption
and performance. Features recorded after enablement can be mediators or
colliders; adding them to a propensity model can invalidate the adjustment.
The [previous study](../49-causal-inference-i-dags-confounders/) explains the
graphical selection of common causes.

Let $T\in\{0,1\}$, baseline covariates $X$, and potential outcomes $Y(0),Y(1)$.
Consistency gives $Y=TY(1)+(1-T)Y(0)$. The main targets are

$$
\operatorname{ATE}=E[Y(1)-Y(0)],\qquad
\operatorname{ATT}=E[Y(1)-Y(0)\mid T=1].
$$

For observational identification, assume conditional exchangeability
$(Y(0),Y(1))\perp T\mid X$, positivity on the target support, and a well-defined
intervention with consistency and no interference for this unit-level design.
These assumptions do not follow from a successful fit or a balance table.
Team-level spillovers may require a different unit and design.

## What a propensity score balances

The assignment probability is $e(X)=P(T=1\mid X)$. Under the true score,
$T\perp X\mid e(X)$: conditioning on the score balances measured baseline
covariates in the population. Estimated scores and finite nearest-neighbor
matches do not guarantee this property in an observed sample. The foundational
result is due to [Rosenbaum and Rubin (1983)](https://doi.org/10.1093/biomet/70.1.41).

A model can predict adoption accurately because treatment was almost
deterministic. That can leave few credible comparators. Model selection should
consider measured balance and support under a causally justified feature set,
without selecting a specification because its outcome estimate is attractive.
This example deliberately fits a prespecified simple parametric assignment
model on all observations. It is not a supervised prediction benchmark or a
cross-fitted semiparametric inference procedure.

## Matching and its population

For treated observation $i$, select the control minimizing

$$
j(i)=\arg\min_{j:T_j=0}|\operatorname{logit}(\hat e_i)
                           -\operatorname{logit}(\hat e_j)|,\qquad
\operatorname{logit}(p)=\log p-\log(1-p).
$$

With replacement, a control can be selected for several treated units. Ties
choose the first control in original row order. The transparent implementation
allocates one control-distance vector per treated unit instead of a full
pairwise matrix. Its quadratic time remains unsuitable for large cohorts.

With all treated retained, the estimator is

$$
\widehat{\tau}_{\mathrm{match}}=
\frac{1}{n_T}\sum_{i:T_i=1}(Y_i-Y_{j(i)}).
$$

For caliper matching, accept a pair only when its absolute logit distance is
at most 0.2 in this study. This is an explicit illustrative design choice, not
a universal optimal threshold or a standard-deviation rule. The mean then
runs over retained treated units. Dropping treated units changes the population
being described; label the result as a retained-treated contrast rather than
silently claiming ATT for every treated unit. The code computes the mean known
synthetic individual effect over those retained rows as its diagnostic target.
That target is sample/design dependent and does not establish transportability.

Frequency weights assign each retained treated unit weight one and each
control its number of uses. Weighted arm means then equal the mean paired
contrast. Treating distinct reused controls as equally weighted would give a
different estimator. A close scalar-score match also need not be a close
multivariate match.

## IPW and normalization

ATE weights are $w_i=T_i/\hat e_i+(1-T_i)/(1-\hat e_i)$. The separately
normalized, Hájek-style contrast implemented here is

$$
\widehat{\tau}_{\mathrm{ATE}}=
\frac{\sum_i T_iY_i/\hat e_i}{\sum_i T_i/\hat e_i}
-\frac{\sum_i(1-T_i)Y_i/(1-\hat e_i)}{\sum_i(1-T_i)/(1-\hat e_i)}.
$$

To see the identification mechanism, use the true propensity and iterated
expectation: $E[TY/e(X)\mid X]=E[Y\mid T=1,X]=E[Y(1)\mid X]$ under the stated
assumptions. Averaging identifies the treated potential-outcome mean; the
control identity follows similarly. Normalized finite-sample ratios are not
exactly unbiased and estimated probabilities add error.

ATT weighting uses $w_i=T_i+(1-T_i)\hat e_i/(1-\hat e_i)$, comparing the
unweighted treated mean with the normalized odds-weighted control mean.
Including ATT weighting makes a comparison with matching more directly aligned
in estimand when no treated units are dropped.

Multiplying all weights within an arm by a constant cancels in these normalized
contrasts. Stabilized ATE weights therefore give the same point estimate here;
they do not remove within-arm weight concentration. Clipping probabilities or
truncating weights changes the procedure and can introduce bias. Trimming
observations can change the target population. Neither silently restores
support. This implementation performs none of those operations and rejects
scores outside the open interval $(0,1)$.

## Diagnostics and uncertainty

For each covariate $k$, the implementation uses the **original unweighted**
pooled sample standard deviation for every adjustment:

$$
s_{k,\mathrm{ref}}=\sqrt{\frac{s^2_{T,k}+s^2_{C,k}}{2}},\qquad
\operatorname{SMD}_k(w)=
\frac{\bar X_{T,k}(w)-\bar X_{C,k}(w)}{s_{k,\mathrm{ref}}}.
$$

Holding the denominator fixed avoids making balance appear improved simply
because a new sample has a larger SD. This convention is explicit; ATT designs
may instead prespecify a treated-population reference SD. Signed SMDs expose
direction. The common $|\mathrm{SMD}|<0.1$ heuristic is no causal guarantee;
means alone can conceal distributional imbalance and missing interactions.
Zero reference variance returns zero for equal means and signed infinity for
unequal means, rather than hiding perfect separation in a constant covariate.

For each arm, report

$$
\operatorname{ESS}(w)=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

For matching controls, this summarizes reuse concentration; it is not a
matched-design degrees-of-freedom calculation. A reused control cannot provide
hundreds of independent outcomes. The example also prints score quantiles,
largest weights and largest matching distance. The fraction outside the
intersection of empirical score ranges is a descriptive sample diagnostic,
not a positivity test; overlapping ranges can still contain sparse regions.

This code supplies point estimates only. Matching uncertainty must account for
control reuse and the matching rule. IPW uncertainty must account for estimated
scores and the sampling structure. Independent-pair formulas and ordinary
unweighted standard errors are inadequate for these designs. ESS is not a
confidence interval. The simulation SD measures error variation across generated
datasets; MCSE of mean error is that SD divided by the square root of the
number of repetitions. It describes precision of a simulation summary rather
than uncertainty for a particular observational effect.

## Executed synthetic study — 2026-11-12

**Status:** executed locally by Codex. The two interpretations below received
a scoped AI-assistant technical review on November 13, 2026
([50-A/B](../../docs/CURATION.md)), not personal author approval. Numerical
records are unchanged; the separate visual interpretations remain pending.

**Hypothesis:** with measured confounding and adequate practical overlap,
matching and IPW will reduce the unadjusted error in this generator. Increasing
assignment strength may reduce effective information and degrade adjustment,
even though theoretical positivity still holds for finite covariates.

**Configuration:** Windows x64; stable Python 3.11.14 in the existing ignored
`.local/environments/s08-reference-2026-10-08` environment; NumPy 2.4.6,
SciPy 1.17.1, scikit-learn 1.9.1. Run from the repository root:

```powershell
.local/environments/s08-reference-2026-10-08/Scripts/python.exe 03-statistics-experimentation/50-causal-inference-ii-propensity-score-matching/example.py --n-samples 4000 --seed 50 --repetitions 10
```

There are 4,000 independent units per dataset, 10 seeds (50–59), and assignment
strengths $a\in\{1,3\}$. Generate two independent standard-normal baseline
covariates and independent standard-normal outcome noise:

$$
e_a(X)=\operatorname{expit}(-0.4+a(1.1X_0+0.8X_1)),\qquad
T=\mathbf{1}\{U<e_a(X)\},\quad U\sim\operatorname{Uniform}(0,1),
$$

$$
Y(0)=3+2X_0+1.2X_1+\epsilon,\qquad Y(1)=Y(0)+2.
$$

Both covariates are observed pre-treatment common causes, and assignment noise
is independent of potential outcomes conditional on them. The continuous
outcome has arbitrary units. Seeds reuse covariates, uniform draws and outcome
noise across strengths, making the scenario comparison paired. Uniform-based
assignment preserves the Bernoulli mechanism; it is a deliberate implementation
choice, not a reproduction of numerical output from the unexecuted source.
The source had no numerical results to preserve.

Propensities are fitted separately for each dataset using
`LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10)` on both
covariates, with default L2 regularization and an intercept. Convergence warnings
raise errors. No outcomes enter the score model. Matching uses replacement;
the separate caliper alternative uses absolute logit distance 0.2. No score
clipping, weight truncation, data trimming or inference procedure is used.

**First-seed results:** all causal diagnostic targets equal 2.0 in this
constant-effect generator. C/T indicates control/treated; SMD lists $X_0,X_1$.

| Strength | Method | Estimate | Signed SMDs | ESS C/T | Maximum weight |
|---:|---|---:|---|---|---:|
| 1 | Unadjusted | 4.3450 | 0.9040, 0.6012 | 2305.0 / 1695.0 | 1.00 |
| 1 | Matching ATT | 1.9670 | 0.0123, −0.0153 | 342.1 / 1695.0 | 25.00 |
| 1 | Caliper retained-treated | 1.9703 | 0.0141, −0.0196 | 353.5 / 1682.0 | 25.00 |
| 1 | IPW ATE | 2.0664 | 0.0087, −0.0030 | 1402.2 / 971.2 | 34.07 |
| 1 | IPW ATT | 1.9430 | −0.0542, 0.0025 | 507.9 / 1695.0 | 33.07 |
| 3 | Unadjusted | 5.3939 | 1.4961, 0.9410 | 2147.0 / 1853.0 | 1.00 |
| 3 | Matching ATT | 2.2023 | 0.0003, 0.2160 | 12.4 / 1853.0 | 504.00 |
| 3 | Caliper retained-treated | 2.0118 | 0.0218, −0.0258 | 75.7 / 1406.0 | 78.00 |
| 3 | IPW ATE | 2.2463 | 0.1177, −0.0418 | 453.1 / 33.1 | 674.43 |
| 3 | IPW ATT | 2.6376 | 0.3132, 0.1162 | 91.1 / 1853.0 | 87.70 |

At strength 1, full matching used 795 distinct controls and had maximum logit
distance 0.5306. The caliper retained 1,682 of 1,695 treated; 1.25% of all units
were outside the empirical common score range. At strength 3, full matching
used 393 distinct controls and maximum distance 8.3071; the caliper retained
1,406 of 1,853 treated and 17.40% of all units were outside that range. For
matching rows, maximum weight is a control-reuse count, not an IPW magnitude.

**Repeated results:** error equals estimate minus the sample-specific known
target. For the unadjusted association, the reference is ATE as a diagnostic.
Here every reference is 2.0. RMSE and error summaries are in outcome units.

| Strength | Method | Mean error | Error SD | Mean-error MCSE | RMSE |
|---:|---|---:|---:|---:|---:|
| 1 | Unadjusted | 2.3119 | 0.0476 | 0.0150 | 2.3124 |
| 1 | Matching ATT | −0.0025 | 0.0603 | 0.0191 | 0.0573 |
| 1 | Caliper retained-treated | 0.0007 | 0.0538 | 0.0170 | 0.0511 |
| 1 | IPW ATE | 0.0431 | 0.0859 | 0.0272 | 0.0922 |
| 1 | IPW ATT | 0.0339 | 0.0890 | 0.0281 | 0.0909 |
| 3 | Unadjusted | 3.4053 | 0.0688 | 0.0218 | 3.4059 |
| 3 | Matching ATT | 0.3060 | 0.2573 | 0.0814 | 0.3915 |
| 3 | Caliper retained-treated | 0.0525 | 0.0633 | 0.0200 | 0.0797 |
| 3 | IPW ATE | 0.5973 | 0.3836 | 0.1213 | 0.6994 |
| 3 | IPW ATT | 0.4938 | 0.1699 | 0.0537 | 0.5194 |

**Scoped technical interpretations:**

- **50-A — Adjustment and overlap:** adjusted methods had lower recorded RMSE
  than the unadjusted association in both scenarios. Stronger assignment
  produced greater repeated error for each adjusted method and concentrated
  effective information; this is conditional on the known confounders and
  correctly specified logistic functional family. Ten replications of this
  generator do not establish a general method ranking or causal identification
  in observational data.
- **50-B — Retention and reuse:** the caliper improved the displayed
  first-seed balance and reduced reuse relative to unrestricted matching, but
  excluded treated units. Its target is the retained-treated population, not
  automatically the original ATT. The constant effect makes all target truths
  equal here, concealing the practical cost of that change. Matching frequency
  weights and IPW weights have different meanings; neither ESS nor balance
  certifies identification or authorizes a rollout.

**Limitations:** ten repetitions provide only modest Monte Carlo precision.
The data favor the correct logistic functional family, although regularization
remains. There are no hidden common causes, clustering, interference, missing
data, longitudinal assignment, model misspecification study or observational
confidence intervals. The observed weight maximum alone does not summarize
all tail risk. Weakened practical overlap is not structural nonpositivity.

The numerical tests additionally check an exact two-stratum heterogeneous
fixture (ATE 4, ATT 5), matching agreement with scikit-learn, frequency weights,
SMD conventions and invalid inputs. Those are software checks, separate from
the repeated experiment and author interpretation review.

## Further experiments — not executed as empirical studies

- Use `--heterogeneity 1` to set $\tau(X)=2+X_0$; compare sample ATE, ATT and
  the retained-treated truth rather than comparing all estimates with 2.
- Add nonlinear assignment terms and deliberately omit them from the model;
  inspect transformations and distributional balance rather than only means.
- Prespecify alternative calipers or trimming populations and report retention,
  balance and target changes. Do not select the threshold using outcome effects.
- Add unmeasured-confounding sensitivity and appropriate design-aware
  uncertainty before interpreting an observational dataset.

## Senior interview discussion

**Why can matching and IPW disagree?** Check estimands first: matching usually
targets treated units, ATE weights the eligible population, and calipers may
restrict the treated population. Then inspect confounding assumptions, residual
imbalance, reused controls, score estimation and weight tails.

**Does a good assignment model establish causality?** It establishes neither
exchangeability nor consistency. AUC evaluates prediction; balance describes
measured covariates. Important unmeasured common causes remain unaddressed.

**Would you split a propensity model like a predictive model?** This simple
parametric point-estimation study uses all observations. Flexible nuisance
learning can require sample splitting/cross-fitting for particular inferential
guarantees. A random holdout alone does not validate causal identification.

**What does double robustness add?** AIPW combines assignment and outcome
models; with identification and regularity conditions, consistency can hold
when one nuisance model is correct. It does not rescue missing confounders or
failed positivity and is outside this implementation.

## Primary references

- [Rosenbaum and Rubin (1983), The central role of the propensity score in
  observational studies for causal effects](https://doi.org/10.1093/biomet/70.1.41)
  — foundational balancing-score result.
- [Hernán and Robins, Causal Inference: What If](https://miguelhernan.org/whatifbook)
  — potential outcomes, identification and inverse probability weighting.
- [scikit-learn LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  — assignment-model API and regularization behavior.

For an applied methodological overview, see
[Austin (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3144483/).
