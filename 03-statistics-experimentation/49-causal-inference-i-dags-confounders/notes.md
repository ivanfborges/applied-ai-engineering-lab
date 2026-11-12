# Causal adjustment: identify first, estimate second

## The question and its assumptions

The unit is one synthetic case; $A=1$ means assistant enablement, and $Y$ is
an arbitrary performance score. We target $\mathbb{E}[Y(1)-Y(0)]$ over the
generated case population. Enablement is distinct from adoption $M$, which
happens after assignment. In real AI evaluation, treatment versions, observation
window, team interference and outcome measurement must be specified before
this estimand is meaningful.

A DAG encodes assumed directions and missing edges. A directed cycle is
excluded: time-index repeated decisions if feedback unfolds over time.
Prediction, correlation, feature importance and SHAP do not establish these
arrows. Identification asks whether the target follows from a distribution
plus assumptions; estimation approximates an identified functional with data.

For an intervention, the SCM equation for $A$ is replaced by a constant.
Descendants are regenerated; pretreatment variables and exogenous disturbances
are retained. This differs from observing cases already assigned $A=a$.
In general $P(Y\mid A=a)\ne P(Y\mid do(A=a))$.

## Paths, d-separation and backdoor

A path follows adjacent nodes irrespective of arrow direction. On a particular
path, an internal node is a collider if both adjacent arrows point into it.
Collider status is path-specific, not a permanent label for every role a
variable might play.

| Structure | Without conditioning on the middle node | With conditioning on it |
|---|---|---|
| Chain $A\to M\to Y$ | Active | Blocked |
| Fork $A\leftarrow C\to Y$ | Active | Blocked |
| Collider $A\to S\leftarrow U$ | Blocked unless a descendant of $S$ is conditioned on | Active |

A path is active iff every noncollider is outside the conditioning set and
every collider has itself or a descendant in that set. Two endpoints are
d-separated iff every path is blocked. Under the Markov property, this implies
conditional independence. D-connection allows dependence; it does not force it
for every parameterization. The converse from observed independence to graphical
separation needs faithfulness, excluding exact cancellations.

For single treatment $A$ and outcome $Y$, the **original backdoor criterion**
requires an adjustment set $Z$ disjoint from the endpoints such that:

1. No member of $Z$ is a descendant of $A$.
2. Every path starting with an arrow into $A$ is blocked by $Z$.

This is a sufficient graphical criterion. Its failure does not prove that all
forms of adjustment or identification are impossible. General adjustment
criteria, frontdoor and do-calculus are outside this implementation's scope.
The criterion and its adjustment implication are developed in
[Pearl, Section 3.3.1](https://ftp.cs.ucla.edu/pub/stat_ser/r354-corrected-reprint.pdf).

`DAG.satisfies_backdoor` enumerates only paths beginning with an incoming
edge and checks the descendant restriction in the original graph. Calling
ordinary d-separation on $A,Y$ in the original graph would also include causal
paths and would answer a different question. Tests include a pretreatment
collider: measuring a variable before treatment does not automatically make it
safe to control for. Node $U$ is explicitly present in the causal graph even
though it is unavailable to the estimators. Marginalizing latent nodes is not
implemented; bidirected mixed graphs are not supported.

## Identification and standardization

Consistency links the observed outcome to $Y(A)$ under well-defined treatment
versions. Conditional exchangeability is $Y(a)\perp A\mid C$; positivity requires
$0<P(A=1\mid C=c)<1$ wherever the target population has mass. Relevant
interference must be absent or modeled. Correct measurement and a credible
causal graph remain substantive requirements.

Given those assumptions, for discrete $C$:

$$
\mathbb{E}[Y(a)] = \sum_c \mathbb{E}[Y\mid A=a,C=c]P(C=c),
$$

$$
ATE = \sum_c\bigl(\mathbb{E}[Y\mid A=1,C=c]
-\mathbb{E}[Y\mid A=0,C=c]\bigr)P(C=c).
$$

Use a common target distribution $P(C)$, not a separate distribution from
each treatment arm. For continuous $C$, an integral replaces the sum; grouping
every unique continuous value is not a sensible estimator. The educational
helper accepts one discrete numeric stratum variable and targets its empirical
whole-sample frequencies. It raises an error when an observed stratum lacks an
arm. Observed counts alone cannot prove population positivity.

In an additive constant-effect linear model, adjusting for $C$ by regression
also estimates the total effect. With effect heterogeneity, a single OLS
treatment coefficient need not equal the marginal ATE; standardization and a
suitable conditional outcome model make the target weights explicit. Even in
this study, finite-sample OLS and stratified differences need not be identical.

## The synthetic SCM and population calculations

All disturbances below are mutually independent and independent of $C,U$.
$M$ and $S$ are unbounded continuous proxies, not probabilities or counts.

$$
\begin{aligned}
C&\sim\operatorname{Bernoulli}(0.5), & U&\sim N(0,1),\\
A\mid C&\sim\operatorname{Bernoulli}(0.2+0.6C),\\
M&=0.7A+\varepsilon_M, & \varepsilon_M&\sim N(0,1),\\
S&=A+U+\varepsilon_S, & \varepsilon_S&\sim N(0,0.25),\\
Y&=1.3A+M+1.5C+1.2U+\varepsilon_Y, & \varepsilon_Y&\sim N(0,1).
\end{aligned}
$$

The direct path is $1.3$, the mediated path is $0.7\times1$, and total ATE
is $2.0$. Each unit's paired oracle potential outcomes differ by $2.0$ when
$M$ is regenerated under intervention. These pairs are simulator information;
real observational data generally reveal only one potential outcome per unit.

The complete graph's edges are listed in the [README](README.md). $U$ causes
$S,Y$ but not $A$, so being latent does not alone make it a confounder.
Adding $U\to A$ would invalidate adjustment by $C$ alone.

Population slopes follow from these equations, independently of any execution:

- **Naive:** $E[C\mid A=1]-E[C\mid A=0]=0.8-0.2=0.6$.
  The treatment slope is $2+1.5(0.6)=2.9$. The bias direction comes from these
  coefficients; observational estimates need not always overstate an effect.
- **Adjust $C$:** the correct constant-effect outcome model has slope $2$.
- **Adjust $C,M$:** slope $1.3$ excludes the mediated path. It equals a
  controlled direct effect in this additive SCM, where there is no
  mediator-outcome confounding or treatment-mediator interaction. Adding a
  mediator to an arbitrary regression does not generally identify a direct
  effect or a natural direct effect.
- **Adjust $C,S$:** write $S=A+W$ with $W=U+\varepsilon_S$.
  Since $W$ is independent of $A,C$, its linear projection coefficient for
  $U$ is $\operatorname{Cov}(U,W)/\operatorname{Var}(W)=1/1.25=0.8$.
  Substitution into the outcome projection yields slope
  $2-1.2(0.8)=1.04$. This is collider bias, not a new causal target.

The implementation fits four declared regressions; it never chooses controls
by cross-validation or predictive accuracy. No propensity scores or matching
are needed for this first study; those belong to Day 50.

## FWL: algebra rather than identification

Let $X=[\mathbf{1},C]$ be the controls design, including an intercept. Regress
$A$ and $Y$ separately on $X$, obtaining residuals $r_A,r_Y$. Then:

$$
\hat\tau_{FWL}=\frac{r_A^\mathsf{T}r_Y}{r_A^\mathsf{T}r_A}.
$$

This equals the treatment coefficient from joint OLS on $[\mathbf{1},A,C]$.
The implementation uses least squares, not an explicit matrix inverse, and
rejects rank-deficient designs and nonfinite values. Controls require an
$n\times k$ matrix; use $n\times0$ for no controls. FWL equivalence also holds
with bad controls: numerical agreement cannot validate causal assumptions.
Nonlinear nuisance learning needs additional care, such as cross-fitting;
this code is not a double machine learning estimator.

## Executed synthetic study — 2026-11-12

**Hypothesis fixed by the declared SCM:** adjustment for $C$ should approach the
total effect, adjustment for $C,M$ should approach the direct path, and naive
or collider-adjusted fits should approach their biased population slopes.
The oracle intervention pairs should differ by exactly $2$ up to floating-point
rounding. FWL should agree with each corresponding joint regression, including
the incorrect causal specifications.

**Configuration:** Windows, stable Python 3.11.14, NumPy 2.4.6,
scikit-learn 1.9.1. Local isolated environment; no dependency installation or
downloads were needed. The baseline equations above were unchanged across ten
samples of 50,000 cases, seeds 49 through 58, using `numpy.random.default_rng`.
All regressions included intercepts. No estimator or control set was tuned on
the outcomes. Command from the repository root:

```powershell
.local\environments\day46\Scripts\python.exe 03-statistics-experimentation/49-causal-inference-i-dags-confounders/example.py --n-samples 50000 --seed 49 --repetitions 10
```

**Measured results:** seed 49 had $C=0$ counts of 19,985 untreated and 4,990
treated, and $C=1$ counts of 4,931 untreated and 20,094 treated. The paired
oracle ATE was 2.000000. FWL/scikit-learn agreement was checked at absolute and
relative tolerances of $10^{-10}$ for every regression across all ten samples.

| Estimator | Seed 49 | Mean, 10 samples | Sample SD | MCSE of mean | Population slope/target |
|---|---:|---:|---:|---:|---:|
| Naive | 2.879234 | 2.895813 | 0.017099 | 0.005407 | 2.90 |
| Adjust $C$ | 1.965690 | 1.993479 | 0.024439 | 0.007728 | 2.00 |
| Adjust $C,M$ | 1.280613 | 1.293501 | 0.014723 | 0.004656 | 1.30 |
| Adjust $C,S$ | 1.007378 | 1.033978 | 0.022378 | 0.007077 | 1.04 |
| Standardize over $C$ | 1.965812 | 1.993526 | 0.024383 | 0.007711 | 2.00 |

MCSE is the between-sample SD divided by $\sqrt{10}$. It describes precision of
the simulation mean, not a confidence interval for a causal effect in a real
dataset. Mean deviations from total ATE were +0.895813, -0.006521, -0.706499,
-0.966022 and -0.006474 in table order. For the mediator fit, deviation from
total ATE reflects a different target; its mean deviation from the direct
coefficient was only -0.006499.

**Interpretation candidates — author review required:** the results are
consistent with the population calculations: $C$ adjustment and standardization
approached total ATE, mediator adjustment approached the direct path, and
collider adjustment introduced a large downward departure from total ATE.
The first-seed $C$ estimate was lower than its ten-sample mean, illustrating
sampling variability. These observations validate behavior under a constructed
model, not the effect of an actual AI product or a proposed real-world DAG.

**Limitations:** only ten repetitions of one linear additive SCM; MCSE and SD
are themselves imprecise. There is no robustness study over hidden confounding,
overlap, treatment versions, nonlinear outcomes, effect heterogeneity or
selection by filtering. The code supplies no standard errors or interval
coverage assessment for the effect estimators. Access to paired potential
outcomes is a simulator privilege. Tests and this run do not approve the
interpretations on the author's behalf.

**Local software checks:** 45 topic tests passed with pytest 9.2.0, without
skips or warnings. They include a separate fixed-seed 30,000-case consistency
and coefficient check and a 1,000-case/two-seed CLI smoke run. These checks
are distinct from the 50,000-case baseline measurement above; repository
validation scope is reported in the delivery summary.

## Suggested follow-ups — not executed

Hold other equations fixed and reduce collider noise, remove the mediated
effect, or change assignment probabilities to 0.02/0.98. Compare repeated-sample
bias and variability with the corresponding analytical targets. Separately,
let latent $U$ influence assignment and update the graph before estimation.
For heterogeneous effects, compare OLS weights with target-standardized
effects. These are proposals, not findings of the executed baseline study.

## Interview practice

**Why not control for every feature?** A control's causal role depends on the
graph and estimand. A mediator can remove part of a total effect, and a
collider or its descendant can create association. More predictors can worsen
causal bias.

**What does a passing backdoor check establish?** A sufficient graphical
condition relative to a proposed DAG, not correctness of that DAG, overlap,
measurement or the fitted outcome model.

**How would you evaluate reranking in a RAG system?** Define assignment unit,
version, target population and end-to-end outcome. Randomize when feasible.
For observational comparisons, query difficulty may affect routing and answer
quality; retrieval quality can mediate reranking's effect. Feedback collection
can select cases through a collider. Diagram these mechanisms before deciding
which logs to control for.

**What if a relevant confounder is missing?** Ordinary adjustment may not
identify the effect. Seek defensible randomization or another design and
assess sensitivity; changing the regression algorithm does not restore
identification. Instruments and difference-in-differences introduce their own
assumptions and are not automatic substitutes.

**Does a larger sample fix this problem?** It reduces sampling variability
around the specification's population slope. Here it does not move $2.9$ or
$1.04$ toward the true total effect $2.0$.
