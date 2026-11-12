# Uplift: from response prediction to intervention value

## Estimand and identification

For treatment assignment $T\in\{0,1\}$, observed outcome is
$Y=TY(1)+(1-T)Y(0)$. Only one potential outcome is observed per decision.
ATE averages $Y(1)-Y(0)$ across the population; CATE averages it conditional
on pre-treatment $X$. A retention probability difference of 0.05 means five
percentage points, not a 5% relative increase. Binary potential-outcome types
(persuadables, sure things, lost causes and harmed customers) are conceptual;
the two marginal response probabilities do not identify each person's type
or the joint distribution of their potential outcomes.

Under consistency, conditional exchangeability
$(Y(0),Y(1))\perp T\mid X$, and $0<e(X)=P(T=1\mid X)<1$,

$$
\tau(x)=\mu_1(x)-\mu_0(x),\qquad \mu_t(x)=E[Y\mid X=x,T=t].
$$

No interference and well-defined treatment versions matter too. Randomization
establishes assignment exchangeability by design; it does not fix spillovers,
noncompliance, missing outcomes or post-treatment leakage. The generator has
full compliance and complete outcomes, so assignment and receipt coincide.
With real noncompliance, preserve the intention-to-treat estimand unless a
separate identification strategy justifies effects of receipt.

## Learners and error propagation

| Approach | Construction | Main trade-off |
|---|---|---|
| S-Learner | Fit one $\mu(X,T)$, predict twice | Shares information; can ignore treatment or shrink its effect |
| T-Learner | Fit $\mu_0,\mu_1$ on separate arms | Simple; each fit uses fewer observations and their errors compound |
| X-Learner | Impute effects from outcome models, fit effect models, blend | Can exploit imbalance and effect structure; more nuisance stages |
| R-Learner | Residualize treatment and outcomes, fit an effect loss | Requires careful nuisance estimation and overlap |
| DR-Learner | Regress an augmented IPW pseudo-outcome | Robustness needs identification and nuisance conditions; cross-fitting matters |

Here $\hat\tau-\tau=\epsilon_1-\epsilon_0$ if
$\hat\mu_t=\mu_t+\epsilon_t$. Good ROC-AUC for outcomes does not guarantee
small CATE error or useful ranking. Random Forest probabilities are averages
of leaf class frequencies, with no automatic probability-calibration guarantee.
No hyperparameter or learner selection occurs in this study.

The educational binned estimator estimates
$\bar Y_{T=1,b}-\bar Y_{T=0,b}$ inside training quantiles of feature 0.
Under this constant-propensity RCT it targets a bin-average effect. It does
not generally identify that same effect in observational bins with remaining
within-bin confounding. More bins trade resolution for arm-specific sample
size; missing arms raise errors rather than silently substituting zero.

## IPW scoring and gain conventions

For known assignment probability $e_i$, define

$$
Z_i=\frac{T_iY_i}{e_i}-\frac{(1-T_i)Y_i}{1-e_i}.
$$

Conditional expectation cancels assignment probabilities:
$E[T Y/e(X)\mid X]=\mu_1(X)$, and similarly for control. Thus
$E[Z\mid X]=\tau(X)$. $Z$ is noisy and can be outside $[-1,1]$ even for a
binary outcome; it is not an individual treatment effect. Known propensity
means the design probability, not the realized fraction treated. The
implementation accepts a scalar or aligned propensity vector, rejects 0/1
and nonfinite values, and never clips extreme propensities silently.

For a ranking fixed without test outcomes and its selected top-$q$ set $S_q$,

$$
\widehat G(q)=\frac1n\sum_{i\in S_q}Z_i.
$$

This is a Horvitz–Thompson gain per member of the full eligible holdout, not
the average effect among selected people. For $q>0$, dividing by the actual
selected fraction changes the denominator to selected people. Multiply gain
by 1,000 for incremental outcomes per 1,000 eligible people. At $q=1$, it
equals the IPW ATE, which need not equal an unweighted difference in means
in a finite, unequally allocated sample. Gains can decrease when additional
people have negative effects or because outcomes are noisy.

Complete score-tie blocks determine curve vertices, with linear interpolation
inside blocks. This avoids exploiting arbitrary input order for AUUC. The
reference random line is $q\widehat G(1)$; its integral is
$\widehat G(1)/2$. The reported Qini area is the unnormalized difference
between AUUC and that integral. It is not a universally standardized metric,
and can be negative. A constant score has zero centered area by construction,
even when the observed ATE is nonzero.

## From ranking to action

For an additive intervention with value $v$ per incremental conversion,
equal per-contact cost $c$ and deterministic policy $\pi$,

$$
V_{inc}(\pi)=E[\pi(X)(v\tau(X)-c)],\qquad
\widehat V_{inc}(\pi)=\frac1n\sum_i\pi(X_i)(vZ_i-c).
$$

Treating nobody has zero incremental value by definition; this is not total
revenue. Without a capacity constraint the plug-in rule is
$v\hat\tau(x)>c$. With equal values/costs and at most $k$ contacts, select up
to the highest $k$ positive-net scores. The example caps contacts at
$\lfloor0.3n\rfloor$. Input-order ties are a deterministic policy convention
independent of treatment/outcome, distinct from tie-averaged ranking metrics.
Variable costs, heterogeneous conversion values, spillovers or fairness
constraints require a different allocation problem.

The random comparator is the expectation over uniformly selecting 30% of
people, not a separately simulated policy deployment. Every learned policy
uses only features at test time. Holdout outcomes evaluate policies; synthetic
truth evaluates oracle diagnostics only. The top-30% comparisons are fixed
capacity policies, while the cost-aware rule can leave capacity unused.

An augmented score for an observational extension is

$$
\phi_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)
+\frac{T_i(Y_i-\hat\mu_1(X_i))}{\hat e_i}
-\frac{(1-T_i)(Y_i-\hat\mu_0(X_i))}{1-\hat e_i}.
$$

Suitable nuisance consistency conditions give double robustness under causal
identification, not immunity to hidden confounding. Cross-fitting helps control
overfitting bias. This topic documents that extension but does not implement it.
Production targeting also needs a valid randomized monitoring population,
assignment-probability logs, mature outcomes, policy versions and guardrails.

## Executed synthetic study — 2026-11-12

**Hypotheses.** In heterogeneous data, effect ranking may concentrate benefit
relative to random targeting and control-response ranking. In the homogeneous
control, there is no true CATE ranking advantage. These are study questions,
not promised learner orderings.

**Execution and configuration.** Codex ran both scenarios locally on Windows
with Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1.
Each scenario has ten complete generate/split/refit/evaluate replications,
using seeds 52–61, 12,000 people per run, 7,200 training and 4,800 holdout,
three independent standard-normal features and Bernoulli(0.5) assignment.
No tuning or test-based selection was performed. Both forests use 160 trees,
maximum depth 6, minimum leaf size 60, seed equal to the replication seed,
and one worker. The binned estimator uses up to six training quantile bins
on feature 0. Policies use value 100, contact cost 5 and 30% capacity.

For heterogeneous data, the response functions are

$$
p_0(x)=\operatorname{expit}(-1+0.8x_0-0.5x_1),\qquad
p_1(x)=\operatorname{clip}(p_0(x)+0.16\tanh x_0-0.10\tanh x_1,0.01,0.99).
$$

The true CATE is the **post-clipping** difference $p_1-p_0$. Clipping changes
the mean and the heterogeneity; the mean need not be zero. The homogeneous
control uses $p_0=\min(\operatorname{expit}(-1+0.8x_0-0.5x_1),0.94)$ and
$p_1=p_0+0.05$, avoiding clipping-induced heterogeneity. Outcomes are sampled
from the probability of the assigned arm; individual potential-outcome pairs
are not generated. Both scenarios reuse seed labels, so they are not two
independent samples for a pooled comparison.

Commands from the repository root, after activating the environment:

```powershell
python 03-statistics-experimentation/52-uplift-modeling/from_scratch.py
python 03-statistics-experimentation/52-uplift-modeling/example.py --output 03-statistics-experimentation/52-uplift-modeling/outputs/2026-11-12-heterogeneous-run01.json
python 03-statistics-experimentation/52-uplift-modeling/example.py --scenario homogeneous --output 03-statistics-experimentation/52-uplift-modeling/outputs/2026-11-12-homogeneous-run01.json
```

The actual interpreter was
`.local/environments/s08-reference-2026-10-08/Scripts/python.exe`; no packages
were installed or changed. Run JSON files stay ignored, with configuration,
versions, UTC timestamp and per-replication metrics. Use new output names on
reruns; the CLI refuses to overwrite these records.

**Results.** Entries below are means ± Monte Carlo standard error of the mean
over ten replications. These are probability units for effects/RMSE,
population-normalized area units for Qini, and illustrative currency per
eligible person for net value.

| Quantity | Heterogeneous | Homogeneous |
|---|---:|---:|
| Known holdout mean CATE | 0.014363 ± 0.000650 | 0.050000 ± 0.000000 |
| IPW holdout ATE | 0.019792 ± 0.004270 | 0.056458 ± 0.004761 |
| T-Learner CATE RMSE (synthetic only) | 0.057505 ± 0.001556 | 0.050541 ± 0.001361 |
| T-Learner Qini area | 0.022653 ± 0.001076 | 0.001273 ± 0.001444 |
| Control-response Qini area | 0.025509 ± 0.001588 | 0.000690 ± 0.001862 |
| Binned-estimator Qini area | 0.022351 ± 0.001452 | 0.001109 ± 0.001700 |
| IPW net: uplift top 30% | 2.291667 ± 0.260875 | 0.350000 ± 0.269616 |
| IPW net: response top 30% | 2.204167 ± 0.316011 | -0.154167 ± 0.421182 |
| IPW net: positive-net uplift, capacity cap | 2.236875 ± 0.246819 | 0.350000 ± 0.269616 |
| IPW net: treat all | -3.020833 ± 0.427013 | 0.645833 ± 0.476057 |
| IPW net: expected random 30% | -0.906250 ± 0.128104 | 0.193750 ± 0.142817 |
| IPW net: treat none | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| Paired IPW net difference: uplift minus response top 30% | 0.087500 ± 0.295833 | 0.504167 ± 0.402991 |

In heterogeneous runs, treated counts ranged from 3,545–3,651 of 7,200
training people and 2,350–2,463 of 4,800 test people. The design probability
remains 0.5; these counts are descriptive, not an SRM test. The cost-aware
policy treated 23.4583%–30% of the holdout. Its oracle net value averaged
2.173065, compared with 2.179685 for uplift top 30% and 2.478333 for response
top 30%. Oracle values are diagnostics available only in this simulation.

**Interpretation candidates — author review required.**

- In this particular heterogeneous generator, response and effect signals both
  depend on the first two features in similar directions. Control-response
  ranking is therefore a strong comparator: its mean Qini exceeded the
  T-Learner's. The small mean IPW net difference between top-30% policies is
  smaller than its Monte Carlo error. These runs do not establish T-Learner
  superiority, even though targeting concentrated benefit relative to the
  random comparator.
- The homogeneous generator has no true CATE ranking advantage, and
  $100\times0.05-5=0$: all policies have zero true incremental net value.
  The nonzero measured gains/net values illustrate sampling variation and
  model noise, not profitable subgroups. Ten repetitions do not establish
  general calibration or a false-positive rate.
- Fixed thresholds and economics can leave capacity unused. Their measured
  ordering does not establish a best treatment fraction or justify a rollout.

**Limitations.** These are small Monte Carlo studies of two chosen generators,
not benchmarks. MCSE measures simulation error of the reported mean, not a
confidence interval for one deployed policy, each CATE, or a real population.
The study has complete outcomes, no hidden confounding, no interference,
fixed values/costs and no compliance or eligibility complications. Response
signals align with uplift here; other generators can reverse the ordering.
No model calibration, subgroup coverage, fairness, production guardrail or
prospective policy experiment was evaluated. Conclusions remain pending
manual author review.

## Further experiments — suggested, not executed

- Repeat low-sample and 10%-treatment designs; update IPW propensities to match
  assignment and inspect variance plus missing-arm failures.
- Vary contact costs using a separate validation set for policy choice, then
  evaluate one frozen choice on a final test set.
- Add observational assignment, specify measured confounders and examine overlap
  before estimating effects. Compare IPW with cross-fitted augmented scores.
- Test calibration of subgroup-average effects on new randomized samples and
  evaluate stability, interference, delayed outcomes and costs in a real design.

These suggestions are separate from the two scenarios in the dated record.
