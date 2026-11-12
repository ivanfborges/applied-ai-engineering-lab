# Error control across hypotheses and time

## Define the decision before choosing the correction

A hypothesis family is the collection of inferential claims covered by a
stated error guarantee. A primary endpoint and a prespecified secondary family
can support different decisions; splitting a family after seeing favorable
results undermines the intended guarantee. Document variants, endpoints,
segments, test directions, missing-data rules, randomization/analysis units,
sample size and stopping policy before observing outcomes.

For a RAG comparison, queries evaluated under both systems may require paired
inference; repeated queries from one user may require cluster-aware analysis.
Correlated metrics are not independent just because they have different names.
The simulations here deliberately use independent hypotheses to make their
guarantees and analytical comparison transparent. They do not implement RAG
evaluation or infer validity for correlated business metrics.

## What each error rate measures

Let $m$ hypotheses contain $m_0$ true nulls. Let $R$ be the number rejected,
$V$ the rejected true nulls and $S=R-V$ the rejected false nulls. Then

$$
\mathrm{FWER}=P(V\geq1),\qquad
\mathrm{FDP}=\frac{V}{\max(R,1)},\qquad
\mathrm{FDR}=E[\mathrm{FDP}].
$$

FDP is zero when there are no rejections. FDR is not $E[V]/E[R]$, the fraction
of false discoveries pooled across repetitions, or a posterior probability
for a specific discovery. A realized FDP can exceed the target even when the
procedure controls FDR. Under the global null $V=R$, so FDP is the indicator
of any rejection and FDR equals FWER. That identity need not hold with signals.

With independent true nulls and exact continuous null p-values, uncorrected
testing at level $\alpha$ gives

$$
P(V\geq1)=1-(1-\alpha)^m.
$$

For $m=20$ and $\alpha=0.05$, this is approximately $0.6415$. Dependence changes
this exact calculation. A valid null p-value must satisfy
$P(p_i\leq t)\leq t$ for all $t\in[0,1]$; conservatively distributed p-values
need not be uniform. A p-value is not $P(H_0\mid\text{data})$.

## Bonferroni and Holm: protect the family

Bonferroni rejects $H_i$ if $p_i\leq\alpha/m$, or equivalently if
$p_i^{\mathrm{adj}}=\min(1,mp_i)\leq\alpha$. For true null index set $I_0$,

$$
P(V\geq1)
\leq\sum_{i\in I_0}P(p_i\leq\alpha/m)
\leq m_0\alpha/m\leq\alpha.
$$

This union-bound argument gives strong FWER control: it holds for every
configuration of true and false nulls and requires no independence.

Holm sorts $p_{(1)}\leq\cdots\leq p_{(m)}$, tests in ascending order against
$\alpha/(m-i+1)$ and stops at the first failure. Its ordered adjusted p-values are

$$
p_{(i)}^{\mathrm{Holm}}=
\min\left(1,\max_{j\leq i}(m-j+1)p_{(j)}\right).
$$

The cumulative maximum enforces step-down consistency. Holm also provides
strong FWER control under arbitrary dependence and rejects at least everything
Bonferroni rejects at the same level. Both can lose power when a large family
contains weak signals; an explicit primary decision can avoid unnecessary tests.

## BH: protect the expected discovery proportion

For target $q$, BH finds the largest $k$ satisfying $p_{(k)}\leq kq/m$ and
rejects ranks $1,\ldots,k$. If no rank passes, it rejects none. Ordered
adjusted p-values are

$$
p_{(i)}^{\mathrm{BH}}=
\min\left(1,\min_{j\geq i}\frac{m}{j}p_{(j)}\right).
$$

The suffix minimum matters. For $p=(0.03,0.04)$ and $q=0.05$, rank 1 fails
its own $0.025$ boundary, but rank 2 passes $0.05$: BH rejects both. Comparing
each rank independently, or stopping at the first failure as in Holm, is wrong.
The implementation restores original order after adjustment and uses inclusive
boundaries. Adjusted p-values depend on the entire supplied family; filtering
to small raw p-values before correcting changes that family.

BH guarantees $\mathrm{FDR}\leq(m_0/m)q\leq q$ under independence and appropriate
PRDS conditions on the true nulls. Positive pairwise correlation alone does
not establish PRDS in every model or test direction. BY replaces $q$ with
$q/H_m$, where $H_m=\sum_{j=1}^m1/j$, to obtain FDR control under arbitrary
dependence with valid null p-values. BY is discussed, not implemented or
empirically studied here. See the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)
for BH/BY assumptions and original references.

## Repeated looks change the decision rule

Under the simulated null, $X_j$ are independent $N(0,1)$ observations, with
known variance. At prespecified cumulative sample size $n$,

$$
Z_n=\frac{\sum_{j=1}^nX_j}{\sqrt n},\qquad
p_n=2\Phi(-|Z_n|).
$$

Each marginal $p_n$ is exactly calibrated at its fixed look. Consecutive looks
share observations: for $n_a\leq n_b$,
$\operatorname{Corr}(Z_{n_a},Z_{n_b})=\sqrt{n_a/n_b}$.
Therefore, $1-(1-\alpha)^L$ is not the appropriate exact false-positive formula
for $L$ nested looks.

The naive policy rejects when $\min_{n\in\mathcal L}p_n\leq\alpha$, which is a
different event from rejecting only at the final $n$. Computing all looks on
a simulated full path reproduces the rejection event of stopping at the first
crossing; this example does not estimate stopping times or sample savings.
The generator draws independent block sums with variances equal to the
additional sample counts and cumulatively sums them. This is distributionally
equivalent to summing individual Gaussian observations, with less storage.

Bonferroni over a finite prespecified schedule uses $\alpha/L$ at each look.
The union bound controls Type I error despite temporal dependence. This is
conservative and does not allow adding arbitrary unbudgeted looks. Correcting
twenty endpoints at each of ten looks separately does not automatically control
the combined decision family. One simple conservative design allocates
$\alpha/(mL)$ to each endpoint/look pair, provided the individual p-values are
valid at their planned times. More efficient group-sequential boundaries or
anytime-valid methods require their own assumptions and implementation.

Monitoring operations or data quality is not itself a significance-based
stopping decision. Safety stops may be operationally necessary, but subsequent
efficacy claims must respect the actual stopping/selection process. Multiplicity
correction does not supply a non-inferiority analysis or certify safe deployment.
The [KDD peeking paper](https://www.kdd.org/kdd2017/papers/view/peeking-at-ab-tests-why-it-matters-and-what-to-do-about-it)
motivates procedures valid under continuous monitoring.

## Measurement contract

The executable study uses 10,000 independent repetitions per scenario, seed 48,
and separate PCG64 streams spawned by `SeedSequence(48).spawn(3)`. Families
have 20 independent normal statistics: all means zero in the first scenario;
the first four means are 3 and the others zero in the second. The shift is a
standardized Z-statistic mean, not a user-level uplift or a sample-size plan.
All p-values are two-sided; the nominal FWER/FDR level is 0.05.

The temporal scenario is a separate global-null experiment with cumulative
looks at 100, 200, ..., 1,000 observations. It uses known variance 1, no
seasonality, clustering, delayed outcomes or changes in allocation.

For independent repetition-level measurements $Y_b$, reports contain

$$
\widehat\mu=\frac1B\sum_{b=1}^B Y_b,\qquad
\mathrm{MCSE}=\frac{s_Y}{\sqrt B},
$$

where $s_Y$ uses $B-1$ degrees of freedom. FWER uses the any-false-rejection
indicator; FDR uses the FDP including zero-discovery runs; average power uses
$S/4$ in the mixed family. Global-null power is undefined and reported as
`null`. These standard errors describe simulation uncertainty, not intervals
for a real treatment effect. A zero estimated standard error at an empirical
boundary is not evidence that the event is impossible. Policy differences use
paired per-trajectory indicators, so their MCSE accounts for shared paths.

## Executed synthetic study — 2026-11-12

Executed by Codex on Windows x64 with stable Python 3.11.14, NumPy 2.4.6 and
SciPy 1.17.1. Independent numerical test references used statsmodels 0.15.0;
the focused suite used pytest 9.2.0. Dependency ranges are not an environment
lock. This is the first baseline record; subsequent runs must retain their own
configuration/date rather than silently replacing these measurements.

Actual commands from the repository root:

```powershell
.local/environments/day46/Scripts/python.exe 03-statistics-experimentation/48-multiple-testing/from_scratch.py
.local/environments/day46/Scripts/python.exe 03-statistics-experimentation/48-multiple-testing/example.py --trials 10000 --seed 48 --output 03-statistics-experimentation/48-multiple-testing/outputs/study-2026-11-12-seed48.json
.local/environments/day46/Scripts/python.exe -m pytest -q 03-statistics-experimentation/48-multiple-testing/tests/test_multiple_testing.py
```

The local environment path is a run record, not a portable setup requirement;
use the [README installation commands](README.md#files-and-execution) elsewhere.
The JSON execution timestamp was `2026-11-12T09:18:35.023843+00:00`. The report
remains ignored; the deliberately curated measurements below provide the
public evidence without linking an unpublished asset.

**Hypothesis 1:** Uncorrected testing increases global-null family-level error;
Bonferroni/Holm protect FWER and BH protects FDR under this independent generator.
Configuration: the measurement contract above, 20 true nulls, 10,000 repetitions.

| Method | Estimated FWER = estimated FDR | MCSE |
|---|---:|---:|
| Uncorrected | 0.6388 | 0.004804 |
| Bonferroni | 0.0498 | 0.002175 |
| Holm | 0.0498 | 0.002175 |
| BH | 0.0507 | 0.002194 |

**Interpretation candidate — author review required:** The raw rate is within
one MCSE of the analytical 0.641514 value. The corrected rates are compatible
with their theoretical error control in this generator. BH's empirical 0.0507
is above 0.05 by less than one MCSE; a finite estimate need not fall below the
nominal bound. Holm and Bonferroni have the same any-rejection event under the
global null, despite potentially different total discovery counts.
**Limitation:** This configuration cannot demonstrate the distinction between
FDR and FWER, which coincide under the global null; simulation cannot prove
strong control under all null configurations or dependence structures.

**Hypothesis 2:** With signals present, FDR and FWER differ and stricter family
protection trades off against true discoveries. Configuration: 16 true nulls,
four non-null $N(3,1)$ statistics, independent within each family, 10,000 repetitions.
Every cell reports **estimate (MCSE)**; power is the average fraction of the
four non-nulls rejected.

| Method | FWER | FDR | Average power |
|---|---:|---:|---:|
| Uncorrected | 0.5555 (0.004969) | 0.160664 (0.001619) | 0.852350 (0.001782) |
| Bonferroni | 0.0421 (0.002008) | 0.017075 (0.000913) | 0.492050 (0.002502) |
| Holm | 0.0466 (0.002108) | 0.018203 (0.000927) | 0.502325 (0.002555) |
| BH | 0.1405 (0.003475) | 0.041671 (0.001119) | 0.622650 (0.002816) |

**Interpretation candidate — author review required:** BH permits more true
discoveries than the FWER procedures in this run while controlling a different
criterion: its FWER is 0.1405, substantially above its estimated FDR. Its
estimated FDR is about 1.5 MCSE above the independent-model bound
$(16/20)0.05=0.04$, compatible with finite simulation variation. Holm's larger
rejection set than Bonferroni is consistent with its deterministic dominance;
this table alone is not a paired significance test of differences in power.
**Limitation:** A single effect size and sparsity pattern cannot establish a
universal ranking of methods or prescribe an error criterion for product decisions.

**Hypothesis 3:** Stopping at the first unadjusted significance crossing
increases Type I error relative to testing only at the final horizon.
Configuration: one global-null normal process, ten nested looks at
$100,200,\ldots,1000$, known variance 1, 10,000 shared trajectories.

| Policy | Null rejection rate | MCSE |
|---|---:|---:|
| Final horizon only, threshold 0.05 | 0.0505 | 0.002190 |
| Any look, threshold 0.05 | 0.2000 | 0.004000 |
| Any look, threshold 0.005 | 0.0262 | 0.001597 |

Paired naive-minus-fixed rate difference: **0.1495**, MCSE **0.003566**.
Paired naive-minus-Bonferroni difference: **0.1738**, MCSE **0.003790**.

**Interpretation candidate — author review required:** The extra crossing
opportunities increase rejection frequency under this null. Allocating the
budget across all ten looks is conservative with these correlated paths.
**Limitation:** These are not independent tests, a group-sequential efficiency
benchmark, an anytime-valid method or an estimate of early-stopping savings.
No alternative temporal process was simulated, so sequential power is unknown.

## Proposed extensions — not executed

- Sweep the family size and check uncorrected FWER against its analytical
  expression under independence.
- Sweep effect size and sparsity with repeated families to study power loss.
- Generate explicit correlated test statistics and assess suitable BH/BY
  assumptions before comparing empirical error rates.
- Vary planned look counts at a fixed final sample size; compare a dedicated
  group-sequential design or an anytime-valid procedure.
- Select a prompt using development queries, then evaluate on independent
  holdout queries to separate winner selection from final inference.

No correlation sweep, sequential-boundary optimization, holdout selection
experiment or visualization is implied by the executed baseline study.

## Local verification — 2026-11-12

The final focused numerical/CLI suite passed **67 tests**, without skips or
warnings. It checks known adjustments, the BH last-passing-rank case,
statsmodels agreement for Bonferroni/Holm/BH, SciPy BH agreement, original-order
restoration, ties, inclusive boundaries, invalid inputs, the mean-FDP denominator,
nested sums, monitoring policies and a small deterministic JSON/CLI round-trip.
Small test fixtures are software checks, not additional scientific experiments.

Repository syntax parsing passed for **323 public Python files**; local-link
validation passed for **1,048 links in 220 Markdown files**. Commands were
`python scripts/validate_repo.py syntax` and
`python scripts/validate_repo.py links`, using the stable local environment
recorded above. `python -m pip check` found no broken requirements.

The saved report was also checked against global-null analytical FWER targets
(uncorrected, Bonferroni, Holm and BH), using a tolerance of four MCSE. All
deviations were less than one MCSE. The global-null FDR/FWER identities, the
final-horizon calibration and the finite-look Bonferroni upper bound also
passed those report checks. These are finite-run consistency checks, not
proofs of control or scientific approval.

Whitespace checking of this change is separate from pre-existing worktree
changes. Full-suite CI, browser/GitHub math rendering and personal author
review were not performed. The synthetic interpretation candidates above
remain **author review required**.
