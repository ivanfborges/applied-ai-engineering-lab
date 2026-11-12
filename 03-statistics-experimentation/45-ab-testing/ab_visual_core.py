"""Synthetic statistical mechanisms for the Day 45 visual lab."""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.stats import beta, binomtest, norm, t, ttest_ind
from from_scratch import sample_size_per_arm, validate_count, validate_probability


@dataclass(frozen=True)
class LabConfig:
    seed: int = 45
    repetitions: int = 3000
    n_per_arm: int = 15000
    baseline: float = .10
    treatment: float = .11
    alpha: float = .05

    def __post_init__(self):
        validate_count(self.seed, 'seed')
        validate_count(self.repetitions, 'repetitions', 100)
        validate_count(self.n_per_arm, 'n_per_arm', 1000)
        for name in ('baseline', 'treatment', 'alpha'):
            validate_probability(getattr(self, name), name)
        if not (.02 <= self.baseline <= .95 and .02 <= self.treatment <= .98):
            raise ValueError('Normal-approximation lab requires baseline in [0.02, 0.95] and treatment in [0.02, 0.98].')
        if self.repetitions > 20000 or self.n_per_arm > 100000:
            raise ValueError('Desktop limits: repetitions <= 20000, n_per_arm <= 100000.')

    def rng(self, module, scenario=0):
        return np.random.default_rng(np.random.SeedSequence([self.seed, module, scenario]))


def binary_inference(x_a, n_a, x_b, n_b, alpha=.05):
    """Pooled null Z and unpooled Wald interval; sparse tables fail explicitly."""
    alpha = validate_probability(alpha, 'alpha')
    arrays = np.broadcast_arrays(x_a, n_a, x_b, n_b)
    if any(a.dtype.kind not in 'iu' or (a < 0).any() for a in arrays):
        raise ValueError('Counts and arm sizes must be nonnegative integers.')
    xa, na, xb, nb = [a.astype(float) for a in arrays]
    if (na <= 0).any() or (nb <= 0).any() or (xa > na).any() or (xb > nb).any():
        raise ValueError('Need positive arm sizes and successes <= sizes.')
    if any((a < 5).any() for a in (xa, na-xa, xb, nb-xb)):
        raise ValueError('Sparse table: require >=5 successes and failures per arm; increase n or use score/exact inference.')
    pa, pb = xa/na, xb/nb
    pooled = (xa+xb)/(na+nb)
    se0 = np.sqrt(pooled*(1-pooled)*(1/na+1/nb))
    se = np.sqrt(pa*(1-pa)/na+pb*(1-pb)/nb)
    difference = pb-pa
    z = difference/se0
    critical = norm.isf(alpha/2)
    return dict(difference=difference, relative_lift=difference/pa, se_null=se0, se=se, z=z,
                p_value=2*norm.sf(abs(z)), low=difference-critical*se,
                high=difference+critical*se, rate_a=pa, rate_b=pb)


def trials(config, n=None, p_b=None, module=3, scenario=0, repetitions=None):
    n = config.n_per_arm if n is None else validate_count(n, 'n', 1000)
    p_b = config.treatment if p_b is None else validate_probability(p_b, 'p_b')
    reps = config.repetitions if repetitions is None else validate_count(repetitions, 'repetitions', 1)
    rng = config.rng(module, scenario)
    xa = rng.binomial(n, config.baseline, reps)
    xb = rng.binomial(n, p_b, reps)
    return dict(x_a=xa, x_b=xb, n=n, **binary_inference(xa,n,xb,n,config.alpha))


def rate_summary(events):
    events = np.asarray(events, bool)
    if events.ndim != 1 or not events.size:
        raise ValueError('Need a nonempty 1D indicator.')
    k, n = int(events.sum()), events.size
    ci = binomtest(k,n).proportion_ci(method='exact')
    return dict(count=k, repetitions=n, rate=k/n,
                mc_se=float(np.sqrt((k/n)*(1-k/n)/n)), mc_ci95=[float(ci.low),float(ci.high)])


def analytical_power(n, difference, baseline=.10, alpha=.05):
    """Two-tail normal planning approximation, not exact binomial test power."""
    baseline = validate_probability(baseline,'baseline')
    alpha = validate_probability(alpha,'alpha')
    n,difference = np.broadcast_arrays(np.asarray(n,float),np.asarray(difference,float))
    p1 = baseline+difference
    if (not np.isfinite(n).all() or not np.isfinite(difference).all() or (n<=0).any()
            or (p1<=0).any() or (p1>=1).any()):
        raise ValueError('Need finite positive n and 0 < baseline + difference < 1.')
    pooled = (baseline+p1)/2
    threshold = norm.isf(alpha/2)*np.sqrt(2*pooled*(1-pooled)/n)
    se_alt = np.sqrt((baseline*(1-baseline)+p1*(1-p1))/n)
    return norm.cdf((-threshold-difference)/se_alt)+norm.sf((threshold-difference)/se_alt)


def population(config,n=320,effect=.08,module=1):
    """Shared-uniform potential outcomes explicitly impose monotonicity."""
    rng = config.rng(module)
    latent = rng.normal(size=n)
    activity = np.exp(.55*latent+rng.normal(0,.25,n))
    history = expit(-2+.7*latent+rng.normal(0,.4,n))
    engagement = expit(-1+.5*latent)
    p0 = np.clip(config.baseline*np.exp(.35*latent),.005,.8)
    p1 = np.minimum(p0+effect,.98)
    u = rng.random(n)
    y0,y1 = (u<p0).astype(int),(u<p1).astype(int)
    assignment = rng.integers(0,2,n)
    observed = np.where(assignment==1,y1,y0)
    features = np.column_stack((engagement,activity,history,latent))
    standardized = (features-features.mean(0))/features.std(0)
    return dict(assignment=assignment,observed=observed,y0=y0,y1=y1,p0=p0,p1=p1,
                features=features,standardized=standardized,
                mean_differences=standardized[assignment==1].mean(0)-standardized[assignment==0].mean(0),
                finite_ate=float((y1-y0).mean()), expected_ate=float((p1-p0).mean()),
                estimate=float(observed[assignment==1].mean()-observed[assignment==0].mean()))


def peeking(config):
    rng = config.rng(9)
    first = max(500,int(np.ceil(25/min(config.baseline,1-config.baseline))))
    if first > config.n_per_arm:
        raise ValueError('Maximum horizon must exceed first look.')
    looks = np.unique(np.linspace(first,config.n_per_arm,30,dtype=int))
    increments = np.diff(np.r_[0,looks])
    shape = (config.repetitions,len(looks))
    xa = rng.binomial(increments,config.baseline,shape).cumsum(1)
    xb = rng.binomial(increments,config.baseline,shape).cumsum(1)
    p = binary_inference(xa,looks,xb,looks,config.alpha)['p_value']
    crossed = p<config.alpha
    any_cross = crossed.any(1)
    indices = np.where(any_cross,crossed.argmax(1),len(looks)-1)
    return dict(looks=looks,p_values=p,any_cross=any_cross,stopped_n=looks[indices],
                fixed=rate_summary(crossed[:,-1]),naive=rate_summary(any_cross))


def allocation_cases(config,n=3000):
    rng = config.rng(10)
    healthy,biased = int(rng.binomial(n,.5)),int(rng.binomial(n,.6))
    return [dict(label=label,n=n,n_b=nb,actual_q=q,intended_q=intended,
                 p_value=float(binomtest(nb,n,intended).pvalue))
            for label,nb,q,intended in (
                ('Healthy: intended 50/50',healthy,.5,.5),
                ('Bug: actual 40/60, intended 50/50',biased,.6,.5),
                ('Configured 40/60: same counts',biased,.6,.6))]


def user_mean_inference(a,b,alpha=.05):
    """Welch inference over independent user means, not binary requests."""
    a,b = np.asarray(a,float),np.asarray(b,float)
    if a.ndim!=1 or b.ndim!=1 or min(a.size,b.size)<2:
        raise ValueError('Need two 1D independent-user samples with >=2 users.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('User outcomes must be finite.')
    va,vb = a.var(ddof=1)/a.size,b.var(ddof=1)/b.size
    if va+vb<=0:
        raise ValueError('Degenerate variance.')
    se = np.sqrt(va+vb)
    df = (va+vb)**2/(va**2/(a.size-1)+vb**2/(b.size-1))
    d = b.mean()-a.mean()
    critical = t.isf(alpha/2,df)
    return dict(difference=float(d),se=float(se),df=float(df),low=float(d-critical*se),
                high=float(d+critical*se),p_value=float(ttest_ind(b,a,equal_var=False).pvalue))


def clustered_trials(config,rho=.20,events_per_user=20,users_per_arm=500,repetitions=None):
    if not 0<=rho<1:
        raise ValueError('ICC must be in [0,1).')
    validate_count(events_per_user,'events_per_user',2)
    validate_count(users_per_arm,'users_per_arm',20)
    reps = config.repetitions if repetitions is None else validate_count(repetitions,'repetitions',1)
    rng = config.rng(11,int(rho*1000))
    naive_reject, user_reject = [], []
    first_counts, naive_first, user_first = None, None, None
    # Bound temporary arrays independently of the requested repetition count.
    for start in range(0, reps, 128):
        shape = (min(128, reps-start), 2, users_per_arm)
        if rho:
            k = 1/rho-1
            risk = rng.beta(config.baseline*k,(1-config.baseline)*k,shape)
        else:
            risk = np.full(shape,config.baseline)
        counts = rng.binomial(events_per_user,risk)
        means = counts/events_per_user
        totals = counts.sum(2)
        naive = binary_inference(totals[:,0],users_per_arm*events_per_user,
                                 totals[:,1],users_per_arm*events_per_user,config.alpha)
        variances = means.var(2,ddof=1)/users_per_arm
        se = np.sqrt(variances.sum(1))
        df = variances.sum(1)**2/((variances**2).sum(1)/(users_per_arm-1))
        differences = means[:,1].mean(1)-means[:,0].mean(1)
        p = 2*t.sf(abs(differences/se),df)
        naive_reject.extend(naive['p_value']<config.alpha)
        user_reject.extend(p<config.alpha)
        if start == 0:
            first_counts = counts[0].copy()
            naive_first = {key:float(value[0]) for key,value in naive.items()}
            user_first = user_mean_inference(means[0,0],means[0,1],config.alpha)
    return dict(rho=rho,events_per_user=events_per_user,users_per_arm=users_per_arm,
                counts=first_counts,naive_first=naive_first,user_first=user_first,
                naive=rate_summary(naive_reject),user=rate_summary(user_reject),batch_size=128,
                design_effect=1+(events_per_user-1)*rho)


def product_experiment(config):
    rng = config.rng(12)
    n = config.n_per_arm
    xa,xb = int(rng.binomial(n,config.baseline)),int(rng.binomial(n,config.treatment))
    primary = binary_inference(xa,n,xb,n,config.alpha)
    tokens_a,tokens_b = rng.lognormal(np.log(1200),.35,n),rng.lognormal(np.log(1900),.35,n)
    latency_a,latency_b = rng.lognormal(np.log(.9),.4,n),rng.lognormal(np.log(1.4),.45,n)
    unsafe_a,unsafe_b = int(rng.binomial(n,.0015)),int(rng.binomial(n,.003))
    safety_upper = float(beta.ppf(.95,unsafe_b+1,n-unsafe_b)) if unsafe_b<n else 1.
    p95_a,p95_b = float(np.quantile(latency_a,.95)),float(np.quantile(latency_b,.95))
    policy = dict(minimum_effect=.005,max_latency_ratio=1.25,max_cost_increment=.005,max_safety_upper=.003)
    metrics = dict(p95_a=p95_a,p95_b=p95_b,cost_a=float(tokens_a.mean()*.00001),
                   cost_b=float(tokens_b.mean()*.00001+.002),safety_a=unsafe_a/n,safety_b=unsafe_b/n,
                   safety_upper_b=safety_upper,tokens_a=float(tokens_a.mean()),tokens_b=float(tokens_b.mean()),
                   quality_a=float(rng.beta(12,8,n).mean()),quality_b=float(rng.beta(16,6,n).mean()),docs_a=5,docs_b=8)
    gates = dict(primary_evidence=bool(primary['p_value']<config.alpha and primary['low']>0),
                 practical_value=bool(primary['low']>policy['minimum_effect']),
                 latency=bool(p95_b/p95_a<=policy['max_latency_ratio']),
                 cost=bool(metrics['cost_b']-metrics['cost_a']<=policy['max_cost_increment']),
                 safety=bool(safety_upper<=policy['max_safety_upper']))
    return dict(primary={k:float(v) for k,v in primary.items()},metrics=metrics,policy=policy,gates=gates,
                n_per_arm=n,status='Candidate for human review' if all(gates.values()) else 'HOLD for investigation')


def planning_sizes(baseline,effects,alpha=.05,power=.8):
    return [sample_size_per_arm(baseline,baseline+effect,alpha,power) for effect in effects]
