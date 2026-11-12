"""Statistical mechanisms for the Day 48 visual lab; no rendering or file writes."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from example import estimate, family_metrics, normal_look_pvalues, peeking_decisions
from from_scratch import valid_level, valid_pvalues


METHODS = ("uncorrected", "bonferroni", "holm", "bh", "by")
LIBRARY_METHODS = {"bonferroni": "bonferroni", "holm": "holm", "bh": "fdr_bh", "by": "fdr_by"}


@dataclass(frozen=True)
class LabConfig:
    mode: str = "quick"
    seed: int = 48
    hypotheses: int = 20
    nonnull_proportion: float = 0.2
    effect: float = 3.0
    alpha: float = 0.05
    repetitions: int = 2000
    frames: int = 24
    null_tests: int = 1000
    looks: int = 10
    final_n: int = 1000

    def __post_init__(self):
        bounds = {"seed": (0, 2**32 - 1), "hypotheses": (2, 500),
                  "repetitions": (100, 50000), "frames": (8, 60),
                  "null_tests": (100, 20000), "looks": (1, 40),
                  "final_n": (1, 1000000)}
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.mode not in ("quick", "full"):
            raise ValueError("mode must be quick or full")
        valid_level(self.alpha)
        if not np.isfinite(self.nonnull_proportion) or not 0 <= self.nonnull_proportion <= 1:
            raise ValueError("nonnull_proportion must be in [0, 1]")
        if not np.isfinite(self.effect) or not 0 <= self.effect <= 8:
            raise ValueError("effect must be finite and in [0, 8]")
        if self.final_n < self.looks:
            raise ValueError("final_n must be at least the number of looks")
        if self.hypotheses * self.repetitions > 2000000:
            raise ValueError("hypotheses × repetitions must be <= 2,000,000 to bound memory")

    @property
    def signals(self):
        """Floor the requested proportion; report the actual integer count."""
        return int(np.floor(self.hypotheses * self.nonnull_proportion))

    def rng(self, experiment):
        """Stable, separate streams: selecting one experiment does not change it."""
        return np.random.default_rng(np.random.SeedSequence([self.seed, experiment]))


def independent_fwer(m, alpha):
    """Exact independent, all-null FWER; broadcastable integer counts and levels."""
    m = np.asarray(m)
    alpha = np.asarray(alpha)
    if (m.dtype.kind not in "iu" or np.any(m < 1) or np.iscomplexobj(alpha)
            or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("expected positive integer counts and levels in [0, 1]")
    with np.errstate(divide="ignore"):
        return -np.expm1(m * np.log1p(-alpha))


def adjusted_matrix(p_values, method):
    """Vectorized educational adjustment: rows are complete hypothesis families."""
    p = valid_pvalues(p_values, ndim=2)
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    m = p.shape[1]
    if method == "uncorrected":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1, p * m)
    order = np.argsort(p, axis=1, kind="stable")
    ordered = np.take_along_axis(p, order, axis=1)
    ranks = np.arange(1, m + 1)
    if method == "holm":
        adjusted = np.maximum.accumulate((m - ranks + 1) * ordered, axis=1)
    else:
        factor = np.sum(1 / ranks) if method == "by" else 1.0
        scaled = ordered * m * factor / ranks
        adjusted = np.minimum.accumulate(scaled[:, ::-1], axis=1)[:, ::-1]
    result = np.empty_like(p)
    np.put_along_axis(result, order, np.minimum(1, adjusted), axis=1)
    return result


def reference_check(p_values, alpha):
    """Check a bounded set of complete families against independent libraries."""
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as error:
        raise ImportError('Install canonical dependencies: python -m pip install -e ".[dev,experimentation]"') from error
    p = valid_pvalues(p_values, ndim=2)[:32]
    errors = {}
    for method, library in LIBRARY_METHODS.items():
        actual = adjusted_matrix(p, method)
        expected = np.stack([multipletests(row, alpha=alpha, method=library)[1] for row in p])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(actual <= alpha, expected <= alpha)
        errors[method] = float(np.max(np.abs(actual - expected)))
    return {"families_checked": len(p), "max_absolute_adjustment_error": errors}


def equicorrelation(m, rho):
    """Validate PSD, including the feasible negative equicorrelation boundary."""
    if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 2:
        raise ValueError("m must be an integer >= 2")
    if not np.isfinite(rho) or not -1 / (m - 1) <= rho <= 1:
        raise ValueError("rho must lie in [-1/(m-1), 1]")
    covariance = (1 - rho) * np.eye(m) + rho * np.ones((m, m))
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("correlation matrix must be positive semidefinite")
    return covariance


def normal_families(config, stream, rho=0.0, signals=None):
    """Two-sided normal tests, using a PSD nonnegative equicorrelation model.

    The common-factor construction is distributionally MVN with unit marginal
    variances. Reusing a stream across rho/effect values is intentional pairing.
    """
    if not 0 <= rho <= 1:
        raise ValueError("the visual generator supports only nonnegative rho")
    equicorrelation(config.hypotheses, rho)
    if signals is None:
        signals = config.signals
    if not isinstance(signals, (int, np.integer)) or not 0 <= signals <= config.hypotheses:
        raise ValueError("signals must be an integer from 0 through hypotheses")
    rng = config.rng(stream)
    independent = rng.normal(size=(config.repetitions, config.hypotheses))
    common = rng.normal(size=(config.repetitions, 1))
    z = np.sqrt(1 - rho) * independent + np.sqrt(rho) * common
    z[:, :signals] += config.effect
    p = 2 * norm.sf(np.abs(z))
    valid_pvalues(p, ndim=2)
    return z, p, np.arange(config.hypotheses) >= (signals if config.effect > 0 else 0)


def discovery_counts(rejected, true_null):
    """One TP/FP/FN/R/FDP value per independent simulated family."""
    # Reuse the original study's validation and FDR denominator contract.
    summary = family_metrics(rejected, true_null)
    tp = rejected[:, ~true_null].sum(axis=1)
    fp = rejected[:, true_null].sum(axis=1)
    r = tp + fp
    counts = {"tp": tp, "fp": fp, "fn": int((~true_null).sum()) - tp,
              "r": r, "fdp": fp / np.maximum(r, 1)}
    summary.update({name: estimate(counts[name]) for name in ("tp", "fp", "fn", "r")})
    return counts, summary


def summarize_family(p, true_null, alpha, methods=METHODS):
    counts, summaries = {}, {}
    for method in methods:
        counts[method], summaries[method] = discovery_counts(adjusted_matrix(p, method) <= alpha, true_null)
    return counts, summaries


def tradeoff_study(config):
    """Common random numbers isolate the effect sweep from changing noise."""
    base, _, true_null = normal_families(config, 4, signals=0)
    effects = np.unique(np.r_[0., 1., 2., 3., 4., config.effect])
    results, selected_counts, references = [], None, None
    for effect in effects:
        z = base.copy()
        z[:, :config.signals] += effect
        p = 2 * norm.sf(np.abs(z))
        true_null = np.arange(config.hypotheses) >= (config.signals if effect > 0 else 0)
        counts, metrics = summarize_family(p, true_null, config.alpha, METHODS[:-1])
        results.append({"effect": float(effect), "actual_signals": int((~true_null).sum()), "metrics": metrics})
        if effect == config.effect:
            selected_counts = counts
            references = reference_check(p, config.alpha)
    return results, selected_counts, references


def sequential_study(config):
    """All monitoring policies see the same independent normal block sums."""
    looks = np.linspace(0, config.final_n, config.looks + 1, dtype=int)[1:]
    sizes = np.diff(np.r_[0, looks])
    sums = config.rng(5).normal(size=(config.repetitions, config.looks)) * np.sqrt(sizes)
    z = np.cumsum(sums, axis=1) / np.sqrt(looks)
    p = normal_look_pvalues(sums, looks)
    decisions = peeking_decisions(p, config.alpha)
    candidates = np.flatnonzero(decisions["naive_peeking"] & ~decisions["fixed_horizon"])
    selected = int(candidates[0]) if candidates.size else 0
    crossings = np.flatnonzero(p[selected] <= config.alpha)
    details = {
        "rates": {name: estimate(mask) for name, mask in decisions.items()},
        "paired_naive_minus_fixed": estimate(decisions["naive_peeking"].astype(float) - decisions["fixed_horizon"]),
        "selected_index_zero_based": selected,
        "selection": "first naive rejection with final non-rejection" if candidates.size else "fallback first path; no requested crossing in pool",
        "pool_size": config.repetitions,
        "first_naive_crossing_index": int(crossings[0]) if crossings.size else None,
        "look_sizes": looks.tolist(), "selected_z": z[selected].tolist(),
        "selected_p": p[selected].tolist(),
        "same_paths_for_all_policies": True,
    }
    return looks, z, p, decisions, details


def dependence_study(config):
    """Global-null and mixed-family observations; no claim of general PRDS."""
    results = []
    for rho in (0., .25, .5, .75, .9):
        entry = {"rho": rho, "min_eigenvalue": float(np.linalg.eigvalsh(equicorrelation(config.hypotheses, rho)).min())}
        for label, signals in (("global_null", 0), ("mixed", config.signals)):
            _, p, true_null = normal_families(config, 6, rho, signals)
            _, entry[label] = summarize_family(p, true_null, config.alpha)
        results.append(entry)
    return results
