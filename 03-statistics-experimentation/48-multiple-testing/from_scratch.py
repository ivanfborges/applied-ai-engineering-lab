"""Educational multiple-testing adjustments; not a statistical library replacement."""

import numpy as np


METHODS = ("uncorrected", "bonferroni", "holm", "bh")


def valid_level(level):
    """Return a finite scalar error-control level strictly between zero and one."""
    if not np.isscalar(level) or isinstance(level, (bool, np.bool_)) or np.iscomplexobj(level):
        raise ValueError("level must be a real scalar strictly between 0 and 1")
    try:
        value = float(level)
    except (TypeError, ValueError) as error:
        raise ValueError("level must be a real scalar strictly between 0 and 1") from error
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("level must be a real scalar strictly between 0 and 1")
    return value


def valid_pvalues(p_values, ndim=1):
    """Validate one family, or a matrix with one family/trajectory per row."""
    if np.iscomplexobj(p_values):
        raise ValueError("p-values must be real")
    try:
        p = np.asarray(p_values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("p-values must be numeric") from error
    if p.ndim != ndim or p.size == 0 or any(size == 0 for size in p.shape):
        raise ValueError(f"p-values must be a nonempty {ndim}D array")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    return p


def adjust_pvalues(p_values, method="bh"):
    """Adjust one complete, prespecified family and preserve its original order.

    Bonferroni/Holm control FWER with valid null p-values under arbitrary
    dependence. BH controls FDR under independence or appropriate PRDS conditions.
    """
    p = valid_pvalues(p_values)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    m = p.size
    if method == "uncorrected":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(1.0, m * p)

    order = np.argsort(p, kind="stable")
    ordered = p[order]
    if method == "holm":
        adjusted = np.maximum.accumulate((m - np.arange(m)) * ordered)
    else:
        scaled = m * ordered / np.arange(1, m + 1)
        # The suffix minimum implements BH's largest-qualifying-rank rule.
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(1.0, adjusted)
    return result


def reject_hypotheses(p_values, level=0.05, method="bh"):
    """Use inclusive rejection boundaries for all methods."""
    level = valid_level(level)
    return adjust_pvalues(p_values, method) <= level


def main():
    p = [0.001, 0.012, 0.021, 0.049, 0.2]
    print("Synthetic illustrative p-values:", p)
    for method in METHODS:
        adjusted = adjust_pvalues(p, method)
        print(f"{method:12s} adjusted={adjusted} reject={adjusted <= 0.05}")


if __name__ == "__main__":
    main()
