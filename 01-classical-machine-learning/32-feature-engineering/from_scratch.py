"""Dense-array transforms for learning, not a library replacement."""
import numpy as np


def _array(values, ndim):
    array = np.asarray(values, dtype=float)
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Expected a nonempty, finite {ndim}D numeric array.")
    return array


def fit_standardization(train):
    """Estimate column means and population standard deviations on training rows."""
    train = _array(train, 2)
    mean, scale = train.mean(axis=0), train.std(axis=0, ddof=0)
    return mean, np.where(scale == 0, 1.0, scale)


def transform_standardization(values, mean, scale):
    values = _array(values, 2)
    mean, scale = np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)
    if (mean.shape != (values.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("Expected finite column parameters and positive scales.")
    return (values - mean) / scale


def fit_quantile_thresholds(train, n_bins=4):
    """Distinct internal linear-quantile thresholds; ties reduce bin count."""
    train = _array(train, 1)
    if (isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer))
            or n_bins < 2):
        raise ValueError("n_bins must be an integer of at least 2.")
    edges = np.quantile(train, np.linspace(0, 1, n_bins + 1), method="linear")
    # A boundary at the minimum would create an empty first interval.
    return np.unique(edges[(edges > train.min()) & (edges < train.max())])


def transform_bins(values, thresholds):
    """Assign equality to the interval on the right; tails use outer bins."""
    values = _array(values, 1)
    thresholds = np.asarray(thresholds, dtype=float)
    if (thresholds.ndim != 1 or not np.isfinite(thresholds).all()
            or np.any(np.diff(thresholds) <= 0)):
        raise ValueError("Thresholds must be finite, strictly increasing, and 1D.")
    return np.searchsorted(thresholds, values, side="right")


def pair_product(values):
    """Only the product of two columns, without duplicate main effects."""
    values = _array(values, 2)
    if values.shape[1] != 2:
        raise ValueError("The interaction requires exactly two columns.")
    with np.errstate(over="ignore"):
        product = values[:, :1] * values[:, 1:]
    if not np.isfinite(product).all():
        raise ValueError("Interaction overflowed; rescale the inputs.")
    return product


def main():
    train = np.array([[25, 50000], [40, 80000], [55, 120000], [30, 65000]])
    holdout = np.array([[35, 70000], [60, 130000]])
    mean, scale = fit_standardization(train)
    scaled_train = transform_standardization(train, mean, scale)
    scaled_holdout = transform_standardization(holdout, mean, scale)
    thresholds = fit_quantile_thresholds(train[:, 1])
    print("Training means:", mean)
    print("Training scales:", scale)
    print("Scaled holdout:\n", scaled_holdout)
    print("Income thresholds:", thresholds)
    print("Holdout bin IDs:", transform_bins(holdout[:, 1], thresholds))
    print("Training interactions:", pair_product(scaled_train).ravel())
    print("Holdout interactions:", pair_product(scaled_holdout).ravel())


if __name__ == "__main__":
    main()
