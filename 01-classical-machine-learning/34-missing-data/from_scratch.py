"""Educational dense numerical median imputation with fit-time indicators."""

from __future__ import annotations

import numpy as np


def numeric_matrix(X):
    """Accept finite numbers and NaN, with at least one row and column."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("X must be a nonempty two-dimensional numerical matrix.")
    if np.isinf(values).any():
        raise ValueError("X may contain NaN for missing values, but not infinity.")
    return values


class MedianImputerWithIndicator:
    """Store training medians; append indicators for columns missing at fit.

    All-empty training columns raise an error rather than silently disappearing.
    This class demonstrates state and schema boundaries, not sklearn integration.
    """

    def fit(self, X):
        values = numeric_matrix(X)
        missing = np.isnan(values)
        empty = np.flatnonzero(missing.all(axis=0))
        if empty.size:
            raise ValueError(f"No observed training values in columns {empty.tolist()}.")
        self.medians_ = np.nanmedian(values, axis=0)
        self.indicator_features_ = np.flatnonzero(missing.any(axis=0))
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        if not hasattr(self, "medians_"):
            raise RuntimeError("Call fit on training data before transform.")
        values = numeric_matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("X must have the same number and order of columns as training.")
        missing = np.isnan(values)
        # The mask comes from the original input, before filling its NaNs.
        filled = np.where(missing, self.medians_, values)
        indicators = missing[:, self.indicator_features_].astype(float)
        return np.column_stack((filled, indicators))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def main():
    train = np.array([[1, 10], [2, np.nan], [3, 30], [np.nan, 40]])
    test = np.array([[np.nan, 15], [4, np.nan]])
    imputer = MedianImputerWithIndicator().fit(train)
    print("Training medians:", imputer.medians_)
    print("Indicator columns:", imputer.indicator_features_)
    print("Test values followed by indicators:\n", imputer.transform(test))


if __name__ == "__main__":
    main()
