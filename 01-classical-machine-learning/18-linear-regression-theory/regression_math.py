"""Numerical experiments for the Day 18 visual lab; synthetic data only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def checked_int(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return int(value)


def finite(value, name, low=-1e6, high=1e6):
    if not np.isscalar(value) or not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}.")
    return float(value)


def generator(seed):
    return np.random.default_rng(checked_int(seed, "seed", 0, 2**32 - 1))


def design_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite vector or matrix.")
    return np.column_stack((np.ones(len(X)), X))


def paired(X, y):
    D = design_matrix(X)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(D),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector with one value per row.")
    return D, y


def fit_ols(X, y):
    D, y = paired(X, y)
    beta, _, rank, singular = np.linalg.lstsq(D, y, rcond=None)
    pred = D @ beta
    return dict(beta=beta, pred=pred, residual=y-pred, rank=int(rank),
                singular=singular, design=D)


def scores(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.ndim != 1 or not y.size or pred.shape != y.shape:
        raise ValueError("Scores need matching nonempty vectors.")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Scores require finite values.")
    rss = float(np.sum((y-pred)**2))
    sst = float(np.sum((y-y.mean())**2))
    return dict(RSS=rss, MSE=rss/len(y), RMSE=np.sqrt(rss/len(y)),
                R2=1-rss/sst if sst > 0 else np.nan, SST=sst)


def line_data(n=80, noise=1.0, intercept=2.0, slope=1.5, seed=42, domain=(-3., 3.)):
    n = checked_int(n, "n", 5, 1000)
    noise = finite(noise, "noise", 0, 20)
    intercept, slope = finite(intercept, "intercept"), finite(slope, "slope")
    low, high = (finite(v, "domain") for v in domain)
    if low >= high:
        raise ValueError("Domain lower bound must be below upper bound.")
    rng = generator(seed)
    x = np.sort(rng.uniform(low, high, n))
    mean = intercept + slope*x
    return x, mean + rng.normal(0, noise, n), mean


def loss_grid(x, y, intercepts, slopes):
    D, y = paired(x, y)
    if D.shape[1] != 2:
        raise ValueError("Loss grid needs exactly one predictor.")
    b0, b1 = np.asarray(intercepts, float), np.asarray(slopes, float)
    if any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in (b0, b1)):
        raise ValueError("Grid axes must be finite nonempty vectors.")
    if len(b0)*len(b1)*len(y) > 5_000_000:
        raise ValueError("Grid too large for this educational demo.")
    residual = y[None, None, :] - b0[None, :, None] - b1[:, None, None]*D[:, 1]
    return np.sum(residual**2, axis=2)


def gradient_descent(x, y, start=(6., -2.), rate_ratio=0.6, steps=80):
    """Optimize MSE; stable step size is below 2 / largest Hessian eigenvalue."""
    D, y = paired(x, y)
    steps = checked_int(steps, "steps", 1, 500)
    ratio = finite(rate_ratio, "rate_ratio", .01, 1.5)
    beta = np.asarray(start, float)
    if beta.shape != (D.shape[1],) or not np.isfinite(beta).all():
        raise ValueError("Start must match the design columns and be finite.")
    hessian = 2*D.T @ D/len(y)
    critical = 2/np.linalg.eigvalsh(hessian)[-1]
    learning_rate = ratio*critical
    path, losses = [beta.copy()], [np.mean((y-D@beta)**2)]
    status = "step budget reached"
    for _ in range(steps):
        candidate = beta - learning_rate*(2*D.T@(D@beta-y)/len(y))
        with np.errstate(over="ignore", invalid="ignore"):
            loss = np.mean((y-D@candidate)**2)
        if not np.isfinite(loss) or loss > 1e10:
            status = "stopped: divergence guard"
            break
        path.append(candidate.copy())
        losses.append(float(loss))
        if np.linalg.norm(candidate-beta) < 1e-10:
            status = "parameter changes below tolerance"
            break
        beta = candidate
    return dict(path=np.array(path), losses=np.array(losses), rate=learning_rate,
                critical=critical, status=status)


def correlated_data(n=100, rho=.98, noise=1., seed=42):
    n = checked_int(n, "n", 5, 1000)
    rho = finite(rho, "rho", -.9999, .9999)
    noise = finite(noise, "noise", 0, 20)
    rng = generator(seed)
    z = rng.normal(size=(n, 2))
    X = np.column_stack((z[:, 0], rho*z[:, 0]+np.sqrt(1-rho**2)*z[:, 1]))
    mean = 2+X @ np.array([3., -2.])
    return X, mean+rng.normal(0, noise, n), mean


def fitted_model(X, y, alpha=0.):
    finite(alpha, "alpha", 0, 10000)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
    model.fit(X, y)
    scale, estimator = model.steps[0][1], model.steps[1][1]
    coef = estimator.coef_/scale.scale_
    intercept = estimator.intercept_-coef @ scale.mean_
    return model, np.r_[intercept, coef]


def stability(n=80, rho=.99, noise=1., alpha=10., repeats=80, seed=42):
    """Repeated independent training samples, evaluated on one fixed fresh design."""
    repeats = checked_int(repeats, "repeats", 5, 200)
    rng = generator(seed)
    test_seed = int(rng.integers(0, 2**32-1))
    Xtest, ytest, mean = correlated_data(250, rho, noise, test_seed)
    betas, predictions = [], []
    for _ in range(repeats):
        X, y, _ = correlated_data(n, rho, noise, int(rng.integers(0, 2**32-1)))
        ols = LinearRegression().fit(X, y)
        ridge, beta = fitted_model(X, y, alpha)
        betas.append([np.r_[ols.intercept_, ols.coef_], beta])
        predictions.append([ols.predict(Xtest), ridge.predict(Xtest)])
    betas, predictions = np.array(betas), np.array(predictions)
    bias2 = np.mean((predictions.mean(axis=0)-mean)**2, axis=1)
    variance = np.mean(predictions.var(axis=0, ddof=0), axis=1)
    mean_error = np.mean((predictions-mean)**2, axis=(0, 2))
    return dict(beta=betas, predictions=predictions, bias2=bias2, variance=variance,
                mean_error=mean_error, Xtest=Xtest, ytest=ytest, mean=mean,
                rmse=np.sqrt(np.mean((predictions-ytest)**2, axis=2)))


def curvature(n=100, noise=1., seed=42):
    x, _, _ = line_data(n, noise, seed=seed)
    rng = generator(seed+1)
    y = 3*x+2*x*x+rng.normal(0, noise, n)
    return x, y, fit_ols(x, y), fit_ols(np.column_stack((x, x*x)), y)


def variance_data(n=150, strength=.6, seed=42):
    strength = finite(strength, "strength", 0, 2)
    x, _, _ = line_data(n, 0, seed=seed, domain=(0, 6))
    z = generator(seed+1).normal(size=n)
    sigma = .3+strength*x
    # Match average conditional variance; use paired standardized noise draws.
    constant = np.sqrt(np.mean(sigma*sigma))
    return x, 2+1.5*x+constant*z, 2+1.5*x+sigma*z, sigma


def projection(target=(2., -1., 4.)):
    y = np.asarray(target, float)
    if y.shape != (3,) or not np.isfinite(y).all():
        raise ValueError("Projection requires three finite target coordinates.")
    fit = fit_ols(np.array([-1., 0., 1.]), y)
    fit["y"] = y
    return fit


def outlier_case(n=50, noise=.7, point_x=8., offset=15., seed=42):
    point_x, offset = finite(point_x, "point_x"), finite(offset, "offset")
    x, y, _ = line_data(n, noise, seed=seed)
    xa, ya = np.r_[x, point_x], np.r_[y, 2+1.5*point_x+offset]
    before, after = fit_ols(x, y), fit_ols(xa, ya)
    D = after["design"]
    leverage = np.sum(np.linalg.svd(D, full_matrices=False)[0]**2, axis=1)[-1]
    xt, yt, _ = line_data(250, noise, seed=seed+1)
    return dict(x=x, y=y, xa=xa, ya=ya, before=before, after=after,
                leverage=leverage, point_residual=after["residual"][-1],
                rmse=[scores(yt, design_matrix(xt)@f["beta"])["RMSE"]
                      for f in (before, after)])


def polynomial_experiment(n=35, noise=.5, seed=42):
    n = checked_int(n, "n", 20, 150)
    noise = finite(noise, "noise", 0, 5)
    rng = generator(seed)
    train = np.sort(rng.uniform(-1, 1, n))
    valid = np.sort(rng.uniform(-1, 1, 180))
    truth = lambda x: 1+2*x-1.5*x*x
    ytrain = truth(train)+rng.normal(0, noise, n)
    yvalid = truth(valid)+rng.normal(0, noise, len(valid))
    grid = np.linspace(-1, 1, 240)
    degrees, errors, curves, conditions = np.arange(1, 16), [], [], []
    for degree in degrees:
        model = make_pipeline(PolynomialFeatures(int(degree), include_bias=False),
                              LinearRegression())
        model.fit(train[:, None], ytrain)
        errors.append([scores(ytrain, model.predict(train[:, None]))["RMSE"],
                       scores(yvalid, model.predict(valid[:, None]))["RMSE"]])
        curves.append(model.predict(grid[:, None]))
        conditions.append(np.linalg.cond(np.vander(train, degree+1, increasing=True)))
    return dict(train=train, valid=valid, ytrain=ytrain, yvalid=yvalid,
                degrees=degrees, errors=np.array(errors), curves=np.array(curves),
                grid=grid, mean=truth(grid), condition=np.array(conditions))


def normal_equations(delta=1.):
    delta = finite(delta, "delta", 1e-10, 1)
    x = np.array([-2., -1., 0., 1., 2.])
    second = x+delta*np.array([1., -1., 1., -1., 1.])
    X = np.column_stack((x, second))
    y = 2+3*x-2*second+np.array([.1, -.2, .1, .2, -.1])
    D = design_matrix(X)
    gram = D.T@D
    normal = np.linalg.pinv(gram)@D.T@y
    stable = np.linalg.lstsq(D, y, rcond=None)[0]
    model = LinearRegression().fit(X, y)
    library = np.r_[model.intercept_, model.coef_]
    return dict(X=D, y=y, gram=gram, rhs=D.T@y, normal=normal, stable=stable,
                library=library, condition=np.linalg.cond(D))