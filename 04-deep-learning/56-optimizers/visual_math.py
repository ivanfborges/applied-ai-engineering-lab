"""Vectorized toy objectives and real optimizer histories, independent of rendering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from from_scratch import METHODS, Optimizer, finite_array, scalar


COLORS = {"sgd": "#2474b5", "momentum": "#e38021", "rmsprop": "#26946a",
          "adam": "#d44852", "adamw": "#8759b6"}
LABELS = {"sgd": "SGD", "momentum": "Momentum", "rmsprop": "RMSProp",
          "adam": "Adam", "adamw": "AdamW"}


@dataclass(frozen=True)
class Objective:
    name: str = "quadratic"
    a: float = 40.0
    b: float = 1.0
    angle: float = 30.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if self.name not in ("quadratic", "rosenbrock"):
            raise ValueError("objective must be quadratic or rosenbrock.")
        scalar(self.a, "a", strict=True)
        scalar(self.b, "b", strict=True)
        scalar(self.angle, "angle", lower=-np.inf)
        if finite_array(self.center, "center").shape != (2,):
            raise ValueError("center must have two coordinates.")

    @property
    def rotation(self):
        c, s = np.cos(np.deg2rad(self.angle)), np.sin(np.deg2rad(self.angle))
        return np.array([[c, s], [-s, c]])

    @property
    def optimum(self):
        return np.array(self.center) if self.name == "quadratic" else np.array([1., 1.])

    @property
    def title(self):
        if self.name == "rosenbrock":
            return "Rosenbrock: (1-x)^2 + 100(y-x^2)^2"
        return f"Rotated quadratic: a={self.a:g}, b={self.b:g}, angle={self.angle:g} deg"

    def value(self, points):
        points = finite_array(points, "objective points")
        if points.shape[-1:] != (2,):
            raise ValueError("objective points must be finite with final dimension 2.")
        with np.errstate(over="raise", invalid="raise"):
            if self.name == "rosenbrock":
                x, y = points[..., 0], points[..., 1]
                return (1 - x)**2 + 100 * (y - x*x)**2
            local = (points - self.center) @ self.rotation.T
            return 0.5 * np.sum(local**2 * [self.a, self.b], axis=-1)

    def gradient(self, point):
        point = finite_array(point, "point")
        if point.shape != (2,):
            raise ValueError("gradient requires one two-dimensional point.")
        with np.errstate(over="raise", invalid="raise"):
            if self.name == "rosenbrock":
                x, y = point
                return np.array([2*(x-1) - 400*x*(y-x*x), 200*(y-x*x)])
            return ((point - self.center) @ self.rotation.T * [self.a, self.b]) @ self.rotation


@dataclass
class Trace:
    method: str
    lr: float
    weight_decay: float
    points: np.ndarray
    losses: np.ndarray
    history: dict[str, np.ndarray]
    status: str
    noise: float

    @property
    def steps(self):
        return len(self.points) - 1


def integer(value, name, minimum=1, maximum=2000):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def simulate(objective: Objective, method: str, lr: float, *, start=(-2., 1.5), steps=80,
             seed=56, noise=0., weight_decay=0., momentum=.9, rho=.99,
             beta1=.9, beta2=.999, epsilon=1e-8, loss_limit=1e12, parameter_limit=1e6) -> Trace:
    steps = integer(steps, "steps")
    seed = integer(seed, "seed", 0, 2**32-1)
    noise = scalar(noise, "noise")
    scalar(loss_limit, "loss_limit", strict=True)
    scalar(parameter_limit, "parameter_limit", strict=True)
    optimizer = Optimizer(start, method, lr, weight_decay=weight_decay, momentum=momentum,
                          alpha=rho, beta1=beta1, beta2=beta2, eps=epsilon)
    if optimizer.parameters.shape != (2,):
        raise ValueError("start must contain two coordinates.")
    rng = np.random.default_rng(seed)
    points = [optimizer.parameters]
    losses = [float(objective.value(points[0]))]
    names = ("gradient", "exact_gradient", "optimizer_input", "adaptive_delta", "decay_delta",
             "total_delta", "previous_first", "first", "second", "first_hat", "second_hat", "scaling")
    history = {name: [] for name in names}
    status = "completed"
    for t in range(1, steps+1):
        previous = optimizer.state.get("first", np.zeros(2))
        try:
            exact = objective.gradient(optimizer.parameters)
            gradient = exact + noise * rng.standard_normal(2)
            point = optimizer.step(gradient)
            loss = float(objective.value(point))
            if not np.isfinite(loss):
                raise FloatingPointError("nonfinite objective")
        except (ValueError, FloatingPointError) as error:
            status = f"stopped before recording step {t}: {error}"
            break
        state = optimizer.state
        first = state.get("first", np.zeros(2))
        second = state.get("second", np.zeros(2))
        first_hat = first / (1-beta1**t) if method in ("adam", "adamw") else first
        second_hat = second / (1-beta2**t) if method in ("adam", "adamw") else second
        scaling = (lr / (np.sqrt(second_hat)+epsilon) if method in ("rmsprop", "adam", "adamw")
                   else np.full(2, lr))
        data = dict(optimizer.last_step, exact_gradient=exact, previous_first=previous,
                    first=first, second=second, first_hat=first_hat, second_hat=second_hat, scaling=scaling)
        points.append(point)
        losses.append(loss)
        for name in names:
            history[name].append(data[name])
        if loss > loss_limit or np.linalg.norm(point) > parameter_limit:
            status = f"divergence guard reached at step {t}: loss>{loss_limit:g} or norm>{parameter_limit:g}"
            break
    return Trace(method, float(lr), weight_decay, np.array(points), np.array(losses),
                 {key: np.asarray(value).reshape(-1, 2) for key, value in history.items()}, status, noise)


def reference_rates(objective):
    if objective.name == "rosenbrock":
        return dict(zip(METHODS, [.001, .0005, .01, .025, .025]))
    curvature = max(objective.a, objective.b)
    return dict(zip(METHODS, [1.6/curvature, 1.4/curvature, .09, .09, .09]))


def learning_rates(objective, method):
    reference = reference_rates(objective)[method]
    high = (2.2/max(objective.a, objective.b) if method == "sgd" and objective.name == "quadratic"
            else 4.4/max(objective.a, objective.b) if method == "momentum" and objective.name == "quadratic"
            else reference*20)
    return {"small": reference/20, "reference": reference, "large": high}


def compare(objective, *, methods=METHODS, lr=None, **kwargs):
    rates = reference_rates(objective)
    return [simulate(objective, method, rates[method] if lr is None else lr, **kwargs)
            for method in methods]


def constant_gradient(steps=20, beta1=.9, beta2=.999, epsilon=1e-8):
    steps = integer(steps, "steps")
    optimizer = Optimizer([0., 0.], "adam", .1, beta1=beta1, beta2=beta2, eps=epsilon)
    gradient = np.array([.7, -1.2])
    first, second, first_hat, second_hat = [], [], [], []
    for t in range(1, steps+1):
        optimizer.step(gradient)
        state = optimizer.state
        first.append(state["first"])
        second.append(state["second"])
        first_hat.append(state["first"]/(1-beta1**t))
        second_hat.append(state["second"]/(1-beta2**t))
    return {"gradient": gradient, "first": np.array(first), "second": np.array(second),
            "first_hat": np.array(first_hat), "second_hat": np.array(second_hat)}


def frame_steps(steps, frames):
    return np.unique(np.linspace(0, steps, min(steps+1, frames), dtype=int))


def grid(objective, traces, resolution=75, fixed=False):
    points = np.vstack([t.points[:1] if fixed else t.points for t in traces] + [objective.optimum[None, :]])
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum((high-low)*.18, .6)
    x, y = np.meshgrid(np.linspace(low[0]-padding[0], high[0]+padding[0], resolution),
                       np.linspace(low[1]-padding[1], high[1]+padding[1], resolution))
    return x, y, objective.value(np.stack((x, y), axis=-1))
