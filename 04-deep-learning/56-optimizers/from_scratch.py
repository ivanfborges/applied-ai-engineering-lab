"""Educational dense float64 optimizer updates; no autograd or training framework."""

from __future__ import annotations

import json
from numbers import Real

import numpy as np


METHODS = ("sgd", "momentum", "rmsprop", "adam", "adamw")


def finite_array(value, name: str) -> np.ndarray:
    """Reject silent complex casts, booleans, empty arrays and nonfinite values."""
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array.")
    array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def scalar(value, name: str, *, lower: float = 0, upper=None, strict=False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    value = float(value)
    if (not np.isfinite(value) or value < lower or (strict and value == lower)
            or (upper is not None and value >= upper)):
        raise ValueError(f"{name} is outside its allowed range.")
    return value


class Optimizer:
    """Own one dense parameter array and its state; step counts advance internally.

    Momentum has no dampening/Nesterov; RMSProp is uncentered without momentum.
    Decay is coupled for all methods except AdamW. Epsilon is outside sqrt.
    Returned parameters/state are copies. Recreate the optimizer to reset it.
    """

    def __init__(self, parameters, method: str, lr: float, *, weight_decay=0.0,
                 momentum=0.9, alpha=0.99, beta1=0.9, beta2=0.999, eps=1e-8):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")
        self.method = method
        self.lr = scalar(lr, "lr", strict=True)
        self.weight_decay = scalar(weight_decay, "weight_decay")
        self.momentum = scalar(momentum, "momentum", upper=1)
        self.alpha = scalar(alpha, "alpha", upper=1)
        self.beta1 = scalar(beta1, "beta1", upper=1)
        self.beta2 = scalar(beta2, "beta2", upper=1)
        self.eps = scalar(eps, "eps", strict=True)
        self._parameters = finite_array(parameters, "parameters")
        self._steps = 0
        self._last_step = None
        self._state = {}
        if method in ("momentum", "adam", "adamw"):
            self._state["first"] = np.zeros_like(self._parameters)
        if method in ("rmsprop", "adam", "adamw"):
            self._state["second"] = np.zeros_like(self._parameters)

    @property
    def parameters(self) -> np.ndarray:
        return self._parameters.copy()

    @property
    def state(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self._state.items()}

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def last_step(self) -> dict[str, np.ndarray] | None:
        """Separate the supplied gradient, optimizer input and actual update components."""
        if self._last_step is None:
            return None
        return {key: value.copy() for key, value in self._last_step.items()}

    def step(self, gradient) -> np.ndarray:
        """Apply one supplied gradient; reject invalid/overflowing updates atomically."""
        gradient = finite_array(gradient, "gradient")
        if gradient.shape != self._parameters.shape:
            raise ValueError("gradient shape must match parameters exactly; no broadcasting.")
        parameters = self._parameters
        step = self._steps + 1
        state = {}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                direction = (gradient if self.method == "adamw" else
                             gradient + self.weight_decay * parameters)
                if self.method == "sgd":
                    update = direction
                elif self.method == "momentum":
                    state["first"] = self.momentum * self._state["first"] + direction
                    update = state["first"]
                elif self.method == "rmsprop":
                    state["second"] = (self.alpha * self._state["second"]
                                       + (1 - self.alpha) * direction**2)
                    update = direction / (np.sqrt(state["second"]) + self.eps)
                else:
                    state["first"] = (self.beta1 * self._state["first"]
                                      + (1 - self.beta1) * direction)
                    state["second"] = (self.beta2 * self._state["second"]
                                       + (1 - self.beta2) * direction**2)
                    first_hat = state["first"] / (1 - self.beta1**step)
                    second_hat = state["second"] / (1 - self.beta2**step)
                    update = first_hat / (np.sqrt(second_hat) + self.eps)
                # Shrink the OLD parameters, independently of the adaptive direction.
                base = ((1 - self.lr * self.weight_decay) * parameters
                        if self.method == "adamw" else parameters)
                result = base - self.lr * update
        except FloatingPointError as error:
            raise ValueError("Update overflowed; rescale inputs or reduce hyperparameters.") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("Update produced nonfinite parameters.")
        self._last_step = {"gradient": gradient.copy(), "optimizer_input": direction.copy(),
                           "adaptive_delta": -self.lr * update,
                           "decay_delta": base - parameters,
                           "total_delta": result - parameters}
        self._parameters, self._state, self._steps = result, state, step
        return result.copy()


def quadratic(parameters) -> tuple[float, np.ndarray]:
    """Synthetic f(w)=0.5*(10*w[0]**2+2*w[1]**2), minimum at zero."""
    parameters = finite_array(parameters, "parameters")
    if parameters.shape != (2,):
        raise ValueError("quadratic requires exactly two parameters.")
    with np.errstate(over="raise", invalid="raise"):
        gradient = np.array([10.0, 2.0]) * parameters
        loss = float(0.5 * parameters @ gradient)
    return loss, gradient


def run_demo() -> dict:
    rates = dict(zip(METHODS, [0.06, 0.02, 0.1, 0.1, 0.1]))
    rows = []
    for method, lr in rates.items():
        optimizer = Optimizer([2.0, -1.0], method, lr)
        initial_loss = quadratic(optimizer.parameters)[0]
        for _ in range(50):
            optimizer.step(quadratic(optimizer.parameters)[1])
        rows.append({"method": method, "lr": lr, "steps": optimizer.steps,
                     "weight_decay": 0.0, "initial_loss": initial_loss,
                     "final_loss": quadratic(optimizer.parameters)[0],
                     "parameters": optimizer.parameters.tolist()})
    pairs = []
    for decay in (0.0, 0.1):
        optimizers = [Optimizer([2.0, -1.0], method, 0.1, weight_decay=decay)
                      for method in ("adam", "adamw")]
        for _ in range(50):
            for optimizer in optimizers:
                optimizer.step(quadratic(optimizer.parameters)[1])
        pairs.append({"weight_decay": decay,
                      "max_parameter_difference": float(np.max(np.abs(
                          optimizers[0].parameters - optimizers[1].parameters))),
                      "parameter_norms": [float(np.linalg.norm(o.parameters))
                                          for o in optimizers]})
    return {"data": "synthetic deterministic quadratic, Hessian eigenvalues 10 and 2",
            "numpy": np.__version__, "runs": rows, "adam_adamw_pairs": pairs,
            "interpretation_status": "author review required"}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, allow_nan=False))
