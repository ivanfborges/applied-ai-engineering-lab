"""Manually apply the chain rule through one nonlinear scalar neuron."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardPass:
    """Values produced by z = wx + b, a = tanh(z), L = (a - y)^2."""

    pre_activation: float
    activation: float
    loss: float


@dataclass(frozen=True)
class BackwardPass:
    """Local derivatives and the composed parameter gradients."""

    d_loss_d_activation: float
    d_activation_d_pre_activation: float
    d_loss_d_pre_activation: float
    d_loss_d_weight: float
    d_loss_d_bias: float


def forward(
    x: float,
    target: float,
    weight: float,
    bias: float,
) -> ForwardPass:
    """Evaluate the scalar computation graph."""
    pre_activation = weight * x + bias
    activation = math.tanh(pre_activation)
    loss = (activation - target) ** 2
    return ForwardPass(pre_activation, activation, loss)


def backward(
    x: float,
    target: float,
    weight: float,
    bias: float,
) -> BackwardPass:
    """Compose local derivatives from the loss back to weight and bias."""
    values = forward(x, target, weight, bias)

    # L = (a - target)^2
    d_loss_d_activation = 2.0 * (values.activation - target)

    # a = tanh(z)
    d_activation_d_pre_activation = 1.0 - values.activation**2

    # The upstream derivative is multiplied by the local derivative.
    d_loss_d_pre_activation = (
        d_loss_d_activation * d_activation_d_pre_activation
    )

    # z = weight * x + bias, so dz/dweight = x and dz/dbias = 1.
    d_loss_d_weight = d_loss_d_pre_activation * x
    d_loss_d_bias = d_loss_d_pre_activation

    return BackwardPass(
        d_loss_d_activation=d_loss_d_activation,
        d_activation_d_pre_activation=d_activation_d_pre_activation,
        d_loss_d_pre_activation=d_loss_d_pre_activation,
        d_loss_d_weight=d_loss_d_weight,
        d_loss_d_bias=d_loss_d_bias,
    )


def numerical_gradient(
    x: float,
    target: float,
    weight: float,
    bias: float,
    parameter: str,
    epsilon: float = 1e-6,
) -> float:
    """Estimate one parameter derivative with centered finite differences."""
    if parameter == "weight":
        loss_plus = forward(x, target, weight + epsilon, bias).loss
        loss_minus = forward(x, target, weight - epsilon, bias).loss
    elif parameter == "bias":
        loss_plus = forward(x, target, weight, bias + epsilon).loss
        loss_minus = forward(x, target, weight, bias - epsilon).loss
    else:
        raise ValueError("parameter must be 'weight' or 'bias'")

    return (loss_plus - loss_minus) / (2.0 * epsilon)


def main() -> None:
    x = 1.5
    target = 0.8
    weight = 0.4
    bias = -0.2

    values = forward(x, target, weight, bias)
    gradients = backward(x, target, weight, bias)
    numerical_d_weight = numerical_gradient(
        x,
        target,
        weight,
        bias,
        parameter="weight",
    )
    numerical_d_bias = numerical_gradient(
        x,
        target,
        weight,
        bias,
        parameter="bias",
    )

    print("Forward pass")
    print(f"z = weight * x + bias: {values.pre_activation:.8f}")
    print(f"a = tanh(z):           {values.activation:.8f}")
    print(f"L = (a - target)^2:    {values.loss:.8f}")

    print("\nLocal derivatives and chain rule")
    print(f"dL/da: {gradients.d_loss_d_activation:.8f}")
    print(f"da/dz: {gradients.d_activation_d_pre_activation:.8f}")
    print(f"dL/dz = dL/da * da/dz: {gradients.d_loss_d_pre_activation:.8f}")

    print("\nGradient check")
    print(f"Manual    dL/dw: {gradients.d_loss_d_weight:.8f}")
    print(f"Numerical dL/dw: {numerical_d_weight:.8f}")
    print(f"Manual    dL/db: {gradients.d_loss_d_bias:.8f}")
    print(f"Numerical dL/db: {numerical_d_bias:.8f}")

    if not math.isclose(
        gradients.d_loss_d_weight,
        numerical_d_weight,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError("The manual weight gradient failed its check.")

    if not math.isclose(
        gradients.d_loss_d_bias,
        numerical_d_bias,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise RuntimeError("The manual bias gradient failed its check.")

    learning_rate = 0.1
    updated_weight = weight - learning_rate * gradients.d_loss_d_weight
    updated_bias = bias - learning_rate * gradients.d_loss_d_bias
    updated_loss = forward(x, target, updated_weight, updated_bias).loss

    print("\nOne parameter update")
    print(f"Loss before update: {values.loss:.8f}")
    print(f"Loss after update:  {updated_loss:.8f}")


if __name__ == "__main__":
    main()
