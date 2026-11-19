"""Inspect an untrained NumPy LSTM forward step; no BPTT or optimizer."""

from numbers import Integral, Real

import numpy as np


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def valid_seed(value):
    value = integer(value, "seed", 0)
    if value >= 2**32:
        raise ValueError("seed must be < 2**32")
    return value


def finite_real(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    return float(value)


def sigmoid(values):
    """Stable sigmoid without evaluating exp(large positive values)."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sigmoid requires finite real inputs")
    return np.exp(-np.logaddexp(0.0, -values))


class SimpleLSTMCell:
    """One unbatched cell, with concatenated [x,h] and gate order i,f,g,o.

    A combined bias equals PyTorch's input bias plus recurrent bias. The
    optional forget bias is illustrative, not a trained retention policy.
    """

    def __init__(self, input_size=2, hidden_size=3, seed=60, forget_bias=1.0):
        self.input_size = integer(input_size, "input_size")
        self.hidden_size = integer(hidden_size, "hidden_size")
        rng = np.random.default_rng(valid_seed(seed))
        self.W = rng.normal(0, 0.3, (4 * self.hidden_size, self.input_size + self.hidden_size))
        self.b = np.zeros(4 * self.hidden_size)
        self.b[self.hidden_size:2 * self.hidden_size] = finite_real(forget_bias, "forget_bias")

    def step(self, x, h_prev, c_prev):
        vectors = []
        for value, size, name in ((x, self.input_size, "x"),
                                  (h_prev, self.hidden_size, "h_prev"),
                                  (c_prev, self.hidden_size, "c_prev")):
            value = np.asarray(value, dtype=np.float64)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite vector of shape ({size},)")
            vectors.append(value)
        x, h_prev, c_prev = vectors
        if (self.W.shape != (4 * self.hidden_size, self.input_size + self.hidden_size)
                or self.b.shape != (4 * self.hidden_size,)
                or not np.isfinite(self.W).all() or not np.isfinite(self.b).all()):
            raise ValueError("cell parameters must have their original shapes and be finite")
        with np.errstate(over="raise", invalid="raise"):
            i_raw, f_raw, g_raw, o_raw = np.split(self.W @ np.concatenate([x, h_prev]) + self.b, 4)
            gates = {"i": sigmoid(i_raw), "f": sigmoid(f_raw),
                     "g": np.tanh(g_raw), "o": sigmoid(o_raw)}
            c = gates["f"] * c_prev + gates["i"] * gates["g"]
            h = gates["o"] * np.tanh(c)
        return h, c, gates


def main():
    cell = SimpleLSTMCell()
    h, c = np.zeros(3), np.zeros(3)
    sequence = ([1.0, 1.0], [0.2, 0.0], [-0.4, 0.0])
    print("Untrained synthetic forward steps; gates are not learned explanations.")
    for t, x in enumerate(sequence, 1):
        h, c, gates = cell.step(x, h, c)
        print(f"t={t} h={h.round(5)} c={c.round(5)} forget={gates['f'].round(5)}")


if __name__ == "__main__":
    main()
