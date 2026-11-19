"""Educational 2D cross-correlation, pooling and spatial bookkeeping in NumPy."""

from numbers import Integral

import numpy as np


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def output_size(size, kernel, stride=1, padding=0, dilation=1):
    """One spatial axis, symmetric padding, floor mode (no ceil_mode)."""
    size = integer(size, "size")
    kernel = integer(kernel, "kernel")
    stride = integer(stride, "stride")
    padding = integer(padding, "padding", 0)
    dilation = integer(dilation, "dilation")
    result = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if result < 1:
        raise ValueError("effective kernel must fit the padded input")
    return result


def receptive_field(layers):
    """Return (bounding span, jump) for one axis of a single sequential path.

    Each layer is (kernel, stride, dilation). Padding affects alignment and
    boundary coverage, which this size-only calculation does not describe.
    """
    span, jump = 1, 1
    for kernel, stride, dilation in layers:
        kernel = integer(kernel, "kernel")
        stride = integer(stride, "stride")
        dilation = integer(dilation, "dilation")
        span += dilation * (kernel - 1) * jump
        jump *= stride
    return span, jump


def _matrix(values, name):
    array = np.asarray(values)
    if (array.ndim != 2 or array.size == 0
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)):
        raise ValueError(f"{name} must be a nonempty real numeric 2D array")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


def correlate2d(image, kernel, *, stride=1, padding=0, dilation=1):
    """Single channel, no bias; scalar stride/padding/dilation on both axes.

    The kernel is not flipped, matching the convention used by nn.Conv2d.
    Explicit loops expose the indexing; they are not an efficient backend.
    """
    image = _matrix(image, "image")
    kernel = _matrix(kernel, "kernel")
    height = output_size(image.shape[0], kernel.shape[0], stride, padding, dilation)
    width = output_size(image.shape[1], kernel.shape[1], stride, padding, dilation)
    stride, padding, dilation = int(stride), int(padding), int(dilation)
    padded = np.pad(image, padding)
    kh, kw = kernel.shape
    output = np.empty((height, width), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for row in range(height):
            for col in range(width):
                top, left = row * stride, col * stride
                patch = padded[top:top + dilation * (kh - 1) + 1:dilation,
                               left:left + dilation * (kw - 1) + 1:dilation]
                output[row, col] = np.sum(patch * kernel)
    return output


def pool2d(image, kernel=2, *, stride=2, mode="max"):
    """Unpadded square max/average windows, floor mode; trailing pixels may drop."""
    image = _matrix(image, "image")
    if mode not in ("max", "average"):
        raise ValueError("mode must be 'max' or 'average'")
    height = output_size(image.shape[0], kernel, stride)
    width = output_size(image.shape[1], kernel, stride)
    kernel, stride = int(kernel), int(stride)
    reduce = np.max if mode == "max" else np.mean
    output = np.empty((height, width), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for row in range(height):
            for col in range(width):
                patch = image[row * stride:row * stride + kernel,
                              col * stride:col * stride + kernel]
                output[row, col] = reduce(patch)
    return output


def main():
    image = np.tile([0., 0., 0., 1., 1.], (5, 1))
    kernel = np.tile([-1., 0., 1.], (3, 1))
    response = correlate2d(image, kernel)
    print("Synthetic intensity step; valid cross-correlation (no kernel flip):")
    print(response)
    print("2x2 max pooling, stride 2:", pool2d(response).tolist())
    print("2x2 average pooling, stride 2:", pool2d(response, mode="average").tolist())
    print("Conv3 -> Pool2 -> Conv3 (span, jump):",
          receptive_field([(3, 1, 1), (2, 2, 1), (3, 1, 1)]))


if __name__ == "__main__":
    main()
