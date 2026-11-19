"""Deterministic inputs and exact spatial bookkeeping shared by figures and UI."""

import numpy as np

from from_scratch import correlate2d, integer, output_size, receptive_field


FILTERS = {
    "Vertical edge": np.array([[-1., 0., 1.]] * 3),
    "Horizontal edge": np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]),
    "Sharpen": np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]),
    "Blur": np.ones((3, 3)) / 9,
    "Identity": np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
    "Random (seed 59)": np.random.default_rng(59).normal(0, 0.4, (3, 3)),
}

# (name, kernel, stride, padding), all dilation 1; includes boundary clipping.
RF_LAYERS = (("Input", 1, 1, 0), ("Conv 3x3", 3, 1, 1),
             ("MaxPool 2x2", 2, 2, 0), ("Conv 3x3 (2)", 3, 1, 1),
             ("Conv 3x3 (3)", 3, 1, 1))


def scene(size=16):
    size = integer(size, "size", 8)
    image = np.zeros((size, size), dtype=float)
    image[2:size - 2, 2] = 1
    image[size // 2, 2:size - 2] = 0.7
    image[2:size // 2, size // 2:size - 2] = 0.45
    rows, cols = np.ogrid[:size, :size]
    image[(rows - 3 * size / 4)**2 + (cols - 3 * size / 4)**2 < (size / 7)**2] = 0.9
    return image


def edge_image(size=8):
    size = integer(size, "size", 4)
    image = np.zeros((size, size))
    image[:, size // 2:] = 1
    return image


def convolution_steps(image, kernel, *, stride=1, padding=0):
    """Observed patches/products/sums in row-major output order."""
    output = correlate2d(image, kernel, stride=stride, padding=padding)
    padded = np.pad(np.asarray(image, dtype=float), padding)
    kh, kw = np.asarray(kernel).shape
    steps = []
    for row, col in np.ndindex(output.shape):
        top, left = row * stride, col * stride
        patch = padded[top:top + kh, left:left + kw]
        products = patch * kernel
        steps.append({"row": row, "col": col, "top": top, "left": left,
                      "patch": patch.copy(), "products": products,
                      "sum": float(products.sum())})
    return padded, output, steps


def shift_image(image, dx=0, dy=0):
    """Zero-filled translation; positive dx moves right, positive dy down. No wrap."""
    if isinstance(dx, bool) or not isinstance(dx, (int, np.integer)):
        raise ValueError("dx must be an integer")
    if isinstance(dy, bool) or not isinstance(dy, (int, np.integer)):
        raise ValueError("dy must be an integer")
    image = np.asarray(image)
    if image.ndim < 2 or not np.isfinite(image).all():
        raise ValueError("image must have finite values and at least two dimensions")
    height, width = image.shape[-2:]
    result = np.zeros_like(image)
    if abs(dx) >= width or abs(dy) >= height:
        return result
    target_rows = slice(max(0, dy), min(height, height + dy))
    target_cols = slice(max(0, dx), min(width, width + dx))
    source_rows = slice(max(0, -dy), min(height, height - dy))
    source_cols = slice(max(0, -dx), min(width, width - dx))
    result[..., target_rows, target_cols] = image[..., source_rows, source_cols]
    return result


def equivariance(image, kernel, dx):
    """Compare C(Tx) with T(Cx) only where both stride-1 windows are interior."""
    original = correlate2d(image, kernel, padding=1)
    shifted_image = shift_image(image, dx)
    response = correlate2d(shifted_image, kernel, padding=1)
    aligned = shift_image(original, dx)
    rows, cols = np.indices(original.shape)
    valid = ((rows >= 1) & (rows < original.shape[0] - 1)
             & (cols >= 1) & (cols < original.shape[1] - 1)
             & (cols - dx >= 1) & (cols - dx < original.shape[1] - 1))
    if np.asarray(kernel).shape != (3, 3) or not valid.any():
        raise ValueError("equivariance demo requires a 3x3 kernel and nonempty interior overlap")
    difference = np.where(valid, response - aligned, np.nan)
    return {"input": np.asarray(image), "shifted": shifted_image,
            "original_map": original, "shifted_map": response, "aligned_map": aligned,
            "difference": difference, "valid": valid,
            "max_error": float(np.max(np.abs(difference[valid])))}


def rf_table(size=20):
    size = integer(size, "size", 8)
    result = []
    for index, (name, kernel, stride, padding) in enumerate(RF_LAYERS):
        if index:
            size = output_size(size, kernel, stride, padding)
        span, jump = receptive_field([(k, s, 1) for _, k, s, _ in RF_LAYERS[1:index + 1]])
        result.append({"layer": name, "size": size, "span": span, "jump": jump})
    return result


def receptive_mask(layer, row, col, size=20):
    """Exact structural support, tracing indices backward and clipping per layer."""
    layer = integer(layer, "layer", 0)
    if layer >= len(RF_LAYERS):
        raise ValueError("layer is outside the receptive-field demonstration")
    table = rf_table(size)
    row, col = integer(row, "row", 0), integer(col, "col", 0)
    if max(row, col) >= table[layer]["size"]:
        raise ValueError("neuron coordinates are outside this layer")
    mask = np.zeros((table[layer]["size"],) * 2, dtype=bool)
    mask[row, col] = True
    for index in range(layer, 0, -1):
        _, kernel, stride, padding = RF_LAYERS[index]
        previous_size = table[index - 1]["size"]
        previous = np.zeros((previous_size, previous_size), dtype=bool)
        for out_row, out_col in np.argwhere(mask):
            for u, v in np.ndindex(kernel, kernel):
                source_row, source_col = out_row * stride + u - padding, out_col * stride + v - padding
                if 0 <= source_row < previous_size and 0 <= source_col < previous_size:
                    previous[source_row, source_col] = True
        mask = previous
    return mask


def rgb_demo(row=2, col=2, output_channel=0):
    """Two learned-style (fixed illustrative) filters across all three channels."""
    image = scene(8)
    channels = np.stack([image, np.flipud(image), np.fliplr(image)])
    weights = np.stack([
        np.stack([FILTERS["Vertical edge"] / 3, FILTERS["Blur"], -FILTERS["Identity"]]),
        np.stack([FILTERS["Horizontal edge"] / 3, -FILTERS["Blur"], FILTERS["Identity"]]),
    ])
    bias = np.array([0.2, -0.1])
    row, col = integer(row, "row", 0), integer(col, "col", 0)
    output_channel = integer(output_channel, "output_channel", 0)
    if row >= 6 or col >= 6 or output_channel >= 2:
        raise ValueError("RGB demo needs row/col in 0..5 and output_channel in 0..1")
    output = np.stack([sum(correlate2d(channels[c], weights[o, c]) for c in range(3)) + bias[o]
                       for o in range(2)])
    patch = channels[:, row:row + 3, col:col + 3]
    contributions = (patch * weights[output_channel]).sum(axis=(1, 2))
    return {"input": channels, "weights": weights, "bias": bias, "output": output,
            "contributions": contributions, "row": row, "col": col, "channel": output_channel}


def architecture_table():
    """Shapes/parameters for the actual 32x32 training CNN, not the 16x16 study."""
    return [
        ("Input", (1, 32, 32), 0, "1"),
        ("Conv 1", (8, 32, 32), 80, "3"),
        ("ReLU 1", (8, 32, 32), 0, "3"),
        ("MaxPool 1", (8, 16, 16), 0, "4"),
        ("Conv 2", (16, 16, 16), 1168, "8"),
        ("ReLU 2", (16, 16, 16), 0, "8"),
        ("MaxPool 2", (16, 8, 8), 0, "10"),
        ("Global mean", (16, 1, 1), 0, "whole image union"),
        ("Linear logits", (2, 1, 1), 34, "whole image union"),
    ]
