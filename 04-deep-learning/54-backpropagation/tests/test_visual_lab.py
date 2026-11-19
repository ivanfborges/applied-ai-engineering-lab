"""Independent numerical identities and real temporary-directory export smoke."""

from pathlib import Path
from html.parser import HTMLParser
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_lab import MODES, build_lab
from visual_math import (
    accumulation, activations, central_difference, clip_global_norm, descent,
    gradient_products, loss_gradient, loss_surface, optional_autograd_check,
    scalar_graph, synthetic_data,
)


@pytest.mark.parametrize("theta", [(.5, .8), (.7, 1.2), (-1.1, .3), (0., 1.), (1., 0.)])
@pytest.mark.parametrize("epsilon", [1e-4, 1e-5, 1e-6])
def test_full_batch_manual_gradients_match_central_differences(theta, epsilon):
    x, y = synthetic_data(19)
    _, analytical = loss_gradient(theta, x, y)
    np.testing.assert_allclose(analytical, central_difference(theta, x, y, epsilon), atol=1e-8, rtol=2e-5)


def test_surface_orientation_and_values():
    x, y = synthetic_data(13)
    w1, w2 = np.meshgrid([-.7, .2, 1.1], [.4, -.9])
    surface = loss_surface(w1, w2, x, y)
    assert surface.shape == (2, 3)
    for i, j in np.ndindex(surface.shape):
        expected = sum(.5 * (w2[i, j] * np.tanh(w1[i, j] * a) - target)**2 for a, target in zip(x, y)) / len(x)
        assert surface[i, j] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("rate", [.001, .01, .1, 1.])
def test_every_trajectory_point_and_simultaneous_update(rate):
    x, y = synthetic_data()
    trace = descent(x, y, learning_rate=rate)
    assert trace["status"] == "completed"
    assert len(trace["loss"]) == 181
    np.testing.assert_array_equal(trace["theta"][0], [.5, .8])
    for position, value, gradient in zip(trace["theta"], trace["loss"], trace["gradient"]):
        assert value == pytest.approx(float(loss_surface(*position, x, y)), abs=1e-14)
        np.testing.assert_allclose(gradient, loss_gradient(position, x, y)[1])
    np.testing.assert_allclose(trace["theta"][1:], trace["theta"][:-1] - rate * trace["gradient"][:-1])


def test_scalar_graph_includes_bias_and_all_chain_factors():
    s = scalar_graph()
    expected = np.array([(s["prediction"]-s["y"]) * s["w2"] * (1-s["h"]**2) * s["x"],
                         (s["prediction"]-s["y"]) * s["w2"] * (1-s["h"]**2),
                         (s["prediction"]-s["y"]) * s["h"]])
    np.testing.assert_allclose(s["gradient"], expected)
    assert np.prod(s["local"]) == pytest.approx(expected[0])
    for i, name in enumerate(("w1", "b", "w2")):
        plus = scalar_graph(**{name: s[name] + 1e-5})["loss"]
        minus = scalar_graph(**{name: s[name] - 1e-5})["loss"]
        assert expected[i] == pytest.approx((plus-minus)/2e-5, abs=1e-9)
    updated = np.array([s["w1"], s["b"], s["w2"]]) - .2 * s["gradient"]
    assert scalar_graph(w1=updated[0], b=updated[1], w2=updated[2])["loss"] < s["loss"]


def test_scalar_chain_products_are_exact():
    for factor in (.5, 1., 1.5):
        expected, value = [1.], 1.
        for _ in range(16):
            value *= factor; expected.append(value)
        np.testing.assert_array_equal(gradient_products(factor), expected)


def test_activation_derivatives_and_conventions():
    z = np.array([-2., -.3, .3, 2.])
    for name, (value, derivative) in activations(z).items():
        numerical = (activations(z+1e-6)[name][0] - activations(z-1e-6)[name][0]) / 2e-6
        np.testing.assert_allclose(derivative, numerical, atol=1e-9)
    assert activations(0)["ReLU"][1] == 0
    for pair in activations(np.array([-1000., 1000.])).values():
        assert all(np.isfinite(array).all() for array in pair)


def test_global_clipping_direction_threshold_and_no_mutation():
    gradient = np.array([3., 4.])
    clipped = clip_global_norm(gradient, 2.)
    np.testing.assert_allclose(clipped, [1.2, 1.6])
    assert np.linalg.norm(clipped) == pytest.approx(2.)
    np.testing.assert_array_equal(gradient, [3., 4.])
    np.testing.assert_array_equal(clip_global_norm([0., 0.]), [0., 0.])
    np.testing.assert_array_equal(clip_global_norm([.1, .2]), [.1, .2])


def test_unequal_microbatch_mean_and_batch_repetition():
    x, y = synthetic_data()
    record = accumulation(x, y)
    np.testing.assert_allclose(record["full"], record["weighted"], atol=1e-15)
    assert not np.allclose(record["full"], record["unweighted"])
    assert not np.allclose(record["full"], record["sum"])
    repeated = loss_gradient((.5, .8), np.tile(x, 2), np.tile(y, 2))[1]
    np.testing.assert_allclose(record["full"], repeated)


@pytest.mark.parametrize("x,y,theta", [([], [], [1., 2.]), ([1], [[1]], [1., 2.]),
                                        ([np.nan], [1], [1., 2.]), ([1], [1], [1.]),
                                        ([1], [1], [np.inf, 1.])])
def test_invalid_data_and_parameters(x, y, theta):
    with pytest.raises(ValueError):
        loss_gradient(theta, x, y)


def test_invalid_controls(tmp_path):
    x, y = synthetic_data()
    for epsilon in (0, -1, np.inf):
        with pytest.raises(ValueError):
            central_difference([1., 1.], x, y, epsilon)
    for kwargs in ({"frames": 1}, {"grid": 1}, {"steps": 0}, {"mode": "unknown"}):
        with pytest.raises(ValueError):
            build_lab(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        accumulation(x, y, sizes=(3, 4))


def test_list_without_plot_import_and_optional_autograd():
    script = Path(__file__).resolve().parents[1] / "visual_lab.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--list"], capture_output=True, text=True, check=True)
    assert all(name in result.stdout for name in MODES)
    x, y = synthetic_data()
    report = optional_autograd_check(x, y)
    assert report["status"] in ("compared", "not performed: PyTorch is not installed")


def test_cli_defaults_to_distinct_ignored_runs_without_touching_previews(monkeypatch):
    import visual_lab
    outputs = []
    monkeypatch.setattr(visual_lab, "build_lab", lambda output, **kwargs: outputs.append(output))
    visual_lab.main(["--all"])
    visual_lab.main(["--only", "gradient_flow"])
    topic = Path(visual_lab.__file__).resolve().parent
    assert outputs[0] != outputs[1]
    for output in outputs:
        assert output.parent == topic / "outputs" / "visual-lab"
        assert output.name.startswith("run-")
        root = topic.parents[1]
        result = subprocess.run(["git", "check-ignore", "--", output.relative_to(root).as_posix()],
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0
        assert not output.exists()  # The fake builder must not create or render anything.


def test_existing_run_is_rejected_without_changing_any_file(tmp_path, monkeypatch):
    import visual_render
    sentinel = tmp_path / "computational_graph.gif"
    sentinel.write_bytes(b"historical preview")
    monkeypatch.setattr(visual_render.Renderer, "computational_graph",
                        lambda self: pytest.fail("a nonempty output must fail before rendering"))
    with pytest.raises(FileExistsError, match="new or empty"):
        build_lab(tmp_path, mode="computational_graph")
    assert sentinel.read_bytes() == b"historical preview"
    assert sorted(p.name for p in tmp_path.iterdir()) == [sentinel.name]


@pytest.mark.parametrize("kind", ["png", "gif", "html"])
def test_renderer_refuses_replacing_a_file_created_after_initialization(tmp_path, kind):
    import matplotlib.pyplot as plt
    from visual_render import Renderer
    renderer = Renderer(tmp_path, frames=2, grid=21)
    name = "loss_landscape_3d.html" if kind == "html" else "existing." + kind
    target = tmp_path / name
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        if kind == "html":
            x, y = synthetic_data()
            traces = [descent(x, y, steps=1, learning_rate=rate) for rate in (.001, .01, .1, 1.)]
            renderer.loss_landscape(x, y, traces)
        else:
            figure = plt.figure()
            if kind == "png":
                renderer.png(figure, name)
            else:
                renderer.gif(figure, lambda frame: None, name)
    assert target.read_bytes() == b"preserve"
    assert plt.get_fignums() == []


@pytest.mark.rendering
def test_all_real_exports_in_temporary_directory(tmp_path):
    from PIL import Image
    import matplotlib.pyplot as plt
    report = build_lab(tmp_path, frames=18, grid=21, steps=20)
    assert len(report["outputs"]) == 14
    assert len(list(tmp_path.glob("run-*.json"))) == 1
    assert plt.get_fignums() == []
    assert len([record for record in report["outputs"] if record["file"].endswith(".gif")]) == 6
    for record in report["outputs"]:
        assert record["bytes"] > 1000
        if record["file"].endswith(".gif"):
            with Image.open(tmp_path / record["file"]) as image:
                image.seek(0); start = np.asarray(image.convert("RGB"))
                image.seek(image.n_frames-1); end = np.asarray(image.convert("RGB"))
                assert not np.array_equal(start, end)
    html = (tmp_path / "loss_landscape_3d.html").read_text(encoding="utf-8")
    assert 'id="reset-camera"' in html
    external_scripts = []
    class ScriptParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "script" and dict(attrs).get("src"):
                external_scripts.append(dict(attrs)["src"])
    ScriptParser().feed(html)
    assert external_scripts == []
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="new or empty"):
        build_lab(tmp_path, frames=18, grid=21, steps=20)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original
