"""Visual-only contracts; no PyTorch import, downloads or writes to curated assets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from from_scratch import METHODS, Optimizer
from visual_math import (Objective, compare, constant_gradient, frame_steps, grid,
                         learning_rates, reference_rates, simulate)


@pytest.mark.parametrize("objective", [Objective(), Objective(angle=71, center=(1, -2)), Objective("rosenbrock")])
def test_objective_gradient_finite_differences_and_known_optimum(objective):
    point = np.array([-.7, .4])
    numeric = [(objective.value(point + axis*1e-5) - objective.value(point - axis*1e-5))/2e-5
               for axis in np.eye(2)]
    np.testing.assert_allclose(objective.gradient(point), numeric, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(objective.gradient(objective.optimum), [0, 0], atol=1e-14)
    assert objective.value(objective.optimum) == pytest.approx(0)


def test_quadratic_rotation_hessian_and_vectorization():
    objective = Objective(a=100, b=2, angle=37)
    hessian = objective.rotation.T @ np.diag([100, 2]) @ objective.rotation
    np.testing.assert_allclose(np.linalg.eigvalsh(hessian), [2, 100])
    points = np.arange(12.).reshape(2, 3, 2)/10
    np.testing.assert_allclose(objective.value(points), .5*np.einsum('...i,ij,...j->...', points, hessian, points))


@pytest.mark.parametrize("bad", [[1j, 0], [np.nan, 1], [True, False], ["1", "2"], [], [1]])
def test_invalid_objective_inputs_are_explicit(bad):
    with pytest.raises(ValueError):
        Objective().value(bad)


def test_visual_workflow_does_not_import_pytorch():
    import subprocess
    result = subprocess.run([sys.executable, "-c",
                             "import sys; sys.path.insert(0,sys.argv[1]); import visualizations; assert 'torch' not in sys.modules",
                             str(Path(__file__).resolve().parents[1])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("decay", [0., .2])
def test_recorded_components_reconstruct_actual_steps_and_moments(method, decay):
    objective = Objective(center=(1, -.5))
    trace = simulate(objective, method, .02, steps=12, weight_decay=decay, noise=.2)
    np.testing.assert_allclose(trace.losses, objective.value(trace.points))
    h = trace.history
    np.testing.assert_allclose(np.diff(trace.points, axis=0), h["total_delta"], atol=1e-15)
    np.testing.assert_allclose(h["total_delta"], h["adaptive_delta"]+h["decay_delta"], atol=1e-15)
    expected_input = h["gradient"] if method == "adamw" else h["gradient"]+decay*trace.points[:-1]
    np.testing.assert_allclose(h["optimizer_input"], expected_input)
    np.testing.assert_allclose(h["decay_delta"], -.02*decay*trace.points[:-1] if method == "adamw" else 0, atol=1e-15)
    if method == "momentum":
        np.testing.assert_allclose(h["first"], .9*h["previous_first"]+expected_input)
    if method == "rmsprop":
        previous = np.vstack(([0, 0], h["second"][:-1]))
        np.testing.assert_allclose(h["second"], .99*previous+.01*expected_input**2)
        np.testing.assert_allclose(h["scaling"], .02/(np.sqrt(h["second"])+1e-8))


def test_constant_gradient_bias_correction_first_twenty_steps():
    history = constant_gradient()
    np.testing.assert_allclose(history["first_hat"], np.tile(history["gradient"], (20, 1)), atol=1e-14)
    np.testing.assert_allclose(history["second_hat"], np.tile(history["gradient"]**2, (20, 1)), atol=1e-14)
    assert np.all(np.abs(history["first"]) < np.abs(history["gradient"]))
    assert np.all(history["second"] < history["gradient"]**2)


@pytest.mark.parametrize("name", ["quadratic", "rosenbrock"])
def test_reference_traces_are_finite_and_no_decay_pair_is_identical(name):
    objective = Objective(name)
    traces = compare(objective, steps=80)
    assert all(t.status == "completed" and t.steps == 80 for t in traces)
    assert all(np.all(np.isfinite(t.points)) and np.all(np.isfinite(t.losses)) for t in traces)
    np.testing.assert_array_equal(traces[3].points, traces[4].points)
    assert all(np.array_equal(t.points[0], [-2, 1.5]) for t in traces)


def test_zero_decay_control_and_nonzero_decay_difference():
    objective = Objective(center=(1, -.5))
    zero = compare(objective, methods=("adam", "adamw"), steps=40)
    np.testing.assert_array_equal(zero[0].points, zero[1].points)
    pair = compare(objective, methods=("adam", "adamw"), steps=40, weight_decay=.1)
    assert not np.allclose(pair[0].points, pair[1].points)


def test_noise_is_reproducible_common_random_numbers_not_literal_minibatches():
    objective = Objective()
    low = simulate(objective, "sgd", .02, noise=.5, steps=20)
    high = simulate(objective, "adam", .02, noise=2., steps=20)
    np.testing.assert_allclose((low.history["gradient"]-low.history["exact_gradient"])/.5,
                               (high.history["gradient"]-high.history["exact_gradient"])/2, atol=1e-13)
    repeated = simulate(objective, "sgd", .02, noise=.5, steps=20)
    np.testing.assert_array_equal(repeated.points, low.points)


def test_divergence_is_detected_without_clipping_or_losing_finite_history():
    objective = Objective()
    trace = simulate(objective, "sgd", .2, steps=80, loss_limit=1e7)
    assert "divergence guard" in trace.status
    assert trace.steps < 80 and trace.losses[-1] > 1e7
    np.testing.assert_allclose(trace.points[1], trace.points[0]-.2*objective.gradient(trace.points[0]))
    for label, rate in learning_rates(objective, "sgd").items():
        candidate = simulate(objective, "sgd", rate, steps=80)
        if label == "reference":
            assert candidate.losses[-1] < candidate.losses[0]
        if label == "large":
            assert candidate.losses[-1] > candidate.losses[0]


def test_diagnostic_copies_and_invalid_step_are_atomic():
    optimizer = Optimizer([1., -1.], "adamw", .01, weight_decay=.1)
    assert optimizer.last_step is None
    optimizer.step([.5, .2])
    before = optimizer.last_step
    altered = optimizer.last_step
    altered["total_delta"][:] = 100
    with pytest.raises(ValueError):
        optimizer.step([np.inf, 0])
    for key in before:
        np.testing.assert_array_equal(optimizer.last_step[key], before[key])


@pytest.mark.parametrize("options", [{"steps": 0}, {"steps": 2001}, {"steps": True},
                                      {"noise": -1}, {"seed": -1}, {"start": [1]}])
def test_bad_simulation_configuration(options):
    with pytest.raises(ValueError):
        simulate(Objective(), "sgd", .01, **options)


def test_grid_matches_exact_objective_and_frames_include_endpoints():
    objective = Objective()
    traces = compare(objective, steps=9)
    x, y, z = grid(objective, traces, 21)
    np.testing.assert_array_equal(z, objective.value(np.stack([x, y], axis=-1)))
    assert frame_steps(9, 5).tolist() == [0, 2, 4, 6, 9]


def test_rate_focus_view_keeps_the_actual_exiting_segment_and_full_loss_history():
    import matplotlib.pyplot as plt
    from visual_render import rate_figure
    objective = Objective()
    traces = [simulate(objective, "sgd", rate, steps=10) for rate in [.002, .04, .055]]
    figure, update = rate_figure(objective, traces, ["small", "reference", "large"])
    update(10)
    focus, loss_axis = figure.axes
    # Two marker entries precede line/current entries for each of three paths.
    plotted = focus.lines[6]
    np.testing.assert_allclose(plotted.get_xdata(), traces[2].points[:2, 0])
    np.testing.assert_allclose(plotted.get_ydata(), traces[2].points[:2, 1])
    np.testing.assert_allclose(loss_axis.lines[2].get_ydata(), traces[2].losses)
    plt.close(figure)


@pytest.mark.parametrize("transformed", [False, True])
def test_surface_points_and_every_animation_frame_match_the_objective(transformed):
    from visual_render import surface_figure
    objective = Objective()
    traces = compare(objective, steps=10)
    figure = surface_figure(objective, traces, resolution=15, frames=5, log_surface=transformed)
    transform = np.log1p if transformed else lambda x: x
    surface = figure.data[0]
    x, y = np.array(surface.x), np.array(surface.y)
    np.testing.assert_allclose(surface.z, transform(objective.value(np.stack([x, y], axis=-1))))
    for frame in figure.frames:
        step = int(frame.name)
        assert len(frame.traces) == 10
        for i, trace in enumerate(traces):
            path, current = frame.data[2*i:2*i+2]
            np.testing.assert_allclose(path.x, trace.points[:step+1, 0])
            np.testing.assert_allclose(path.z, transform(trace.losses[:step+1]))
            np.testing.assert_allclose(current.z, [transform(objective.value(trace.points[step]))])
            assert path.visible is None and current.visible is None
            assert current.customdata[-1][0] == step
    for button in figure.layout.updatemenus[0].buttons[1:]:
        visible = button.args[1]
        assert all(visible[:3]) and sum(visible[3:]) == 3


@pytest.mark.rendering
def test_real_offline_html_export_has_embedded_runtime(tmp_path):
    import re
    from visual_render import export_surface, surface_figure
    figure = surface_figure(Objective(), compare(Objective(), steps=3), resolution=11, frames=3)
    path = export_surface(figure, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html and "Plotly.addFrames" in html
    assert not re.search(r'<script[^>]+src=[\"\']https?://', html)
    assert "precomputed endpoints" in html and "Recorded update" in html


@pytest.mark.rendering
@pytest.mark.parametrize("scene", ["trajectories", "momentum", "rmsprop", "adam", "adamw", "learning-rate", "noise"])
def test_real_png_and_multiframe_pillow_export_in_temporary_directory(tmp_path, scene):
    from PIL import Image
    from visualizations import generate, parser
    args = parser().parse_args(["--scene", scene, "--steps", "3", "--quality", "quick", "--output", str(tmp_path)])
    manifest = generate(args)
    pngs = list(tmp_path.glob("*.png"))
    gifs = list(tmp_path.glob("*.gif"))
    assert len(gifs) == 1 and len(pngs) == (2 if scene == "adam" else 1)
    with Image.open(gifs[0]) as image:
        assert image.n_frames == 4
        assert image.info["loop"] == 0
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] > 0
            frames.append(np.asarray(image.convert("RGB")))
        assert not np.array_equal(frames[0], frames[-1])
    for path in pngs:
        with Image.open(path) as image:
            image.verify()
    assert manifest["interpretation_status"] == "author review required"
