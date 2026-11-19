"""Opt-in TkAgg widget smoke. Run locally; excluded from the default headless test suite."""

from pathlib import Path
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent
    import numpy as np
    import torch
    from visual_core import LabExperiments
    from visual_figures import latent_view, style, threshold_view

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    style()
    lab = LabExperiments()
    lab.epochs, lab.frames = 3, 6
    figures = []
    try:
        fig, threshold = threshold_view(lab, True)
        figures.append(fig)
        fig.canvas.draw()
        window = fig.canvas.manager.window
        window.withdraw()
        window.update()
        threshold["controls"]["threshold"].set_val(threshold["max_score"])
        assert threshold["state"]["metrics"]["alerts"] == 0
        button = threshold["controls"]["0.975"]
        x, y = button.ax.transAxes.transform((.5, .5))
        for event_name in ("button_press_event", "button_release_event"):
            fig.canvas.callbacks.process(event_name, MouseEvent(event_name, fig.canvas, x, y, button=1))
        assert threshold["state"]["threshold"] == lab.base()["thresholds"]["0.975"]
        expected = lab.base()["scores"] > threshold["state"]["threshold"]
        assert expected.sum() == threshold["state"]["metrics"]["alerts"]
        fig.canvas.draw()
        window.update()
        print("TkAgg threshold: slider, synthetic mouse button events, metrics and canvas redraw passed")

        fig, latent = latent_view(lab, True)
        figures.append(fig)
        fig.canvas.manager.window.withdraw()
        latent["controls"]["camera"].set_val(40)
        latent["controls"]["sample"].set_val(170)
        fig.canvas.draw()
        fig.canvas.manager.window.update()
        assert latent["state"] == {"azimuth": 40., "sample": 170}
        assert all(ax.azim == 40 for ax in fig.axes if ax.name == "3d")
        assert np.isfinite(lab.geometry()["latent"]).all()
        print("TkAgg latent: camera/sample sliders, real 3D views and canvas redraw passed")
        print("Programmatic GUI smoke complete. Human mouse/keyboard exploration and other backends remain separate checks.")
    finally:
        for fig in figures:
            plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
