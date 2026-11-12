"""Generate the Day 50 visual learning sequence or open its native viewer."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
import numpy as np
import PIL
import scipy
import sklearn

from visual_core import (
    Design, analyze_teams, population_truth, record_analysis, repeated_study,
    repeated_summary, separation_sweep, simulate_teams,
)


def generate(output, seed=50, full=False, only=None):
    matplotlib.use("Agg")
    from visualizations import (
        draw_balance, draw_confounding, draw_estimators, draw_extremes, draw_hidden,
        draw_ipw, draw_matching, draw_overlap, draw_surface, save_chapter, style,
    )
    style()
    output = Path(output)
    if (output/"run.json").exists():
        raise ValueError("This directory already contains run.json; choose a new output directory.")
    selected = set(range(1, 10)) if only is None else set(only)
    if not selected or not selected.issubset(range(1, 10)):
        raise ValueError("Select chapter numbers 1 through 9.")
    n, frames, repetitions, dpi = (2500, 56, 120, 110) if full else (600, 18, 24, 85)
    started = time.perf_counter()
    baseline = analyze_teams(simulate_teams(n, seed))
    sweep = separation_sweep(n, seed, frames) if selected & {1, 3, 7} else None
    rows = repeated_study(n, seed, repetitions) if 8 in selected else []
    matching = analyze_teams(simulate_teams(100 if full else 60, seed)) if 4 in selected else None
    hidden = analyze_teams(simulate_teams(n, seed, Design(hidden=1.5))) if 9 in selected else None
    structural = simulate_teams(n, seed, Design(structural=True)) if 3 in selected else None
    artifacts = []

    def save(number, name, draw, states):
        if number in selected:
            print(f"Rendering {name} ({len(states)} states)...", flush=True)
            artifacts.extend(save_chapter(output, name, draw, states, dpi=dpi, fps=6 if full else 4))

    save(1, "01_confounding", lambda fig, i: draw_confounding(fig, sweep[i]), list(range(frames)))
    save(2, "02_propensity_surface", lambda fig, angle: draw_surface(fig, baseline, angle),
         np.linspace(-55, 305, 72 if full else 24, endpoint=False).tolist())
    save(3, "03_overlap", lambda fig, i: draw_overlap(fig, sweep[i], structural), list(range(frames)))
    save(4, "04_matching", lambda fig, i: draw_matching(fig, matching, i),
         list(range(len(matching["full_matches"].treated)+1)) if matching else [0])
    save(5, "05_ipw_ate", lambda fig, fraction: draw_ipw(fig, baseline, fraction),
         np.linspace(0, 1, frames).tolist())
    save(5, "05_ipw_att", lambda fig, _: draw_ipw(fig, baseline, 1, "ATT"), [0])
    save(6, "06_balance", lambda fig, _: draw_balance(fig, baseline), [0])
    save(7, "07_extreme_weights", lambda fig, i: draw_extremes(fig, sweep, i), list(range(frames)))
    save(8, "08_estimators", lambda fig, _: draw_estimators(fig, rows), [0])
    save(9, "09_hidden_confounding", lambda fig, _: draw_hidden(fig, hidden), [0])
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "utc_time": datetime.now(timezone.utc).isoformat(), "mode": "full" if full else "quick",
        "seed": seed, "n": n, "frames_per_sweep": frames, "repetitions_per_scenario": repetitions,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__, "pillow": PIL.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "command": sys.argv, "hypothesis": "Observed adjustment may reduce selection error; poor overlap can concentrate information.",
        "interpretation_status": "author review required",
        "limitations": "Synthetic generator; point estimates only; sweep uses one fixed random tape; MCSE is simulation precision.",
        "baseline": record_analysis(baseline),
        "matching_illustration": record_analysis(matching) if matching else None,
        "hidden_illustration": record_analysis(hidden) if hidden else None,
        "sweep": [record_analysis(item) for item in sweep] if sweep else [],
        "repeated": rows, "repeated_summary": repeated_summary(rows) if rows else [],
        "population_truth": {"constant": population_truth(Design()),
                             "heterogeneous": population_truth(Design(heterogeneity=3))},
        "artifacts": [str(p.relative_to(output)) for p in artifacts],
        "elapsed_seconds": time.perf_counter()-started,
    }
    (output/"run.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Generated {len(artifacts)} assets in {output.resolve()} ({record['elapsed_seconds']:.1f}s)", flush=True)
    return record


def interactive(seed=50, output=None, smoke_test=False):
    import matplotlib.pyplot as plt
    from visualizations import make_dashboard
    fig, controls, toggle, reset, state = make_dashboard(seed=seed)
    if not plt.get_backend().lower().startswith(("tk", "qt", "wx", "macosx")):
        plt.close(fig)
        raise RuntimeError("Interactive mode needs a GUI backend, e.g. TkAgg with tkinter; generation uses Agg independently.")
    timers = []
    if smoke_test:
        from matplotlib.backend_bases import MouseEvent

        def exercise():
            try:
                slider = controls["Separation"]
                x, y = slider.ax.transData.transform((2.2, .5))
                for kind in ("button_press_event", "button_release_event"):
                    event = MouseEvent(kind, fig.canvas, x, y, button=1)
                    fig.canvas.callbacks.process(kind, event)
                assert abs(slider.val-2.2) < .05
                toggle.set_active(0)
                controls["Caliper"].set_val(0)
                fig.axes[0].view_init(elev=32, azim=25)
                fig.canvas.draw()
                if output:
                    output.mkdir(parents=True, exist_ok=True)
                    fig.savefig(output/"interactive_smoke.png", dpi=120)
                print(f"Native GUI smoke: mouse slider, heterogeneity, zero caliper and 3D redraw; updates={state['updates']}", flush=True)
            except Exception as exc:
                state["smoke_error"] = exc
            finally:
                plt.close(fig)

        timer = fig.canvas.new_timer(interval=1000)
        timer.single_shot = True
        timer.add_callback(exercise)
        timer.start()
        timers.append(timer)
    plt.show()
    if "smoke_error" in state:
        raise state["smoke_error"]
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="600 units, 18 sweep frames, 24 repetitions (default)")
    group.add_argument("--full", action="store_true", help="2500 units, 56 sweep frames, 120 repetitions")
    group.add_argument("--interactive", action="store_true", help="Native Matplotlib sliders and mouse-rotatable 3D")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--only", type=int, nargs="+", choices=range(1, 10), help="Generate selected chapters")
    parser.add_argument("--smoke-test", action="store_true", help="Exercise native viewer controls and close automatically")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed > 2**32-1:
        parser.error("seed must be between 0 and 2**32-1.")
    if args.smoke_test and not args.interactive:
        parser.error("--smoke-test requires --interactive.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    mode = "interactive" if args.interactive else "full" if args.full else "quick"
    output = args.output_dir or Path(__file__).resolve().parent/"outputs"/f"{mode}-{stamp}"
    if args.interactive:
        return interactive(args.seed, output, args.smoke_test)
    return generate(output, args.seed, args.full, args.only)


if __name__ == "__main__":
    main()
