"""On-demand educational GIF exports, with dated configuration manifests."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy
from scipy.stats import lognorm, norm

from simulation import CONTROL, TREATMENT, uncertainty_path


def _frame(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).convert("P",palette=Image.Palette.ADAPTIVE)


def export_gifs(output_dir, control=CONTROL, treatment=TREATMENT, n=3000, seed=47,
                revenue=2., frames=24, width=900, height=500):
    """Render only on an explicit call. No import-time or dashboard-refresh export."""
    if not 6 <= frames <= 48 or not 400 <= width <= 1200 or not 300 <= height <= 800:
        raise ValueError("Use 6–48 frames and bounded 400–1200 × 300–800 resolution")
    if not 100 <= n <= 20000:
        raise ValueError("GIF n must be between 100 and 20000")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination=Path(output_dir)/f"gif-run-{stamp}"
    destination.mkdir(parents=True,exist_ok=False)
    path=uncertainty_path(control,treatment,n,seed+202,frames)
    truth=treatment.success*treatment.conversion-control.success*control.conversion
    sigmas=np.linspace(.2,1.1,frames)
    max_latency=treatment.median*np.exp(norm.ppf(.995)*sigmas[-1])
    x=np.linspace(.01,max_latency,700)
    gif_paths=[]
    for name in ("latency_percentiles","sample_size_uncertainty"):
        rendered=[]
        with plt.rc_context({"font.size":11,"axes.spines.top":False,"axes.spines.right":False}):
            fig,ax=plt.subplots(figsize=(width/100,height/100),dpi=100)
            try:
                for i in range(frames):
                    ax.clear()
                    if name=="latency_percentiles":
                        sigma=sigmas[i]
                        ax.plot(x,lognorm.pdf(x,sigma,scale=treatment.median),color="#E69F00",lw=2)
                        for q,color,label in ((.5,"#0072B2","median"),(.95,"#CC79A7","p95"),(.99,"#333333","p99")):
                            value=treatment.median*np.exp(norm.ppf(q)*sigma)
                            ax.axvline(value,color=color,ls="--",label=f"{label} = {value:.2f}s")
                        ax.set(xlim=(0,max_latency),ylim=(0,2.2/treatment.median),xlabel="Fully observed first-attempt duration (seconds)",ylabel="Population density",
                               title=f"SYNTHETIC · Fixed median, growing tail · sigma={sigma:.2f}")
                        footer="Analytical lognormal distribution; p95/p99 increase while the median stays fixed."
                    else:
                        end=min(i,len(path)-1)
                        part=path.iloc[:end+1]
                        ax.errorbar(part.n,100*part.lift,yerr=np.vstack((100*(part.lift-part.low),100*(part.high-part.lift))),
                                    fmt="o-",ms=3,color="#E69F00",capsize=3,label="Nested sample + conservative 95% interval")
                        ax.axhline(100*truth,color="#0072B2",ls="--",label=f"Fixed true effect = {100*truth:.2f} pp")
                        ax.axhline(0,color="#777777",ls=":")
                        ax.set(xlim=(0,n*1.03),ylim=(100*path.low.min()-1,100*path.high.max()+1),xlabel="Independent users per arm",
                               ylabel="Conversion lift (percentage points)",title=f"SYNTHETIC · Sample size and uncertainty · n={int(path.iloc[end].n):,}")
                        footer="Individual estimates fluctuate. Repeated inspection is not valid optional-stopping inference."
                    ax.legend(loc="upper right",fontsize=9)
                    ax.grid(alpha=.2)
                    fig.subplots_adjust(left=.1,right=.96,bottom=.19,top=.87)
                    # Replace the footer instead of accumulating text across frames.
                    for text in list(fig.texts):
                        text.remove()
                    fig.text(.5,.04,footer,ha="center",fontsize=9)
                    rendered.append(_frame(fig))
            finally:
                plt.close(fig)
        file=destination/f"{name}.gif"
        rendered[0].save(file,save_all=True,append_images=rendered[1:],duration=180,loop=0,optimize=False)
        gif_paths.append(file)
    record={"run_utc":datetime.now(timezone.utc).isoformat(),"kind":"educational render, not product experiment",
            "hypothesis":"Fixed-median tail spread changes percentiles; growing n usually improves precision without monotone realized intervals.",
            "configuration":{"seed":seed,"users_per_arm":n,"control":asdict(control),"treatment":asdict(treatment),
                             "revenue":revenue,"frames":frames,"resolution":[width,height],
                             "latency_sigma_range":[.2,1.1],"uncertainty_seed":seed+202,
                             "pointwise_confidence":.95},
            "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                        "scipy":scipy.__version__,"matplotlib":matplotlib.__version__,"pillow":Image.__version__},
            "result":{"files":[p.name for p in gif_paths],"bytes":[p.stat().st_size for p in gif_paths],
                      "true_marginal_lift":truth,"sample_path":path.to_dict(orient="records")},
            "interpretation_candidate":"Inspect the fixed median, moving tail quantiles and fluctuating confidence intervals.",
            "interpretation_status":"author review required",
            "limitation":"Synthetic analytical tails and one nested sample; no sequential guarantees or production validation."}
    (destination/"manifest.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return gif_paths


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"outputs")
    parser.add_argument("--users-per-arm",type=int,default=3000)
    parser.add_argument("--seed",type=int,default=47)
    parser.add_argument("--frames",type=int,default=24)
    args=parser.parse_args()
    for file in export_gifs(args.output_dir,n=args.users_per_arm,seed=args.seed,frames=args.frames):
        print(file)
