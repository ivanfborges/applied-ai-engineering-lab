"""Self-contained offline Plotly pages with embedded numerical controls."""

from html import escape
import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from visual_core import json_safe


PAGES = {
    "geometry": [("ffn_geometric_transformations.html", "geometry")],
    "activations": [("activation_distribution_comparison.html", "activations")],
    "residual": [("residual_vector_addition.html", "residual"), ("residual_stream_trajectory.html", "residual-stream")],
    "gradients": [("analytical_gradient_propagation.html", "scalar-gradients")],
    "normalization": [("layer_normalization_geometry.html", "layernorm")],
    "normalization-comparison": [("normalization_methods_comparison.html", "normalization"),
                                  ("layernorm_vs_rmsnorm_geometry.html", "normalization-geometry")],
    "pre-post-norm": [("normalization_depth_comparison.html", "pre-post-norm")],
    "swiglu": [("gated_feature_transformations.html", "swiglu")],
    "complexity": [("transformer_parameter_explorer.html", "parameters"),
                    ("attention_vs_ffn_complexity.html", "complexity")],
}

EXPLANATIONS = {
    "ffn": ("How does one shared FFN process four separate tokens?",
            "Actual 8→32→8 arrays; read values and follow the highlighted token. Each row uses the same matrices; no cross-token operation occurs. This separates feature processing from attention. Untrained random weights do not establish semantic quality."),
    "geometry": ("What changes when the same affine projections contain different activations?",
                 "Track stable sample IDs and cluster colors. Hidden stages display coordinates 0–2 of 12; input/output display their original 3D coordinates. These are different feature spaces: interpolation is a presentation transition, not a physical trajectory. No PCA or separation improvement claim."),
    "activations": ("How do activation slopes and input distributions affect local signals?",
                    "Identity, ReLU, tanh-approximate GELU and SiLU share identical inputs. ReLU derivative at zero is defined as zero. Change mean/scale and compare signed outputs; smoothness and negative outputs do not by themselves predict learning quality."),
    "residual": ("How does adding a branch differ from replacing a state?",
                 "Compute y=x+alpha F(x) in original 3D coordinates. The translated branch starts at the tip of x. Alpha changes direction and magnitude. Additive pathways influence optimization but do not guarantee invertibility or stability."),
    "residual-stream": ("How does a residual stream evolve through multiple layers?",
                        "Both trajectories use the same layer parameters and initial state; nonlinear branch inputs subsequently differ. Alpha scales only the residual branch. Each plotted vertex is computed; no identity-preservation guarantee."),
    "scalar-gradients": ("When can identity paths attenuate, cancel or amplify gradients?",
                         "Analytical scalar products: |a|^L and |1+alpha a|^L. Exact zeros are reported separately and omitted from the log axis. Overflow/underflow are explicit float64 statuses. This is not a measured Transformer or a singular-value analysis."),
    "gradients": ("How do actual paired networks propagate a random-target gradient?",
                  "PyTorch tanh layers share initial weights, input and mean-MSE target. Compare L2, RMS, relative gradients and activation scale. Seed bands are min/max, not confidence intervals. No normalization or optimizer steps; not a convergence experiment."),
    "layernorm": ("Why does standardization approach a circle in three dimensions?",
                  "Centering is an orthogonal projection onto x+y+z=0. With negligible epsilon, nonconstant normalized vectors have norm sqrt(3), forming a circle in that plane. Gamma/beta may alter this geometry; near-constant vectors collapse when epsilon dominates. No Gaussian distribution is implied."),
    "normalization": ("Which samples and tokens share normalization statistics?",
                      "Input B,T,D is transposed to B,D,T for BatchNorm1d; training reduces B,T per feature. LN/RMS reduce D per token. Change a token or batch member and inspect an unchanged sample. Eval BN uses fixed running statistics, calibrated on the base batch with momentum 0.1; no running update occurs in the browser."),
    "normalization-geometry": ("How do mean centering and RMS rescaling differ geometrically?",
                               "LN lies near the zero-mean plane; RMSNorm rescales without centering. Axes are original three feature coordinates. Affine parameters are disabled for this comparison; normalization alone is not evidence of learned quality."),
    "pre-post-norm": ("How do depth, readout and normalization placement change measured signals?",
                      "Measured four-head PyTorch Transformer stacks, paired initial weights and synthetic random-target MSE. A Pre+final-LN sensitivity condition changes only the readout. Explore actual CSV metrics and seed variation. Missing layers are absent; no interpolation/extrapolation or trained-model ranking."),
    "swiglu": ("How can signed multiplicative gates change a feature?",
               "SiLU(gate) times feature is computed live. Gates are neither probabilities nor restricted to [0,1]; products can amplify, suppress or flip signs. The animation uses independent learned projections at random initialization; approximately matched parameter counts do not imply matched quality."),
    "parameters": ("Where do block parameters and weight bytes go?",
                   "Conventional full-width Q/K/V/output MHA, dense or gated FFN, two affine LayerNorms. Biases are included. FP32/FP16/BF16/ideal INT8 describe weights only, excluding scales, embeddings, activations, gradients, optimizer and cache. No runtime memory or latency measurement."),
    "complexity": ("How do full-sequence and cached one-token arithmetic workloads differ?",
                   "Count matrix-multiplication MACs; 1 MAC corresponds to 2 FLOPs by convention. Full attention=4Td²+2T²d; one cached token=4d²+2Td, where T includes the new token. FFN=(2 or 3)Tdh or (2 or 3)dh. Bias, nonlinearities, norms, softmax and memory movement are excluded. Counts are not latency."),
    "transformer-block": ("Which paths carry real activations and gradients in a complete block?",
                          "Actual 1×4×8 Pre-LN block with two heads and FFN width 32. Forward arrays and backward VJPs follow the implemented graph. At each residual sum the identity and learned-path vectors add; their norms need not add. Schematic arrows encode dependencies, not execution timing. No training."),
}

EXPLANATIONS["normalization-comparison"] = EXPLANATIONS["normalization"]

STYLE = """
body{margin:0;background:#f5f8fc;color:#17283c;font:16px system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:26px}h1{font-size:27px;margin-bottom:10px}
p{line-height:1.55;max-width:1150px}.badge{color:#53647b;font-size:13px;letter-spacing:.06em}
#controls{display:flex;flex-wrap:wrap;gap:16px;background:white;padding:18px;border:1px solid #dbe3ee;border-radius:12px}
label{display:flex;flex-direction:column;gap:7px;font-size:13px;min-width:150px}input,select,button{font:inherit}
button{border:1px solid #b7c7d9;background:#fff;border-radius:6px;padding:8px 12px;cursor:pointer}
#plot,#secondary{height:520px;background:white;margin-top:18px;border-radius:12px}#secondary:empty{display:none}
pre{white-space:pre-wrap;word-break:break-word;background:#e8eef6;padding:16px;border-radius:10px;font-size:13px;max-height:280px;overflow:auto}
a{color:#006ca8}footer{margin-top:22px;font-size:13px;color:#526176}
"""


def write_page(folder, filename, mode, config, data):
    question, explanation = EXPLANATIONS[mode]
    keys = {"geometry": ["geometry"], "activations": ["activation_inputs"],
            "residual": ["residual"], "residual-stream": ["residual"],
            "scalar-gradients": [], "layernorm": ["norm_vectors"],
            "normalization": ["normalization"], "normalization-geometry": ["normalization"],
            "pre-post-norm": ["transformer"], "swiglu": ["swiglu"], "parameters": [], "complexity": []}
    payload = {"mode": mode, "config": config.__dict__, **{key: data[key] for key in keys[mode]}}
    encoded = json.dumps(json_safe(payload), allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    script = Path(__file__).with_name("visual_controls.js").read_text(encoding="utf-8")
    source = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
              f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(question)}</title><style>{STYLE}</style></head>'
              f'<body><main><div class="badge">DAY 70 · SYNTHETIC DATA · AUTHOR REVIEW REQUIRED</div><h1>{escape(question)}</h1>'
              f'<p>{escape(explanation)}</p><section id="controls" aria-label="Experiment controls"></section>'
              '<div id="plot" aria-label="Main visualization"></div><div id="secondary" aria-label="Complementary visualization"></div>'
              '<pre id="metrics" aria-live="polite"></pre><footer>Offline standalone HTML. Drag 3D plots to rotate; scroll to zoom. '
              'Controls recompute numerical values or select measured records. No trained-model or latency claim.</footer></main>'
              f'<script id="lab-data" type="application/json">{encoded}</script><script>{get_plotlyjs()}</script><script>{script}</script></body></html>')
    path = Path(folder) / filename
    path.write_text(source, encoding="utf-8")
    return path
