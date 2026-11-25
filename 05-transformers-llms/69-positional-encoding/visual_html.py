"""Offline Plotly learning pages; local runtime shared by the exported folder."""

import json
from pathlib import Path

from plotly.offline import get_plotlyjs

from visual_core import Config, addition_data, fixture, json_safe, multifrequency_data, permutation_data, precision_data, projected_geometry


PAGES = {
    "sinusoidal": ["sinusoidal_frequencies.html"],
    "construction": ["positional_dimension_comparison.html"],
    "embedding-space": ["positional_embedding_space_3d.html", "original_dimensions_3d.html"],
    "addition": ["position_aware_embeddings_3d.html"],
    "similarity": ["positional_similarity_surface_3d.html"],
    "absolute-relative": ["absolute_vs_relative_interactive.html"],
    "permutation": ["permutation_interactive.html"],
    "rope": ["rope_interactive_geometry.html"],
    "rope-multifrequency": ["rope_attention_contributions.html", "rope_embedding_trajectory_3d.html"],
    "alibi": ["alibi_interactive.html", "alibi_attention_surface_3d.html"],
    "comparison": ["attention_mechanisms_interactive.html"],
    "extrapolation": ["position_extrapolation.html"],
    "rag": ["rag_evidence_position.html"],
}

DESCRIPTIONS = {
    "permutation": ("What changes when content moves but coordinates stay fixed?", "Shared untrained projections; no causal mask/dropout. Undo the row permutation before comparing outputs."),
    "sinusoidal": ("Which frequencies encode a position?", "Exact sine/cosine coordinates. Frequency is radians/token; wavelength is tokens/cycle."),
    "construction": ("How are dimension pairs assembled into a matrix?", "One fixed color scale, -1 to +1. Missing columns are unavailable, not zero-valued encodings."),
    "embedding-space": ("What does the positional trajectory look like?", "PCA is fitted once per width on the whole reference window. Prefix changes reuse that basis. Exact original coordinates are a separate view; projected distances can lose information."),
    "addition": ("How does position move identical content vectors?", "Exact original coordinates and a shared PCA-derived linear basis. No mean subtraction in vector projections: (x+e)B=xB+eB."),
    "similarity": ("Does equal relative distance give equal pure positional similarity?", "Pure sinusoidal cosine, not semantic similarity or trained attention. Similarity may oscillate."),
    "absolute-relative": ("Which positional relationships survive a global shift?", "Absolute table entries are seeded random initialization, not learned language representations. Relative scalar bias is a teaching choice."),
    "rope": ("Why does a common position shift preserve a rotary dot product?", "Fixed original Q/K content; independent 2D rotations. A shared shift preserves relative angle, not absolute vector coordinates."),
    "rope-multifrequency": ("How do independent rotation planes add to a full score?", "Per-pair dot contributions sum before sqrt(d) scaling. The 3D PCA view is a projection, not a single physical 3D rotation."),
    "alibi": ("How does recency bias compete with content relevance?", "Bias is added after content-logit scaling. Gray/blank future cells are masked, not finite penalties. Teaching slopes are supplied explicitly."),
    "comparison": ("How does each position mechanism alter the same attention layer?", "Six mechanisms share content, projections and causal visibility. These untrained synthetic distributions cannot rank model quality."),
    "extrapolation": ("Can a position be represented accurately beyond a hypothetical window?", "128 is a hypothetical training boundary; no model was trained. Sampled query rows avoid quadratic 4096-by-4096 matrices. Interpolation changes coordinates, not proven generalization."),
    "rag": ("How would evidence-position evaluation be designed?", "Prompt assembly only; whitespace-separated display units are not model tokens. No language model, accuracy curve or measured answer quality."),
}


def write_runtime(folder):
    (folder / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    source = Path(__file__).with_name("visual_controls.js")
    (folder / "visual_controls.js").write_bytes(source.read_bytes())


def payload(group, config):
    result = {"group": group, "config": config.__dict__}
    if group in ("permutation", "absolute-relative", "comparison"):
        x, weights, table = fixture(config, 128)
        result.update(x=x, weights=weights, table=table)
    if group == "permutation":
        result["permutation"] = permutation_data(config)
    if group == "embedding-space":
        dimensions = sorted(set([8, 16, 32, 64, 128, config.dim]))
        result["geometry"] = {str(dim): projected_geometry(config, dim) for dim in dimensions}
    if group == "addition":
        result["addition"] = addition_data(config)
    if group == "rope-multifrequency":
        result["rope"] = multifrequency_data(config)
    if group == "extrapolation":
        result["precision"] = precision_data(config)
    return json_safe(result)


def write_page(folder, group, config, filename, pages):
    title, limitation = DESCRIPTIONS[group]
    data = payload(group, config)
    data["initial_view"] = "exact" if filename == "original_dimensions_3d.html" else "pca"
    navigation = "".join(f'<a href="{names[0]}">{name}</a>' for name, names in pages.items())
    source = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Day 69 · __GROUP__</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#182b43;background:#f1f5fa}*{box-sizing:border-box}
body{margin:0}header,main,footer{max-width:1500px;margin:auto;padding:18px 28px}header{background:#12253d;color:white;max-width:none}
header h1{font-size:27px;margin:8px 0}header p{max-width:1200px;line-height:1.5;color:#dce9fa}.tag{font-size:13px;color:#f7bb6d}
nav{display:flex;flex-wrap:wrap;gap:6px;padding:8px 0}nav a{padding:6px 9px;background:#e0e8f3;border-radius:5px;color:#163b6b;font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:12px;background:white;border:1px solid #cbd7e7;border-radius:9px;padding:14px;margin:15px 0}
label{display:flex;flex-direction:column;gap:4px;min-width:150px;font-size:13px;font-weight:600}input,select,button{font:inherit;min-height:32px}input[type=range]{width:170px}
button{background:#185a99;color:white;border:0;border-radius:5px;padding:5px 12px;cursor:pointer}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #d88023;outline-offset:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:14px}.chart{background:white;border:1px solid #d2ddeb;border-radius:8px;min-height:400px;overflow:hidden}
#metrics{white-space:pre-wrap;background:#13283f;color:#eaf3ff;padding:16px;border-radius:8px;font:14px/1.6 Consolas,monospace;margin:12px 0}
#error{color:#a11920;white-space:pre-wrap}.help{font-size:14px;line-height:1.6}footer{font-size:13px;color:#52627a}
@media(max-width:600px){header,main,footer{padding:14px}header h1{font-size:22px}.controls{gap:8px}label{min-width:135px}}
</style><script src="plotly.min.js"></script></head><body>
<header><div class="tag">DAY 69 · POSITIONAL ENCODING · SYNTHETIC / UNTRAINED</div><h1>__TITLE__</h1><p>__LIMIT__</p></header>
<main><a href="index.html">All demonstrations</a><nav>__NAV__</nav>
<p class="help">Change controls to recompute the numerical operations. Drag a 3D plot to orbit; scroll to zoom. Blue: query/content; green: key; orange: position; purple: relative displacement; gray: unavailable.</p>
<div id="controls" class="controls"></div><div id="error" role="alert"></div>
<div id="metrics" aria-live="polite">Computing…</div><div id="charts" class="grid"></div>
</main><footer>Offline local computation · No trained-model quality measurement · Interpretation candidates require author review.</footer>
<script id="lab-data" type="application/json">__DATA__</script><script src="visual_controls.js"></script></body></html>"""
    for old, new in (("__GROUP__", group), ("__TITLE__", title), ("__LIMIT__", limitation),
                     ("__NAV__", navigation), ("__DATA__", json.dumps(data, allow_nan=False).replace("</", "<\\/"))):
        source = source.replace(old, new)
    path = folder / filename
    path.write_text(source, encoding="utf-8")
    return path


def write_gallery(folder, selected, artifacts):
    cards = []
    for group in selected:
        title, limit = DESCRIPTIONS[group]
        links = " · ".join(f'<a href="{name}">{name}</a>' for name in artifacts[group])
        cards.append(f"<section><h2>{group}: {title}</h2><p>{limit}</p><p>{links}</p></section>")
    (folder / "index.html").write_text("""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Day 69 positional learning lab</title>
<style>body{font:16px/1.6 Segoe UI,Arial;background:#f1f5fa;color:#18334e;max-width:1100px;margin:auto;padding:24px}section{background:white;border:1px solid #ccd9e8;border-radius:8px;padding:18px;margin:18px 0}h2{font-size:21px}a{color:#185a99;overflow-wrap:anywhere}</style>
<h1>Day 69 · Positional Encoding</h1><p>Thirteen numerical learning questions. Synthetic vectors, zero training updates. Open any HTML below; keep the entire export folder together for offline Plotly. GIFs/PNGs are regenerable artifacts.</p>
<p><a href="report.json">Configuration, measurements and author-review boundaries</a></p>""" + "".join(cards) + "</html>", encoding="utf-8")
