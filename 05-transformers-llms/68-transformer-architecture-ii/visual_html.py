"""Offline Plotly documents and an integrated gallery; no application server required."""

from html import escape
import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from visual_core import (HEAD_COLORS, TOKENS, default_case, diversity_case,
                         numerical_report, scaling_samples, subspace_cases)

PAGES = {
    "subspaces": ("02_representation_subspaces.html", "Representation subspaces",
        "How do full-input projections change geometry?", "Choose a head, Q/K/V map and direct 3D or PCA view. Rotate either scene and inspect exact pair distances below.",
        "Random untrained projections. Direct 3→3 maps are geometric analogies, not fixed-width MHA. High-dimensional views use separately fitted centered PCA: axes and cross-view distances are not comparable. Exact distances and angles use original coordinates."),
    "geometry": ("03_query_key_value_geometry.html", "Query, key and value geometry",
        "How do direction and magnitude change routing?", "Rotate and resize a query, change key norms, and compare dot products, cosine and the weighted value vector.",
        "Constructed 2D vectors, one head, d_head=2. Two keys share a direction but differ in norm. Probabilities are computational routing, not confidence or linguistic evidence."),
    "heads": ("05_attention_heads_interactive.html", "Independent attention heads",
        "What does averaging maps hide?", "Choose the tensor stage, mask and query. Individual heads share a probability scale; compare the average and final output.",
        "Sentence words are labels for random synthetic vectors. Heads have no learned linguistic roles. Raw/scaled/masked logits are signed values; probabilities use a separate fixed 0–1 color scale."),
    "graph": ("06_token_attention_graph.html", "Token routing graph",
        "Which keys does each query route through?", "Arrowheads point from query to key. Select all heads, a query, an edge threshold and causal or bidirectional routing.",
        "Edge color identifies a head; width and opacity encode probability. Self-loops are drawn outside the node ring. Edges show information routing, not causal explanations or verified language relationships."),
    "scaling": ("08_attention_scaling_interactive.html", "Scaling and saturation",
        "Why divide logits by sqrt(d_head)?", "Use nested paired IID normal samples. Change dimension and magnitudes; compare distributions and the same six-key softmax row.",
        "Samples are independent N(0,1) components before magnitude changes. The reference variance assumes independence. Scaling controls dimension-dependent variance; it does not guarantee uniform attention or task quality. Sample variance has Monte Carlo error."),
    "head-count": ("09_head_count_tradeoffs.html", "Head count and width",
        "What changes at fixed model width?", "Choose a divisible head count and inspect widths, parameters, explicit maps and memory.",
        "Theoretical standard bias-free MHA with equal head widths. Map storage is one materialized probability tensor only. No model training, quality, memory profiling or latency comparison is implied."),
    "diversity": ("12_head_diversity.html", "Head diversity and redundancy",
        "Do identical maps imply identical outputs?", "Heads 0 and 1 have identical Q/K projections but distinct V projections. Compare maps, outputs and zero-head interventions.",
        "Computed untrained control. Cosine compares flattened arrays; it depends on the fixture. Ablation measures layer-output change, not downstream importance or the safety of pruning a trained model."),
    "sharing": ("13_mha_gqa_mqa_interactive.html", "MHA, GQA and MQA",
        "Which query heads share cached K/V?", "Hold query heads fixed and select a valid KV divisor. Follow query→KV connections and calculated parameter/cache payloads.",
        "Architectural illustration and exact count formulas, not trained-model conversion. Counts exclude biases. Sharing can affect quality; equal quality or speed is not assumed."),
    "kv-cache": ("14_kv_cache_explorer.html", "KV cache memory explorer",
        "How do context, batch and KV sharing affect storage?", "Adjust every formula factor; inspect context curves, architecture bars and the rotatable context×batch surface.",
        "Theoretical payload only: excludes weights, temporary activations, allocator overhead, metadata, quantization scales and cache management. INT4 assumes ideally packed 4-bit elements. Capacity planes are hypothetical budgets, not allocatable cache guarantees."),
    "complexity": ("15_attention_complexity.html", "Dense attention complexity",
        "Which costs grow quadratically, and in which workload?", "Compare prefill MAC counts, projection cost, explicit map storage, KV growth and one-token cached decoding.",
        "Counts are mathematical models, not measured throughput or bandwidth. One MAC is one multiply-accumulate, about two floating-point operations. Dense prefill remains quadratic; optimized kernels may avoid materializing maps. Cache grows linearly in length."),
    "validation": ("16_numerical_validation.html", "Numerical validation",
        "What evidence supports these pictures?", "Inspect the actual new CPU float64 checks, causal control, tiled-softmax agreement and variance estimates.",
        "Correctness evidence is limited to these inputs and tolerances. Tests do not approve interpretation, linguistic specialization, GPU kernels, training or production readiness."),
}

STYLE = """
:root{color-scheme:light;font-family:Segoe UI,Arial,sans-serif;color:#172c42;background:#eef2f6}
*{box-sizing:border-box}body{margin:0}header{background:#102b43;color:white;padding:26px 4vw}
header p{color:#d2e4f2;max-width:1000px}h1{font-size:28px;margin:8px 0}h2{font-size:20px}
main{max-width:1600px;margin:auto;padding:20px 3vw}a{color:#00669b}header a{color:#b9e3ff}
.tag{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#77d4c6}
.controls{display:flex;flex-wrap:wrap;gap:14px;background:white;padding:18px;border-radius:10px}
label{display:flex;flex-direction:column;font-size:13px;gap:5px;min-width:125px;font-weight:600}
select,input,button{font:inherit;padding:7px;border:1px solid #a7b8c6;border-radius:5px;background:white}
input[type=range]{padding:0;accent-color:#0072b2;min-width:135px}button{cursor:pointer;color:#004a74}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:18px 0}
.plot{height:395px;background:white;border-radius:9px;overflow:hidden}.plot.tall{height:510px}
.note,.metrics{background:white;padding:16px 20px;border-radius:8px;margin:16px 0;line-height:1.55}
.note{border-left:4px solid #d58b00}.metrics strong{color:#006a68}.metrics{overflow:auto}
table{border-collapse:collapse;font-size:13px;width:100%}td,th{border-bottom:1px solid #dce4eb;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}
.status{font-size:12px;color:#526b80}.gallery{display:grid;grid-template-columns:240px minmax(0,1fr);height:calc(100vh - 150px)}
nav{overflow:auto;background:white;padding:12px}nav a{display:block;padding:7px 10px;text-decoration:none;border-radius:5px}nav a:hover{background:#e5f0f5}nav h3{font-size:13px;margin:14px 8px 4px}
iframe{width:100%;height:100%;border:0}article{background:white;padding:20px;margin:18px 0;border-radius:9px}article img{width:100%;max-height:700px;object-fit:contain}
@media(max-width:800px){.grid{grid-template-columns:1fr}.gallery{grid-template-columns:1fr;height:auto}nav{max-height:220px}iframe{height:85vh}.plot{height:370px}}
"""


def serializable(value):
    """Strict JSON; masked negative infinities become null, labeled as −∞ in UI."""
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def payload(page, seed, samples):
    data = {"page": page, "seed": seed, "headColors": HEAD_COLORS, "tokens": TOKENS}
    if page == "subspaces":
        data["cases"] = subspace_cases(seed)
    elif page in ("heads", "graph", "diversity"):
        layer, traces = default_case(seed)
        data["traces"] = traces
        data["wo"] = layer.wo
        if page == "diversity":
            data["diversity"] = diversity_case(seed)
    elif page == "scaling":
        data["samples"] = scaling_samples(seed, samples)
    elif page == "validation":
        data["report"] = numerical_report(seed, samples)
    return serializable(data)


def write_page(page, folder, seed=168, samples=4000):
    filename, title, question, instructions, limitation = PAGES[page]
    data = json.dumps(payload(page, seed, samples), allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    controls = Path(__file__).with_name("visual_controls.js").read_text(encoding="utf-8")
    source = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Day 68 · {escape(title)}</title>
<style>{STYLE}</style><script>{get_plotlyjs()}</script></head><body>
<header><div class="tag">Day 68 · Local CPU laboratory · seed {seed}</div><h1>{escape(title)}</h1>
<p>{escape(question)}</p></header><main><p>{escape(instructions)}</p>
<div id="controls" class="controls"></div><div id="metrics" class="metrics"></div>
<div id="plots" class="grid"></div><aside class="note">{escape(limitation)}<br><b>Interpretation: author review required.</b></aside>
<p id="status" class="status" role="status">Initializing local calculations…</p></main>
<script>const DATA={data};{controls}</script></body></html>"""
    path = folder / filename
    path.write_text(source, encoding="utf-8")
    return path


CATEGORIES = {
    "Architecture overview": ("architecture", "single-vs-multi"),
    "Representation subspaces": ("subspaces",),
    "Query-key-value geometry": ("geometry",),
    "Matrix operations": ("matrices", "concatenation"),
    "Attention head exploration": ("heads", "graph"),
    "Scaling and masking": ("scaling", "causal"),
    "Head count and diversity": ("head-count", "diversity"),
    "MHA, GQA and MQA": ("sharing",),
    "KV cache and complexity": ("kv-cache", "complexity", "flash"),
    "Numerical validation": ("validation",),
}


def write_gallery(folder, groups, artifacts, animations):
    links = {}
    for group in groups:
        htmls = [name for name in artifacts[group] if name.endswith(".html")]
        links[group] = htmls[0] if htmls else f"gallery.html#{group}"
    sections = []
    for group, animation in animations.items():
        frames = animation["frames"]
        options = "".join(f'<option value="{escape(frame)}">Stage {i + 1}</option>' for i, frame in enumerate(frames))
        sections.append(f'<article id="{group}"><h2>{escape(group.replace("-", " ").title())}</h2>'
                        f'<p>{escape(animation["caption"])}</p><label>Pause and inspect a stage<select onchange="this.parentElement.nextElementSibling.src=this.value">{options}</select></label>'
                        f'<img src="{escape(frames[0])}" alt="Computed {escape(group)} stage">'
                        f'<p><a href="{escape(animation["gif"])}">Open animated GIF</a> · '
                        f'<a href="{escape(frames[0])}" download>Download first frame</a></p></article>')
    gallery = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Day 68 animation stages</title><style>{STYLE}</style><main><h1>Animation stages</h1><p>Actual calculations or explicitly constructed examples. Select a frame to pause at a mathematical stage. GIF links animate the complete sequence. No trained semantics; author review required.</p>{"".join(sections)}</main></html>'
    (folder / "gallery.html").write_text(gallery, encoding="utf-8")
    navigation = []
    for category, members in CATEGORIES.items():
        present = [group for group in members if group in groups]
        if present:
            navigation.append(f'<h3>{escape(category)}</h3>' + "".join(
                f'<a href="{escape(links[group])}" target="viewer">{escape(group.replace("-", " ").title())}</a>' for group in present))
    first = links[groups[0]]
    index = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Day 68 · Multi-Head Attention Lab</title><style>{STYLE}</style><header><div class="tag">Interactive technical laboratory</div><h1>Multi-Head Attention · Day 68</h1><p>Explore one forward pass, geometry, routing and serving arithmetic. Synthetic untrained computations, constructed controls and theoretical counts are labeled separately. Author review required.</p></header><div class="gallery"><nav aria-label="Learning modules">{"".join(navigation)}<hr><a href="report.json" target="_blank">Execution record (JSON)</a></nav><iframe name="viewer" title="Selected learning module" src="{first}"></iframe></div></html>'
    (folder / "index.html").write_text(index, encoding="utf-8")
