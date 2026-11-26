"""Standalone Plotly HTML pages with local controls and embedded calculation data."""

import html
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


TITLES = {
    "space": "The structured output space",
    "masking": "Explore constrained token probabilities",
    "semantics": "Structural compliance and source fidelity",
    "reliability": "How many attempts meet a reliability target?",
    "surface": "Independent retry reliability in three dimensions",
    "cost": "Retry reliability, expected cost and latency",
    "correlation": "Independent, persistent and feedback-aware failures",
}
FILES = {"space": "structured_output_space.html", "masking": "interactive_token_masking.html",
         "semantics": "structural_vs_semantic_correctness.html", "reliability": "retry_reliability_curves.html",
         "surface": "retry_reliability_surface.html", "cost": "retry_cost_reliability_tradeoff.html",
         "correlation": "retry_failure_simulation.html"}


def control(label, name, value, minimum, maximum, step):
    return (f'<label for="{name}">{html.escape(label)}</label>'
            f'<input id="{name}" type="number" min="{minimum}" max="{maximum}" step="{step}" value="{value}">')


def controls(demo):
    if demo == "space":
        return ('<label for="gate">Activate constraints progressively</label><select id="gate">'
                '<option value="0">0. No gates</option><option value="1">1. Parse JSON</option>'
                '<option value="2">2. Schema + domain</option><option value="3">3. Cross-field consistency</option>'
                '<option value="4">4. Synthetic source evidence</option></select>')
    if demo == "masking":
        result = control("Temperature (> 0)", "temperature", 1, .1, 3, .1)
        result += ('<label for="schema">Schema constraint</label><select id="schema">'
                   '<option value="numeric">Numeric, non-nullable</option><option value="nullable">Numeric, nullable</option>'
                   '<option value="all">All toy tokens</option><option value="custom">Custom mask</option></select>')
        for i, label in enumerate(("100", "200", '"1500"', "null", "true")):
            result += control(f"Logit: {label}", f"logit{i}", (1.2, .8, 2, -.2, .1)[i], -10, 10, .1)
            result += f'<label class="check"><input id="allow{i}" type="checkbox" {"checked" if i < 2 else ""}>Allow {html.escape(label)}</label>'
        return result + control("Sampling seed", "seed", 777, 0, 99999, 1) + '<button id="sample">Draw 1000 seeded samples</button>'
    if demo in ("reliability", "cost"):
        result = control("Per-attempt success p", "p", .525, 0, 1, .025)
        result += control("Maximum total attempts", "n", 10, 1, 30, 1)
        result += control("Target reliability", "target", .95, 0, 1, .01)
        if demo == "cost":
            result += control("Synthetic cost per attempt ($)", "cost", .02, 0, 1, .005)
            result += control("Assumed seconds per attempt", "duration", .8, 0, 10, .1)
            result += control("Expected cost budget per request ($)", "budget", .04, 0, 10, .01)
        return result
    if demo in ("surface", "cost-surface"):
        return '<label for="threshold">Reliability contour</label><select id="threshold"><option>.9</option><option selected>.95</option><option>.99</option></select>'
    if demo == "correlation":
        return '<label for="metric">Simulation metric</label><select id="metric"><option value="success">Success probability</option><option value="attempts">Expected attempts</option><option value="exhaustion">Budget exhaustion probability</option></select>'
    return '<p>Hover over a point to inspect its checks, input and known synthetic reference. Click legend entries to filter categories.</p>'


NOTES = {
    "space": "Illustrative clustered coordinates, not LLM embeddings, distances, probabilities or estimated population proportions. Records and eligibility come from actual validators. Legend filtering changes display only.",
    "masking": "Whole toy tokens and a user-supplied mask, not a tokenizer or full grammar decoder. q(v)=p(v)/allowed mass for allowed tokens. Zero allowed tokens blocks generation explicitly. Temperature is strictly positive.",
    "semantics": "Scores are fractions of explicit checks, not calibrated probabilities or LLM confidence. Schema score: object, exact key set, eight field types, three literals/enums, three money bounds and document-ID format (17 checks). Semantic score: eight exact field matches plus arithmetic (9 checks).",
    "reliability": "Theoretical independent attempts with constant p. All budgets include the initial call. p=0 cannot reach a positive target; p=1 succeeds on call one. A target of 1 is not finitely achievable when 0<p<1.",
    "surface": "P(success)=1-(1-p)^n. Only integer attempt rows are computed; interpolation between rows is visual continuity. This is not measured model reliability. Contours mark 90%, 95%, 99%.",
    "cost": "Independent attempts, constant p/cost/duration and stop on first success. No parallel calls, backoff, queueing, validation cost or timeout. Cost per accepted response is a population ratio E[cost]/P(success). It is undefined if success probability is zero.",
    "cost-surface": "Expected generation cost = c * sum((1-p)^k), k=0..n-1; c=$0.02/call. Only integer budgets computed. Surface color gives final reliability. Assumed costs, not provider prices or measured billing.",
    "correlation": "Actual seeded Monte Carlo, not LLM observations. Persistent mixture: 25% permanent failures, remaining requests p=0.7; independent p=0.525. Feedback probabilities are configured conditional success probabilities, not an experimentally established feedback benefit. Error bars are +/-1.96 binomial SE (normal approximation).",
}


def write_page(path: Path, demo: str, lab: dict):
    title = TITLES.get(demo, "Expected retry generation cost surface")
    payload = {"demo": demo}
    if demo in ("space", "semantics"):
        payload["validation"] = lab["validation"]
    if demo == "correlation":
        payload["simulation"] = lab["simulation"]
    script = Path(__file__).with_name("visual_controls.js").read_text(encoding="utf-8")
    encoded = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{html.escape(title)} | Day 77</title><style>'
                'body{margin:0;color:#243746;background:#f5f8fc;font:16px system-ui,sans-serif}'
                'header,main{max-width:1400px;margin:auto;padding:22px}h1{font-size:30px;margin:6px 0}'
                '.tag{color:#0072b2;font-weight:700;font-size:13px;letter-spacing:.08em}'
                '.shell{display:grid;grid-template-columns:260px 1fr;gap:20px}aside,.card{background:white;border:1px solid #d9e2eb;border-radius:12px;padding:18px}'
                'aside label{display:block;font-size:14px;margin:12px 0 4px}input[type=number],select{box-sizing:border-box;width:100%;padding:8px;border:1px solid #9caebe;border-radius:5px}'
                '.check{display:flex;gap:8px}button{margin-top:14px;padding:10px;border:0;border-radius:5px;background:#0072b2;color:white;cursor:pointer}'
                '#status{display:block;white-space:pre-wrap;border-left:4px solid #0072b2;padding:15px;background:#edf4fa;line-height:1.6}'
                '#plot,#plot2{min-height:500px}#detail{white-space:pre-wrap;font:13px ui-monospace,monospace;overflow-wrap:anywhere}'
                '.note{font-size:14px;line-height:1.6;color:#435a6b}footer{margin:20px;color:#536879;font-size:13px}'
                '@media(max-width:850px){.shell{grid-template-columns:1fr}#plot,#plot2{min-height:420px}}'
                '</style></head><body><header><div class="tag">DAY 77 / STRUCTURED OUTPUTS</div>'
                f'<h1>{html.escape(title)}</h1><p>Synthetic local learning lab. Representation, correctness, evidence and authority are separate checks.</p></header>'
                f'<main><div class="shell"><aside>{controls(demo)}</aside><section class="card">'
                '<output id="status" role="status" aria-live="polite">Calculating...</output>'
                '<div id="plot"></div><div id="plot2"></div><pre id="detail"></pre></section></div>'
                f'<p class="note">{html.escape(NOTES[demo])}</p></main>'
                '<footer>Offline HTML: Plotly runtime included. No model calls. Author interpretation review required.</footer>'
                f'<script>{get_plotlyjs()}</script><script>const LAB={encoded};\n{script}</script></body></html>')
    path.write_text(document, encoding="utf-8")


def render_html(demo: str, root: Path, lab: dict) -> list[str]:
    if demo not in FILES:
        return []
    write_page(root / FILES[demo], demo, lab)
    files = [FILES[demo]]
    if demo == "cost":
        write_page(root / "expected_retry_cost_surface.html", "cost-surface", lab)
        files.append("expected_retry_cost_surface.html")
    return files
