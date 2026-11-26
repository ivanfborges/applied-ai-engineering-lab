"""Offline Plotly shell with checkpoint playback, configuration views and reset."""

import html
import json
from pathlib import Path

from plotly.offline import get_plotlyjs


STYLE = """
body{margin:0;background:#eef3f8;color:#183247;font:16px system-ui,sans-serif}
header{background:#183247;color:white;padding:24px 4vw}h1{font-size:25px;margin:0 0 12px}
header p{max-width:1100px;line-height:1.5}main{display:grid;grid-template-columns:280px minmax(0,1fr);gap:20px;padding:22px 3vw}
aside,article{background:white;border:1px solid #cddbe7;border-radius:12px;padding:20px}
label{display:block;margin-bottom:20px;font-size:14px}input,select{width:100%;margin-top:8px;font:inherit;padding:6px;box-sizing:border-box}
button{background:#1b638b;color:white;border:0;border-radius:5px;padding:10px;margin:3px;cursor:pointer}
#chart{height:650px}#metrics{font:13px ui-monospace,monospace;white-space:pre-wrap;line-height:1.6;padding:14px;background:#edf3f7}
.limit{font-size:14px;line-height:1.6;color:#465d70}a{color:#17648f}aside p{line-height:1.6;font-size:14px}
@media(max-width:850px){main{grid-template-columns:1fr}#chart{height:520px}}
"""


def write_interactive(path, title, description, views, limitation):
    """views map named configurations to actual recorded/calculated Plotly frames."""
    path = Path(path)
    runtime = path.parent / "plotly.min.js"
    if not runtime.exists():
        runtime.write_text(get_plotlyjs(), encoding="utf-8")
    payload = {}
    for name, frames in views.items():
        payload[name] = []
        for figure, state in frames:
            figure.update_layout(template="plotly_white", font={"family": "Arial", "color": "#183247"},
                                 margin={"t": 80, "l": 70, "r": 60, "b": 85},
                                 legend={"orientation": "h", "y": -.18}, uirevision="camera")
            entry = json.loads(figure.to_json())
            # to_json converts NumPy arrays to typed buffers in recent Plotly.
            payload[name].append({"data": entry["data"], "layout": entry["layout"], "state": state})
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    options = "".join(f'<option value="{html.escape(k)}">{html.escape(k)}</option>' for k in views)
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{STYLE}</style><script src="plotly.min.js"></script></head><body>
<header><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></header>
<main><aside><label for="view">Configuration / view<select id="view">{options}</select></label>
<label for="checkpoint">Recorded / calculated frame<input type="range" id="checkpoint" min="0" max="0" value="0" step="1"></label>
<button id="play">Play</button><button id="reset">Reset</button>
<p>Rotate, pan and zoom 3D views with the mouse. Use the Plotly toolbar to reset the camera. Click legend labels to toggle traces.</p>
<p><a href="index.html">Return to gallery</a></p></aside><article><div id="chart"></div>
<pre id="metrics"></pre><p class="limit">{html.escape(limitation)}</p>
<p class="limit">Synthetic educational evidence. Interpretation: author review required.</p></article></main>
<script id="lab-data" type="application/json">{encoded}</script><script>
const P=JSON.parse(document.getElementById('lab-data').textContent);
const view=document.getElementById('view'), checkpoint=document.getElementById('checkpoint');
let timer=null; window.labReady=false;
function stop(){{if(timer)clearInterval(timer);timer=null;document.getElementById('play').textContent='Play'}}
function draw(){{window.labReady=false;let frames=P[view.value];checkpoint.max=frames.length-1;
checkpoint.value=Math.min(Number(checkpoint.value),frames.length-1);let item=frames[Number(checkpoint.value)];
window.labState={{view:view.value,frame:Number(checkpoint.value),...item.state}};
document.getElementById('metrics').textContent=JSON.stringify(window.labState,null,2);
return Plotly.react('chart',item.data,item.layout,{{responsive:true,displaylogo:false}}).then(()=>{{window.labReady=true}})}}
view.addEventListener('change',()=>{{stop();checkpoint.value=0;draw()}});
checkpoint.addEventListener('input',()=>{{stop();draw()}});
document.getElementById('play').addEventListener('click',()=>{{if(timer){{stop();return}}document.getElementById('play').textContent='Pause';
timer=setInterval(()=>{{if(!window.labReady)return;if(Number(checkpoint.value)>=Number(checkpoint.max)){{stop();return}}
checkpoint.value=Number(checkpoint.value)+1;draw()}},650)}});
document.getElementById('reset').addEventListener('click',()=>{{stop();checkpoint.value=0;Plotly.purge('chart');draw()}});draw();
</script></body></html>'''
    path.write_text(document, encoding="utf-8")


def write_gallery(folder, reports):
    folder = Path(folder)
    items = []
    for path in sorted(folder.iterdir()):
        if path.suffix not in (".gif", ".png", ".html") or path.name == "index.html":
            continue
        name = html.escape(path.name)
        preview = f'<img loading="lazy" src="{name}" alt="{name.replace("_", " ")}">' if path.suffix != ".html" else ''
        items.append(f'<article><h2><a href="{name}">{name}</a></h2>{preview}<p>{path.stat().st_size:,} bytes · generated locally</p></article>')
    limitations = ''.join(f'<li><b>{html.escape(name)}</b>: {html.escape(report["limitation"])}</li>' for name, report in reports.items())
    (folder / "index.html").write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Day 74 visual lab</title>
<style>{STYLE}main{{display:block}}article{{margin:20px 0}}img{{max-width:100%;height:auto}}h2{{font-size:18px}}</style></head>
<body><header><h1>Day 74 — Fine-Tuning and Alignment</h1><p>Synthetic calculations and actual CPU training. Author review required.</p></header>
<main><article><h2>Evidence boundaries</h2><ul>{limitations}</ul><p>JSON records and generated assets are ignored. This gallery is not a published preview selection.</p></article>{''.join(items)}</main></body></html>''', encoding="utf-8")
    return [p.name for p in sorted(folder.iterdir()) if p.suffix in (".png", ".gif", ".html") and p.name != "index.html"]
