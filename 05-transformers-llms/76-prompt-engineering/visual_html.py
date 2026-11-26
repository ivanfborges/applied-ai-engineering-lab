"""Offline Plotly pages with accessible native controls; no server/CDN required."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder


def state(label: str, figure: go.Figure, detail: str = "", metrics: dict | None = None) -> dict:
    figure.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif", size=14),
                         margin=dict(l=65, r=35, t=65, b=65), height=550,
                         paper_bgcolor="#ffffff", colorway=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    return {"label": label, "figure": json.loads(figure.to_json()), "detail": detail,
            "metrics": metrics or {}}


def card(key: str, title: str, question: str, takeaway: str, assumption: str,
         states: list[dict], section: str, evidence: str = "SYNTHETIC ILLUSTRATION — NOT LLM RESULTS") -> dict:
    return {"key": key, "title": title, "question": question, "takeaway": takeaway,
            "assumption": assumption, "states": states, "section": section, "evidence": evidence}


def write_page(path: Path, cards: list[dict], title: str, animations: list[Path] | None = None) -> None:
    """Embed Plotly once and render charts lazily so a dashboard stays responsive."""
    payload = json.dumps(cards, cls=PlotlyJSONEncoder, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e")
    nav = "".join(f'<button type="button" data-section="{html.escape(section)}">{html.escape(section)}</button>'
                  for section in dict.fromkeys(item["section"] for item in cards))
    animations_html = ""
    if animations:
        for gif in animations:
            encoded = base64.b64encode(gif.read_bytes()).decode("ascii")
            animations_html += f'<details><summary>{html.escape(gif.stem)} — synthetic animation</summary><img alt="{html.escape(gif.stem)}" src="data:image/gif;base64,{encoded}"></details>'
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#172b42;font:16px/1.5 Arial,sans-serif}
header{background:#132b44;color:white;padding:28px max(24px,calc((100vw - 1200px)/2))}header p{max-width:1000px}
main,nav{max-width:1240px;margin:auto;padding:18px}nav{display:flex;flex-wrap:wrap;gap:8px}
button,select,input{font:inherit}button,select{padding:9px;border:1px solid #52677e;border-radius:6px;background:white;color:#172b42}
button{cursor:pointer}button.active{background:#00679e;color:white}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #D55E00}
.chapter{padding:24px;margin-bottom:24px;background:white;border:1px solid #c7d3de;border-radius:12px}
h1{margin:0}h2{font-size:25px}.badge{font-size:13px;font-weight:bold;letter-spacing:.03em;color:#8a3800;background:#fff0da;padding:8px;display:inline-block}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:18px 0}.controls input{min-width:210px;flex:1}
.plot{width:100%;min-height:550px}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:#e9f3fa;border-left:4px solid #0072B2;padding:8px 14px}
.detail{white-space:pre-wrap;background:#f3f6fa;border-radius:6px;padding:16px;max-height:360px;overflow:auto;font:14px/1.5 Consolas,monospace}
.note{border-left:4px solid #D55E00;padding-left:14px}details{background:white;padding:12px;margin-bottom:10px}img{max-width:100%;height:auto}
@media(max-width:600px){main{padding:8px}.chapter{padding:12px}header{padding:20px}.plot{min-height:500px}}
</style><script>__PLOTLY__</script></head><body><header><h1>__TITLE__</h1>
<p>Explore assumptions, inspect failure cases, and separate structural validity from supported evidence.
All simulation charts are explicitly synthetic. Author review required for interpretations.</p></header>
<nav aria-label="Learning sections">__NAV__</nav><main><div id="chapters"></div>
<section id="animations"><h2>Animation gallery</h2>__ANIMATIONS__</section></main>
<script id="lab-data" type="application/json">__DATA__</script><script>
const cards=JSON.parse(document.getElementById('lab-data').textContent);
const timers=new Map(); const readiness=new Map();
const root=document.getElementById('chapters');
function renderCard(item){
 const section=document.createElement('section');section.className='chapter';section.id=item.key;section.dataset.section=item.section;
 const h=document.createElement('h2');h.textContent=item.title;section.append(h);
 const badge=document.createElement('span');badge.className='badge';badge.textContent=item.evidence;section.append(badge);
 const q=document.createElement('p');q.textContent=item.question;section.append(q);
 const controls=document.createElement('div');controls.className='controls';
 const label=document.createElement('label');label.htmlFor=item.key+'-select';label.textContent='Configuration';controls.append(label);
 const select=document.createElement('select');select.id=item.key+'-select';
 item.states.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.label;select.append(o)});controls.append(select);
 const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=item.states.length-1;slider.step=1;slider.value=0;slider.setAttribute('aria-label',item.title+' step');controls.append(slider);
 const play=document.createElement('button');play.textContent='Play';play.type='button';controls.append(play);
 const pause=document.createElement('button');pause.textContent='Pause';pause.type='button';controls.append(pause);section.append(controls);
 const metrics=document.createElement('div');metrics.className='metrics';metrics.setAttribute('aria-live','polite');section.append(metrics);
 const plot=document.createElement('div');plot.className='plot';plot.id=item.key+'-plot';section.append(plot);
 const detail=document.createElement('pre');detail.className='detail';section.append(detail);
 const takeaway=document.createElement('p');takeaway.className='note';takeaway.textContent='Takeaway: '+item.takeaway;section.append(takeaway);
 const assumption=document.createElement('p');assumption.textContent='Assumptions / limits: '+item.assumption;section.append(assumption);
 root.append(section);
 async function update(index){
   section.dataset.ready='false';
   index=Number(index);const current=item.states[index];select.value=index;slider.value=index;metrics.replaceChildren();
   Object.entries(current.metrics).forEach(([k,v])=>{const m=document.createElement('span');m.className='metric';m.textContent=k+': '+(v===null?'undefined (zero denominator)':v);metrics.append(m)});
   detail.textContent=current.detail;section.dataset.step=index;
   await Plotly.react(plot,current.figure.data,current.figure.layout,{responsive:true,displaylogo:false,scrollZoom:false});
   section.dataset.ready='true';
 }
 select.addEventListener('change',()=>update(select.value));slider.addEventListener('input',()=>update(slider.value));
 play.addEventListener('click',()=>{clearInterval(timers.get(item.key));timers.set(item.key,setInterval(()=>update((Number(select.value)+1)%item.states.length),1200))});
 pause.addEventListener('click',()=>clearInterval(timers.get(item.key)));
 readiness.set(item.key,()=>update(select.value));return section;
}
cards.forEach(renderCard);
function showSection(name){
 document.querySelectorAll('.chapter').forEach(s=>{s.hidden=s.dataset.section!==name;if(!s.hidden){if(s.dataset.ready!=='true')readiness.get(s.id)();else Plotly.Plots.resize(s.querySelector('.plot'))}});
 document.querySelectorAll('nav button').forEach(b=>{b.classList.toggle('active',b.dataset.section===name);b.setAttribute('aria-pressed',String(b.dataset.section===name))});
 for(const [key,timer] of timers){clearInterval(timer);timers.delete(key)}
 document.body.dataset.section=name;
}
document.querySelectorAll('nav button').forEach(b=>b.addEventListener('click',()=>showSection(b.dataset.section)));
if(cards.length)showSection(cards[0].section);
</script></body></html>'''
    for marker, value in {"__TITLE__": html.escape(title), "__NAV__": nav,
                          "__ANIMATIONS__": animations_html, "__DATA__": payload,
                          "__PLOTLY__": get_plotlyjs()}.items():
        document = document.replace(marker, value)
    path.write_text(document, encoding="utf-8")
