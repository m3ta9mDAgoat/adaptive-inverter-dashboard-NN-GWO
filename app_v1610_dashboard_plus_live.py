from __future__ import annotations

import os
import re
import types
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
import uvicorn

ROOT = Path(__file__).resolve().parent
DASHBOARD_FILE = ROOT / "Adaptive_Inverter_Victory_Showcase_V16_10.html"
APP_FILE = ROOT / "app.py"

# -----------------------------------------------------------------------------
# Load ONLY the computational backend from the existing V16.12 app.py.
# This keeps ANN + GWO + supervisor unchanged and skips its old UI.
# -----------------------------------------------------------------------------
def load_backend_only():
    source = APP_FILE.read_text(encoding="utf-8")
    lines = source.splitlines()
    cut = None
    for i, line in enumerate(lines):
        if line.strip().startswith("APP = build_app("):
            cut = i
            break
    if cut is None:
        raise RuntimeError("Could not locate `APP = build_app()` in app.py")

    backend_source = "\n".join(lines[:cut]) + "\n"
    module = types.ModuleType("adaptive_inverter_backend")
    module.__file__ = str(APP_FILE)
    module.__name__ = "adaptive_inverter_backend"
    exec(compile(backend_source, str(APP_FILE), "exec"), module.__dict__)
    return module

engine = load_backend_only()

# -----------------------------------------------------------------------------
# Live page helpers
# -----------------------------------------------------------------------------
CORE_METRICS = [
    "Junction temperature",
    "MOSFET hot loss",
    "Electrical efficiency",
    "Delivered requested power",
    "Switching frequency",
    "THD",
    "Output power",
    "RMS current",
    "RMS voltage",
    "Current ripple",
    "Relative thermal lifetime",
]

DEFAULT_METRICS = [
    "Junction temperature",
    "MOSFET hot loss",
    "Electrical efficiency",
    "Delivered requested power",
    "Switching frequency",
    "THD",
]

PRESET_PRIORITY = [
    "V09 — headline thermal benefit",
    "V11 — thermal boundary / rescue",
    "V13 — complete-supervisor derating",
    "V06 — continuous GWO interpolation challenge",
    "V04 — high-load transition / 14 kHz",
    "V01 — light load / AI stays at 24 kHz",
    "Validation — 10 Ω high-PF reference",
    "Custom",
]
PRESETS = [x for x in PRESET_PRIORITY if x in getattr(engine, "CASE_PRESETS", {})]
if not PRESETS:
    PRESETS = list(getattr(engine, "CASE_PRESETS", {}).keys())


def _num(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def metric_value(table: pd.DataFrame, metric: str, col: str):
    if table is None or len(table) == 0 or col not in table.columns:
        return np.nan
    row = table[table["Metric"].astype(str).eq(metric)]
    if row.empty:
        return np.nan
    return _num(row.iloc[0][col])


def strip_html(text: str):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


def state_from_status(status_html: str, label="AI"):
    text = strip_html(status_html)
    m = re.search(rf"{label}:\s*([^—]+)—", text)
    return m.group(1).strip().replace("_", " ") if m else "N/A"


def fmt(v, d=2, suffix=""):
    return "N/A" if not np.isfinite(_num(v)) else f"{float(v):.{d}f}{suffix}"


def decision_cards(status_html: str, table: pd.DataFrame):
    ai_f = metric_value(table, "Switching frequency", "AI")
    fx_f = metric_value(table, "Switching frequency", "Fixed")
    ai_t = metric_value(table, "Junction temperature", "AI")
    fx_t = metric_value(table, "Junction temperature", "Fixed")
    ai_l = metric_value(table, "MOSFET hot loss", "AI")
    fx_l = metric_value(table, "MOSFET hot loss", "Fixed")
    ai_q = metric_value(table, "Delivered requested power", "AI")
    fx_q = metric_value(table, "Delivered requested power", "Fixed")
    ai_state = state_from_status(status_html, "AI")

    dt = fx_t - ai_t if np.isfinite(ai_t) and np.isfinite(fx_t) else np.nan
    dl = fx_l - ai_l if np.isfinite(ai_l) and np.isfinite(fx_l) else np.nan

    if np.isfinite(ai_f) and np.isfinite(fx_f):
        if abs(ai_f - fx_f) < 0.01:
            banner = f"No frequency change required — AI selected the same {ai_f:.2f} kHz operating point as the fixed reference."
            klass = "same"
        else:
            banner = f"Adaptive action: {fx_f:.2f} kHz fixed reference → {ai_f:.2f} kHz selected by AI."
            klass = "active"
    else:
        banner = "Supervisor protection action detected."
        klass = "warn"

    return f"""
    <div class='run-banner {klass}'>{banner}</div>
    <div class='live-kpis'>
      <div class='live-kpi hero'><span>AI SELECTED FREQUENCY</span><strong>{fmt(ai_f,3,' kHz')}</strong><small>Final GWO + safeguard command</small></div>
      <div class='live-kpi'><span>OPERATING STATE</span><strong class='state'>{ai_state}</strong><small>Thermal supervisor result</small></div>
      <div class='live-kpi'><span>TEMPERATURE SAVED</span><strong>{fmt(dt,2,' °C')}</strong><small>Fixed − AI</small></div>
      <div class='live-kpi'><span>MOSFET LOSS SAVED</span><strong>{fmt(dl,2,' W')}</strong><small>Fixed − AI</small></div>
      <div class='live-kpi'><span>DELIVERED POWER</span><strong>{fmt(ai_q,1,'%')}</strong><small>Fixed: {fmt(fx_q,1,'%')}</small></div>
    </div>
    """


def core_comparison_figure(table: pd.DataFrame):
    metrics = [
        ("Junction temperature", "Temperature (°C)"),
        ("MOSFET hot loss", "MOSFET loss (W)"),
        ("Electrical efficiency", "Efficiency (%)"),
        ("Delivered requested power", "Delivered power (%)"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[x[1] for x in metrics], vertical_spacing=0.18)
    for i, (metric, title) in enumerate(metrics):
        r = i // 2 + 1
        c = i % 2 + 1
        ai = metric_value(table, metric, "AI")
        fx = metric_value(table, metric, "Fixed")
        fig.add_trace(go.Bar(x=["AI", "Fixed 24 kHz"], y=[ai, fx], text=[fmt(ai,2), fmt(fx,2)], textposition="outside", marker_color=["#16b989", "#4f6f93"], showlegend=False), row=r, col=c)
    fig.update_layout(height=610, margin=dict(l=35,r=20,t=70,b=25), paper_bgcolor="#ffffff", plot_bgcolor="#f7f9fc", font=dict(family="Inter, Arial", color="#142033"), bargap=0.38)
    fig.update_yaxes(gridcolor="#e7edf5", zeroline=False)
    return fig


def run_live(preset, PF, R, Tamb, study_s, fixed_kHz, mode, metrics, plot_metric, Ea, progress=gr.Progress()):
    requested = list(dict.fromkeys((metrics or DEFAULT_METRICS) + [plot_metric] + DEFAULT_METRICS))
    status, table, selected_fig, convergence_fig, candidates = engine.simulate_live_case(
        preset, PF, R, Tamb, study_s, fixed_kHz, mode,
        requested, plot_metric, Ea, progress=progress
    )
    cards = decision_cards(status, table)
    visible = table[table["Metric"].isin(metrics or DEFAULT_METRICS)].reset_index(drop=True)
    core_fig = core_comparison_figure(table)
    return cards, visible, core_fig, selected_fig, convergence_fig, candidates, status


def load_preset(name):
    pf, r, ta, ts, note = engine.load_preset(name)
    return pf, r, ta, ts, note


LIVE_CSS = """
.gradio-container{max-width:1500px!important;margin:auto!important;padding:22px 26px 45px!important;background:#eef3f8!important;font-family:Inter,Arial,sans-serif!important}
body{background:#eef3f8!important}.live-head{padding:28px 30px;border-radius:24px;background:linear-gradient(125deg,#071b34,#0d5272);color:#fff;box-shadow:0 18px 50px rgba(7,27,52,.18);margin-bottom:18px}.live-head h1{margin:0 0 8px;font-size:36px}.live-head p{margin:0;color:#d9ebf7;font-size:15px;line-height:1.6}.panel{background:#fff;border:1px solid #dbe4ef;border-radius:20px;padding:18px!important;box-shadow:0 10px 32px rgba(20,48,80,.08)}
.live-kpis{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:14px;margin:15px 0}.live-kpi{background:#fff;border:1px solid #dbe4ef;border-radius:17px;padding:17px 16px;box-shadow:0 8px 24px rgba(20,48,80,.07)}.live-kpi.hero{background:linear-gradient(135deg,#071b34,#0e5276);color:white;border:none}.live-kpi span{display:block;font-size:10px;font-weight:800;letter-spacing:.06em;color:#68788e}.live-kpi.hero span,.live-kpi.hero small{color:#d8eaf6}.live-kpi strong{display:block;font-size:25px;margin:9px 0 5px;color:#102038}.live-kpi.hero strong{color:white}.live-kpi strong.state{font-size:16px}.live-kpi small{color:#738196}.run-banner{padding:14px 17px;border-radius:14px;font-weight:750;margin:14px 0}.run-banner.active{background:#e8f8f1;color:#096348;border:1px solid #bce8d5}.run-banner.same{background:#edf5ff;color:#0c477f;border:1px solid #cce2fa}.run-banner.warn{background:#fff3dc;color:#80510d;border:1px solid #efd298}
#run-btn{background:#071b34!important;color:white!important;font-weight:800!important;min-height:48px!important;border-radius:12px!important}footer{display:none!important}@media(max-width:900px){.live-kpis{grid-template-columns:1fr 1fr}}
"""


def build_live_ui():
    with gr.Blocks(title="Adaptive Inverter — Live Interactive", css=LIVE_CSS) as live:
        gr.HTML("""<div class='live-head'><h1>Live Interactive Case Lab</h1><p>Run the trained ANN, continuous GWO search and supervisory controller on a new operating point. Results are calculated live — not retrieved from Monte Carlo.</p></div>""")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                preset = gr.Dropdown(PRESETS, value=PRESETS[0] if PRESETS else None, label="Important case / preset")
                load_btn = gr.Button("Load preset")
                preset_note = gr.Markdown("")
                pf = gr.Slider(0.60, 0.99, value=0.70, step=0.001, label="Power factor")
                r = gr.Slider(5.0, 120.0, value=10.0, step=0.01, label="Resistance (Ω)")
                ambient = gr.Slider(15, 55, value=25, step=1, label="Ambient temperature (°C)")
                study = gr.Slider(10, 600, value=100, step=10, label="Thermal study time (s)")
                fixed = gr.Slider(6, 24, value=24, step=1, label="Fixed reference frequency (kHz)")
                mode = gr.Radio(["Complete supervisor", "Frequency only"], value="Complete supervisor", label="Evaluation mode")
                metrics = gr.CheckboxGroup(CORE_METRICS, value=DEFAULT_METRICS, label="Metrics to display")
                plot_metric = gr.Dropdown(CORE_METRICS, value="Junction temperature", label="Detailed metric plot")
                Ea = gr.Slider(0.30, 1.10, value=0.70, step=0.05, label="Arrhenius activation energy Ea (eV)")
                run = gr.Button("Run live ANN + GWO", elem_id="run-btn")

            with gr.Column(scale=2):
                decision = gr.HTML()
                results = gr.Dataframe(label="Selected metrics", interactive=False, wrap=True)

        gr.Markdown("## Core comparison")
        core_plot = gr.Plot()
        gr.Markdown("## Selected metric detail")
        detail_plot = gr.Plot()
        with gr.Accordion("GWO convergence and candidate evidence", open=False):
            convergence = gr.Plot(label="GWO convergence")
            candidate_table = gr.Dataframe(label="Candidate evidence", interactive=False, wrap=True)
            raw_status = gr.HTML()

        load_btn.click(load_preset, inputs=[preset], outputs=[pf, r, ambient, study, preset_note])
        run.click(
            run_live,
            inputs=[preset, pf, r, ambient, study, fixed, mode, metrics, plot_metric, Ea],
            outputs=[decision, results, core_plot, detail_plot, convergence, candidate_table, raw_status],
        )
    return live

live_ui = build_live_ui()

# -----------------------------------------------------------------------------
# FastAPI shell: exact V16.10 dashboard on one page, live lab on second page.
# -----------------------------------------------------------------------------
api = FastAPI(title="Adaptive Inverter Dashboard V16.10 + Live")

SHELL = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Adaptive Inverter</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#07111f;font-family:Inter,Arial,sans-serif}body{overflow:hidden}.nav{height:56px;background:#071523;border-bottom:1px solid #1b3147;display:flex;align-items:center;justify-content:space-between;padding:0 18px;color:white}.brand{font-weight:800;letter-spacing:.01em}.tabs{display:flex;gap:8px}.tab{border:1px solid #29445e;background:#0d2236;color:#cfe0ef;padding:9px 14px;border-radius:10px;font-weight:750;cursor:pointer}.tab.active{background:#ffffff;color:#0c2034;border-color:#fff}.frame{width:100%;height:calc(100vh - 56px);border:0;background:white;display:block}
</style>
</head>
<body>
<div class="nav"><div class="brand">Adaptive Inverter — V16.10</div><div class="tabs"><button id="dashBtn" class="tab active" onclick="showPage('/dashboard-raw',this)">V16.10 Dashboard</button><button id="liveBtn" class="tab" onclick="showPage('/live/',this)">Live Interactive</button></div></div>
<iframe id="mainFrame" class="frame" src="/dashboard-raw"></iframe>
<script>
function showPage(url,el){document.getElementById('mainFrame').src=url;document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');}
</script>
</body>
</html>
"""

@api.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(SHELL)

@api.get("/dashboard-raw")
def dashboard_raw():
    if not DASHBOARD_FILE.exists():
        return HTMLResponse(
            "<h2>Missing Adaptive_Inverter_Victory_Showcase_V16_10.html</h2><p>Upload the original V16.10 HTML file to the repository root.</p>",
            status_code=404,
        )
    return FileResponse(DASHBOARD_FILE, media_type="text/html")

# Mount the separate interactive application.
api = gr.mount_gradio_app(api, live_ui, path="/live")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(api, host="0.0.0.0", port=port)
