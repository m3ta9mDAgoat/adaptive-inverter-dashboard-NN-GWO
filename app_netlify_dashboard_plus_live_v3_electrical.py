from __future__ import annotations

import math
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
from fastapi.responses import RedirectResponse
import uvicorn

ROOT = Path(__file__).resolve().parent
APP_FILE = ROOT / "app.py"

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

ALL_METRICS = [
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


def fmt(v, d=2, suffix=""):
    return "N/A" if not np.isfinite(_num(v)) else f"{float(v):.{d}f}{suffix}"


def thermal_state(tj):
    tj = _num(tj)
    if not np.isfinite(tj):
        return "N/A"
    if tj >= 175.0:
        return "FAIL / SHUTDOWN"
    if tj > 150.0:
        return "DERATING REQUIRED"
    return "VALID FULL POWER"


def pass_fail(cond):
    return "PASS" if bool(cond) else "FAIL"


def electrical_case_values(table: pd.DataFrame, col: str):
    v = metric_value(table, "RMS voltage", col)
    thd = metric_value(table, "THD", col)
    irms = metric_value(table, "RMS current", col)
    ripple_a = metric_value(table, "Current ripple", col)
    v_error = abs(v - 220.0) / 220.0 * 100.0 if np.isfinite(v) else np.nan
    ripple_pct = 100.0 * ripple_a / irms if np.isfinite(ripple_a) and np.isfinite(irms) and irms > 0 else np.nan
    ripple_limit_a = 0.20 * irms if np.isfinite(irms) else np.nan
    return {
        "v": v, "v_error": v_error, "thd": thd, "irms": irms,
        "ripple_a": ripple_a, "ripple_pct": ripple_pct,
        "ripple_limit_a": ripple_limit_a,
    }


def full_power_constraint_table(full_table: pd.DataFrame):
    ai = electrical_case_values(full_table, "AI")
    fx = electrical_case_values(full_table, "Fixed")
    ai_t = metric_value(full_table, "Junction temperature", "AI")
    fx_t = metric_value(full_table, "Junction temperature", "Fixed")

    rows = [
        ["RMS output voltage", "215.6–224.4 V (220 V ±2%)", fmt(ai['v'],2," V"), pass_fail(215.6 <= ai['v'] <= 224.4), fmt(fx['v'],2," V"), pass_fail(215.6 <= fx['v'] <= 224.4)],
        ["Voltage error", "≤ 2%", fmt(ai['v_error'],3,"%"), pass_fail(ai['v_error'] <= 2.0), fmt(fx['v_error'],3,"%"), pass_fail(fx['v_error'] <= 2.0)],
        ["Voltage THD", "≤ 5%", fmt(ai['thd'],3,"%"), pass_fail(ai['thd'] <= 5.0), fmt(fx['thd'],3,"%"), pass_fail(fx['thd'] <= 5.0)],
        ["Current ripple — relative RMS", "≤ 20% of Irms", fmt(ai['ripple_pct'],3,"%"), pass_fail(ai['ripple_pct'] <= 20.0), fmt(fx['ripple_pct'],3,"%"), pass_fail(fx['ripple_pct'] <= 20.0)],
        ["Current ripple — absolute RMS", "≤ 0.20 × Irms", f"{fmt(ai['ripple_a'],3,' A')} / limit {fmt(ai['ripple_limit_a'],3,' A')}", pass_fail(ai['ripple_a'] <= ai['ripple_limit_a']), f"{fmt(fx['ripple_a'],3,' A')} / limit {fmt(fx['ripple_limit_a'],3,' A')}", pass_fail(fx['ripple_a'] <= fx['ripple_limit_a'])],
        ["Junction temperature", "≤150°C full-power valid; 175°C shutdown", fmt(ai_t,2,"°C"), thermal_state(ai_t), fmt(fx_t,2,"°C"), thermal_state(fx_t)],
    ]
    return pd.DataFrame(rows, columns=["Constraint", "Limit", "AI value", "AI status", "Fixed value", "Fixed status"])


def relative_life_factor(ai_t, fx_t, ea=0.70):
    ai_t, fx_t = _num(ai_t), _num(fx_t)
    if not (np.isfinite(ai_t) and np.isfinite(fx_t)):
        return np.nan
    if ai_t >= 175 or fx_t >= 175:
        return np.nan
    k = 8.617333262e-5
    tai = ai_t + 273.15
    tfx = fx_t + 273.15
    try:
        return float(math.exp((float(ea)/k) * ((1.0/tai) - (1.0/tfx))))
    except OverflowError:
        return np.inf


def comparison_summary(full_table, sup_table, ea):
    ai_f = metric_value(sup_table, "Switching frequency", "AI")
    fx_f = metric_value(sup_table, "Switching frequency", "Fixed")
    ai_q = metric_value(sup_table, "Delivered requested power", "AI")
    fx_q = metric_value(sup_table, "Delivered requested power", "Fixed")
    ai_t_full = metric_value(full_table, "Junction temperature", "AI")
    fx_t_full = metric_value(full_table, "Junction temperature", "Fixed")
    ai_loss = metric_value(full_table, "MOSFET hot loss", "AI")
    fx_loss = metric_value(full_table, "MOSFET hot loss", "Fixed")
    life = relative_life_factor(ai_t_full, fx_t_full, ea)

    fixed_reason = []
    if np.isfinite(fx_t_full) and fx_t_full > 150:
        fixed_reason.append(f"full-power Tj = {fx_t_full:.1f}°C exceeds the 150°C derating boundary")
    if np.isfinite(fx_q) and fx_q < 99.99:
        fixed_reason.append(f"the supervisor therefore reduced delivered power to {fx_q:.1f}%")
    if fixed_reason:
        story = "; ".join(fixed_reason) + ". Lower post-derating temperature/loss must not be interpreted as better full-power performance."
        story_class = "warn"
    else:
        story = "Both controllers remain at comparable delivered power, so the full-power thermal and loss comparison is directly comparable."
        story_class = "good"

    return f"""
    <div class='decision-grid'>
      <div class='decision-card hero'><span>AI SELECTED FREQUENCY</span><strong>{fmt(ai_f,3,' kHz')}</strong><small>Final GWO + safeguard command</small></div>
      <div class='decision-card'><span>AI FULL-POWER STATE</span><strong class='state'>{thermal_state(ai_t_full)}</strong><small>{fmt(ai_t_full,2,'°C')}</small></div>
      <div class='decision-card'><span>FIXED FULL-POWER STATE</span><strong class='state'>{thermal_state(fx_t_full)}</strong><small>{fmt(fx_t_full,2,'°C')} at 24 kHz</small></div>
      <div class='decision-card'><span>DELIVERED POWER</span><strong>{fmt(ai_q,1,'%')} / {fmt(fx_q,1,'%')}</strong><small>AI / Fixed after supervisor</small></div>
      <div class='decision-card'><span>FULL-POWER ΔT</span><strong>{fmt(fx_t_full-ai_t_full,2,'°C')}</strong><small>Fixed − AI at equal request</small></div>
      <div class='decision-card'><span>FULL-POWER ΔMOSFET</span><strong>{fmt(fx_loss-ai_loss,2,' W')}</strong><small>Fixed − AI at equal request</small></div>
    </div>
    <div class='story {story_class}'><b>What happened:</b> {story}</div>
    <div class='life-note'><b>Relative thermal-aging factor (full-power temperatures):</b> AI/Fixed ≈ {fmt(life,2,'×')}. This is an Arrhenius relative factor, not a claim of absolute service life.</div>
    """


def electrical_constraint_figure(table: pd.DataFrame):
    ai = electrical_case_values(table, "AI")
    fx = electrical_case_values(table, "Fixed")
    labels = ["AI", "Fixed 24 kHz"]
    colors = ["#16b989", "#4f6f93"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "RMS output voltage (V)",
            "Voltage error (%)",
            "Voltage THD (%)",
            "Current ripple — relative RMS (%)",
            "Current ripple — absolute RMS (A)",
            "Electrical-limit utilization (%)",
        ],
        vertical_spacing=0.20, horizontal_spacing=0.10,
    )

    # 1) Absolute voltage and accepted band.
    fig.add_trace(go.Bar(x=labels, y=[ai['v'], fx['v']], text=[fmt(ai['v'],2),fmt(fx['v'],2)], textposition='outside', marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_hrect(y0=215.6, y1=224.4, fillcolor='rgba(13,155,114,.10)', line_width=0, row=1, col=1)
    fig.add_hline(y=220, line_dash='dot', line_color='#0d9b72', annotation_text='220 V target', row=1, col=1)

    # 2) Voltage error.
    fig.add_trace(go.Bar(x=labels, y=[ai['v_error'], fx['v_error']], text=[fmt(ai['v_error'],2),fmt(fx['v_error'],2)], textposition='outside', marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_hline(y=2, line_dash='dash', line_color='#d84949', annotation_text='2% limit', row=1, col=2)

    # 3) THD.
    fig.add_trace(go.Bar(x=labels, y=[ai['thd'], fx['thd']], text=[fmt(ai['thd'],2),fmt(fx['thd'],2)], textposition='outside', marker_color=colors, showlegend=False), row=1, col=3)
    fig.add_hline(y=5, line_dash='dash', line_color='#d84949', annotation_text='5% limit', row=1, col=3)

    # 4) Relative ripple.
    fig.add_trace(go.Bar(x=labels, y=[ai['ripple_pct'], fx['ripple_pct']], text=[fmt(ai['ripple_pct'],2),fmt(fx['ripple_pct'],2)], textposition='outside', marker_color=colors, showlegend=False), row=2, col=1)
    fig.add_hline(y=20, line_dash='dash', line_color='#d84949', annotation_text='20% limit', row=2, col=1)

    # 5) Absolute ripple and exact dynamic ampere limits for this case.
    fig.add_trace(go.Bar(name='Actual ripple', x=labels, y=[ai['ripple_a'], fx['ripple_a']], text=[fmt(ai['ripple_a'],2),fmt(fx['ripple_a'],2)], textposition='outside', marker_color=colors, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(name='Absolute limit', x=labels, y=[ai['ripple_limit_a'], fx['ripple_limit_a']], mode='markers+lines+text', text=[f"limit {fmt(ai['ripple_limit_a'],2)}",f"limit {fmt(fx['ripple_limit_a'],2)}"], textposition='top center', marker=dict(symbol='diamond',size=11,color='#d84949'), line=dict(color='#d84949',dash='dash'), showlegend=False), row=2, col=2)

    # 6) How much of each limit is being used. 100% = exact boundary.
    cats=['Voltage error','THD','Ripple']
    ai_use=[100*ai['v_error']/2.0,100*ai['thd']/5.0,100*ai['ripple_pct']/20.0]
    fx_use=[100*fx['v_error']/2.0,100*fx['thd']/5.0,100*fx['ripple_pct']/20.0]
    fig.add_trace(go.Bar(name='AI',x=cats,y=ai_use,marker_color='#16b989',showlegend=True),row=2,col=3)
    fig.add_trace(go.Bar(name='Fixed',x=cats,y=fx_use,marker_color='#4f6f93',showlegend=True),row=2,col=3)
    fig.add_hline(y=100,line_dash='dash',line_color='#d84949',annotation_text='constraint boundary',row=2,col=3)

    fig.update_layout(height=760, margin=dict(l=38,r=25,t=85,b=45), paper_bgcolor='#fff', plot_bgcolor='#f8fafc', font=dict(family='Inter, Arial', color='#142033'), bargap=.34, barmode='group', legend=dict(orientation='h',y=1.08,x=0))
    fig.update_yaxes(gridcolor='#e7edf5', zeroline=False)
    return fig


def full_power_figure(table: pd.DataFrame):
    items = [
        ("Junction temperature", "Full-power junction temperature (°C)"),
        ("MOSFET hot loss", "Full-power MOSFET loss (W)"),
        ("Electrical efficiency", "Comparable-power efficiency (%)"),
        ("Output power", "Full-power output power (W)"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[x[1] for x in items], vertical_spacing=0.18)
    for i, (metric, _) in enumerate(items):
        r, c = i//2 + 1, i%2 + 1
        ai = metric_value(table, metric, "AI")
        fx = metric_value(table, metric, "Fixed")
        fig.add_trace(go.Bar(x=["AI", "Fixed 24 kHz"], y=[ai, fx], text=[fmt(ai,2), fmt(fx,2)], textposition="outside", marker_color=["#16b989", "#4f6f93"], showlegend=False), row=r, col=c)
        if metric == "Junction temperature":
            fig.add_hline(y=150, line_dash="dash", line_color="#e59b18", annotation_text="150°C derating", row=r, col=c)
            fig.add_hline(y=175, line_dash="dot", line_color="#d84949", annotation_text="175°C shutdown", row=r, col=c)
    fig.update_layout(height=620, margin=dict(l=40,r=25,t=75,b=35), paper_bgcolor="#fff", plot_bgcolor="#f8fafc", font=dict(family="Inter, Arial", color="#142033"), bargap=.38)
    fig.update_yaxes(gridcolor="#e7edf5", zeroline=False)
    return fig


def supervisor_figure(table: pd.DataFrame):
    items = [
        ("Delivered requested power", "Delivered requested power (%)"),
        ("Junction temperature", "Post-supervisor junction temperature (°C)"),
    ]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[x[1] for x in items])
    for i, (metric, _) in enumerate(items, start=1):
        ai = metric_value(table, metric, "AI")
        fx = metric_value(table, metric, "Fixed")
        fig.add_trace(go.Bar(x=["AI", "Fixed 24 kHz"], y=[ai,fx], text=[fmt(ai,2),fmt(fx,2)], textposition="outside", marker_color=["#1769e0", "#7b8da4"], showlegend=False), row=1, col=i)
    fig.update_layout(height=390, margin=dict(l=40,r=25,t=70,b=35), paper_bgcolor="#fff", plot_bgcolor="#f8fafc", font=dict(family="Inter, Arial", color="#142033"), bargap=.40)
    fig.update_yaxes(gridcolor="#e7edf5", zeroline=False)
    return fig


def fair_metrics_table(full_table: pd.DataFrame, sup_table: pd.DataFrame, ea):
    comparable = ["Junction temperature", "MOSFET hot loss", "Electrical efficiency", "THD", "Output power", "RMS current", "RMS voltage", "Current ripple", "Switching frequency"]
    rows=[]
    units={
        "Junction temperature":"°C","MOSFET hot loss":"W","Electrical efficiency":"%","THD":"%",
        "Output power":"W","RMS current":"A","RMS voltage":"V","Current ripple":"A RMS","Switching frequency":"kHz"
    }
    for m in comparable:
        ai=metric_value(full_table,m,"AI"); fx=metric_value(full_table,m,"Fixed")
        rows.append([m, ai, fx, ai-fx if np.isfinite(ai) and np.isfinite(fx) else np.nan, units.get(m,"")])
    ai_e = electrical_case_values(full_table, "AI")
    fx_e = electrical_case_values(full_table, "Fixed")
    rows.append(["Voltage error", ai_e["v_error"], fx_e["v_error"], ai_e["v_error"]-fx_e["v_error"], "%"] )
    rows.append(["Current ripple — relative RMS", ai_e["ripple_pct"], fx_e["ripple_pct"], ai_e["ripple_pct"]-fx_e["ripple_pct"], "% of Irms"] )
    rows.append(["Current ripple — absolute limit", ai_e["ripple_limit_a"], fx_e["ripple_limit_a"], ai_e["ripple_limit_a"]-fx_e["ripple_limit_a"], "A RMS"] )
    aiq=metric_value(sup_table,"Delivered requested power","AI"); fxq=metric_value(sup_table,"Delivered requested power","Fixed")
    rows.append(["Delivered requested power (after supervisor)", aiq, fxq, aiq-fxq, "%"])
    life=relative_life_factor(metric_value(full_table,"Junction temperature","AI"), metric_value(full_table,"Junction temperature","Fixed"), ea)
    rows.append(["Relative thermal-aging factor (full power)", life, 1.0, life-1 if np.isfinite(life) else np.nan, "×"])
    return pd.DataFrame(rows, columns=["Metric", "AI", "Fixed", "AI − Fixed", "Unit"])


def load_preset(name):
    pf, r, ta, ts, note = engine.load_preset(name)
    return pf, r, ta, ts, note


def run_complete_analysis(preset, PF, R, Tamb, study_s, fixed_kHz, ea, progress=gr.Progress()):
    # Two deliberate evaluations:
    # 1) frequency-only = fair equal-request comparison
    # 2) complete supervisor = final service/derating outcome
    requested = ALL_METRICS

    progress(0.01, desc="Stage 1/2 — full-power fair comparison")
    full_status, full_table, full_detail, full_conv, full_candidates = engine.simulate_live_case(
        preset, PF, R, Tamb, study_s, fixed_kHz, "Frequency only",
        requested, "Junction temperature", ea, progress=progress
    )

    progress(0.56, desc="Stage 2/2 — complete supervisor outcome")
    sup_status, sup_table, sup_detail, sup_conv, sup_candidates = engine.simulate_live_case(
        preset, PF, R, Tamb, study_s, fixed_kHz, "Complete supervisor",
        requested, "Delivered requested power", ea, progress=progress
    )

    summary = comparison_summary(full_table, sup_table, ea)
    constraints = full_power_constraint_table(full_table)
    electrical_fig = electrical_constraint_figure(full_table)
    metrics = fair_metrics_table(full_table, sup_table, ea)
    fig_full = full_power_figure(full_table)
    fig_sup = supervisor_figure(sup_table)

    # Candidate evidence is useful but secondary.
    candidates = full_candidates if full_candidates is not None else sup_candidates
    progress(1.0, desc="Complete")
    return summary, constraints, electrical_fig, metrics, fig_full, fig_sup, full_conv, candidates


LIVE_CSS = """
.gradio-container{max-width:1500px!important;margin:auto!important;padding:22px 28px 55px!important;background:#eef3f9!important;font-family:Inter,Segoe UI,Arial,sans-serif!important}
body{background:#eef3f9!important}.topnav{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.backlink{display:inline-block;padding:10px 14px;border-radius:10px;background:#0b1f3a;color:#fff!important;text-decoration:none!important;font-weight:800}.live-head{padding:30px 32px;border-radius:24px;background:linear-gradient(125deg,#071b34,#0e5276);color:#fff;box-shadow:0 18px 50px rgba(7,27,52,.18);margin-bottom:18px}.live-head h1{margin:0 0 8px;font-size:36px}.live-head p{margin:0;color:#d9ebf7;font-size:15px;line-height:1.6}.panel{background:#fff;border:1px solid #dbe4ef;border-radius:20px;padding:18px!important;box-shadow:0 10px 32px rgba(20,48,80,.08)}
.decision-grid{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:12px;margin:14px 0}.decision-card{background:#fff;border:1px solid #dbe4ef;border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(20,48,80,.06)}.decision-card.hero{background:linear-gradient(135deg,#071b34,#0e5276);color:white;border:none}.decision-card span{display:block;font-size:10px;font-weight:850;letter-spacing:.05em;color:#68788e}.decision-card.hero span,.decision-card.hero small{color:#d8eaf6}.decision-card strong{display:block;font-size:22px;margin:8px 0 5px;color:#102038}.decision-card.hero strong{color:white}.decision-card strong.state{font-size:14px;line-height:1.25}.decision-card small{color:#738196}.story{padding:15px 17px;border-radius:14px;margin:12px 0;line-height:1.55}.story.warn{background:#fff3dc;color:#80510d;border:1px solid #efd298}.story.good{background:#e8f8f1;color:#096348;border:1px solid #bce8d5}.life-note{padding:13px 16px;border-radius:13px;background:#edf5ff;color:#174b7a;border:1px solid #cce2fa;margin-bottom:15px}
#run-btn{background:#0b1f3a!important;color:white!important;font-weight:850!important;min-height:50px!important;border-radius:12px!important}footer{display:none!important}@media(max-width:1050px){.decision-grid{grid-template-columns:1fr 1fr 1fr}}@media(max-width:720px){.decision-grid{grid-template-columns:1fr 1fr}}
"""


def build_live_ui():
    with gr.Blocks(title="Adaptive Inverter — Live Interactive", css=LIVE_CSS) as live:
        gr.HTML("<div class='topnav'><a class='backlink' href='https://neural-network-gwo-adaptive-inverter.netlify.app/' target='_blank' rel='noopener'>← V16.10 Dashboard</a><b>Live Interactive</b></div>")
        gr.HTML("""<div class='live-head'><h1>Live Interactive Case Lab</h1><p>Fresh ANN + continuous GWO + supervisor evaluation. The page separates equal-power performance from post-derating outcomes so the comparison remains physically fair.</p></div>""")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel"]):
                preset = gr.Dropdown(PRESETS, value=PRESETS[0] if PRESETS else None, label="Important case / preset")
                load_btn = gr.Button("Load preset")
                preset_note = gr.Markdown("")
                pf = gr.Slider(0.60,0.99,value=0.70,step=0.001,label="Power factor")
                r = gr.Slider(5.0,120.0,value=10.0,step=0.01,label="Resistance (Ω)")
                ambient = gr.Slider(15,55,value=25,step=1,label="Ambient temperature (°C)")
                study = gr.Slider(10,600,value=100,step=10,label="Thermal study time (s)")
                fixed = gr.Slider(6,24,value=24,step=1,label="Fixed reference frequency (kHz)")
                ea = gr.Slider(0.30,1.10,value=0.70,step=0.05,label="Arrhenius activation energy Ea (eV)")
                run = gr.Button("Run full live analysis", elem_id="run-btn")
            with gr.Column(scale=2):
                summary = gr.HTML()

        gr.Markdown("## 1. Full-power constraint check")
        gr.Markdown("Both controllers are evaluated at the same requested operating point. This is where electrical and thermal feasibility must be judged.")
        constraints = gr.Dataframe(interactive=False, wrap=True)
        gr.Markdown("### Electrical limits — graphical view")
        gr.Markdown("The red lines are the actual hard limits used in electrical feasibility. Relative and absolute current-ripple RMS are shown together; the absolute ampere limit is calculated from 20% of the case RMS current.")
        electrical_plot = gr.Plot()

        gr.Markdown("## 2. Fair full-power comparison")
        gr.Markdown("Temperature, MOSFET loss, efficiency and THD are compared before derating, at equal requested power.")
        full_plot = gr.Plot()
        metrics_table = gr.Dataframe(interactive=False, wrap=True, label="Comparable metrics + final delivered power")

        gr.Markdown("## 3. Supervisor outcome")
        gr.Markdown("This section shows what each controller finally delivers after protection/derating. A lower post-derating temperature is not treated as a full-power performance win.")
        sup_plot = gr.Plot()

        with gr.Accordion("Engineering evidence — GWO convergence and candidates", open=False):
            convergence = gr.Plot(label="GWO convergence")
            candidates = gr.Dataframe(label="Candidate evidence", interactive=False, wrap=True)

        load_btn.click(load_preset, inputs=[preset], outputs=[pf,r,ambient,study,preset_note])
        run.click(run_complete_analysis, inputs=[preset,pf,r,ambient,study,fixed,ea], outputs=[summary,constraints,electrical_plot,metrics_table,full_plot,sup_plot,convergence,candidates])
    return live


live_ui = build_live_ui()
api = FastAPI(title="Adaptive Inverter V16.10 + Live")

FLOAT_NAV = """
<style>
#v1610-live-nav{position:fixed;top:12px;right:16px;z-index:2147483647;display:flex;gap:8px;font-family:Inter,Segoe UI,Arial,sans-serif}
#v1610-live-nav a{display:inline-block;padding:10px 14px;border-radius:11px;background:#0b1f3a;color:#fff;text-decoration:none;font-weight:800;box-shadow:0 8px 24px rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.18)}
</style>
<div id="v1610-live-nav"><a href="/live/">Live Interactive →</a></div>
"""

@api.get("/")
def root():
    return RedirectResponse(url="https://neural-network-gwo-adaptive-inverter.netlify.app/", status_code=302)

@api.get("/dashboard")
def dashboard():
    return RedirectResponse(url="https://neural-network-gwo-adaptive-inverter.netlify.app/", status_code=302)

api = gr.mount_gradio_app(api, live_ui, path="/live")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(api, host="0.0.0.0", port=port)
