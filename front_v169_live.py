from __future__ import annotations

import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Keep the current V16.12 computational engine untouched.
import app as engine

# -----------------------------------------------------------------------------
# Executive V16.9-style presentation layer
# -----------------------------------------------------------------------------

CORE_METRICS = [
    "Junction temperature",
    "MOSFET hot loss",
    "Electrical efficiency",
    "Delivered requested power",
    "Switching frequency",
    "THD",
]

DEFAULT_DETAIL_METRICS = [
    "Junction temperature",
    "MOSFET hot loss",
    "Electrical efficiency",
    "Delivered requested power",
    "THD",
    "Relative thermal lifetime",
]

PRESET_ORDER = [
    "V09 — headline thermal benefit",
    "V11 — thermal boundary / rescue",
    "V13 — complete-supervisor derating",
    "V06 — continuous GWO interpolation challenge",
    "V04 — high-load transition / 14 kHz",
    "V01 — light load / AI stays at 24 kHz",
    "Validation — 10 Ω high-PF reference",
    "Custom",
]
PRESET_CHOICES = [p for p in PRESET_ORDER if p in engine.CASE_PRESETS]

MC_BAND = pd.read_csv(engine.MC_BAND_PATH) if engine.MC_BAND_PATH.exists() else pd.DataFrame()
MC_PAIR = pd.read_csv(engine.MC_SUMMARY_PATH) if engine.MC_SUMMARY_PATH.exists() else pd.DataFrame()

V169_CSS = r"""
:root{
  --ink:#0b172a; --muted:#5e6c84; --panel:#ffffff; --bg:#eef3f9;
  --line:#d9e2ef; --navy:#0b1f3a; --blue:#1769e0; --cyan:#00a6c8;
  --green:#0d9b72; --amber:#e59b18; --red:#d84949; --purple:#6d55d9;
  --shadow:0 14px 40px rgba(18,42,75,.10);
}
body, .gradio-container{background:var(--bg)!important;color:var(--ink)!important}
.gradio-container{max-width:1540px!important;margin:auto!important;padding:18px 22px 50px!important;font-family:Inter,Segoe UI,Arial,sans-serif!important}
#hero-v169{
  position:relative;overflow:hidden;border-radius:28px;padding:38px 42px;color:white;
  background:linear-gradient(125deg,rgba(6,26,52,.98),rgba(10,73,109,.96));
  box-shadow:0 24px 70px rgba(6,26,52,.24);margin-bottom:20px
}
#hero-v169:after{content:"";position:absolute;width:430px;height:430px;right:-150px;top:-210px;border-radius:50%;border:68px solid rgba(255,255,255,.055)}
#hero-v169 .eyebrow{display:inline-flex;padding:7px 12px;border:1px solid rgba(255,255,255,.28);border-radius:999px;font-size:12px;font-weight:750;letter-spacing:.08em;text-transform:uppercase;color:#dff7ff}
#hero-v169 h1{font-size:clamp(30px,4.2vw,58px);line-height:1.03;margin:20px 0 12px;max-width:1050px}
#hero-v169 p{font-size:17px;line-height:1.65;color:#d9eaf8;max-width:1040px;margin:0}
#hero-v169 .proof{margin-top:24px;padding:15px 18px;border-left:4px solid #45d6ff;background:rgba(255,255,255,.075);border-radius:0 14px 14px 0;max-width:1050px;font-weight:650;line-height:1.55}
.exec-section{background:white;border:1px solid rgba(216,226,239,.75);border-radius:23px;padding:24px;box-shadow:var(--shadow);margin:18px 0}
.exec-section h2{margin:0;font-size:25px}.exec-sub{color:var(--muted);line-height:1.55;margin-top:6px;max-width:980px}
.kpi-grid-v169{display:grid;grid-template-columns:repeat(6,minmax(150px,1fr));gap:16px;margin:16px 0}
.kpi-v169{background:white;border-radius:18px;padding:19px 18px;border-top:5px solid var(--blue);box-shadow:var(--shadow);min-height:142px}
.kpi-v169.green{border-color:var(--green)}.kpi-v169.cyan{border-color:var(--cyan)}.kpi-v169.amber{border-color:var(--amber)}.kpi-v169.purple{border-color:var(--purple)}.kpi-v169.red{border-color:var(--red)}
.kpi-label-v169{font-size:11px;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.055em}
.kpi-value-v169{font-size:29px;font-weight:850;margin:11px 0 7px;white-space:nowrap}.kpi-note-v169{font-size:12px;color:var(--muted);line-height:1.4}
.decision-grid{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:14px;margin:14px 0}
.decision-card{background:#fff;border:1px solid var(--line);border-radius:17px;padding:17px 16px;box-shadow:0 8px 24px rgba(18,42,75,.07);min-height:125px}
.decision-card.primary{background:linear-gradient(135deg,#071b34,#0e4d70);color:white;border:0}.decision-card.primary .dc-label,.decision-card.primary .dc-note{color:#d6e8f6}.decision-card.good{border-top:5px solid var(--green)}.decision-card.warn{border-top:5px solid var(--amber)}.decision-card.red{border-top:5px solid var(--red)}.decision-card.blue{border-top:5px solid var(--blue)}
.dc-label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:800}.dc-value{font-size:27px;font-weight:850;margin:10px 0 6px}.dc-note{font-size:11px;color:var(--muted);line-height:1.35}
.action-banner{border-radius:17px;padding:15px 18px;margin:14px 0;font-weight:750;line-height:1.5}.action-banner.active{background:#eaf8f2;border:1px solid #b9e4d3;color:#075f46}.action-banner.same{background:#f2f8ff;border:1px solid #d3e8ff;color:#063d77}.action-banner.limit{background:#fff4de;border:1px solid #f0d29b;color:#81500b}
#live-controls{background:rgba(255,255,255,.88)!important;border:1px solid rgba(255,255,255,.9)!important;border-radius:18px!important;padding:14px 16px!important;box-shadow:var(--shadow)!important}
#run-live-v169{background:var(--navy)!important;color:white!important;border-radius:12px!important;font-weight:800!important;font-size:15px!important;min-height:46px!important}
#load-preset-v169{border-radius:11px!important}
.compact-table table{font-size:13px!important}.compact-table thead th{background:var(--navy)!important;color:white!important}
.gr-accordion{border:1px solid var(--line)!important;border-radius:15px!important;background:white!important}
footer{display:none!important}
@media(max-width:1200px){.kpi-grid-v169{grid-template-columns:repeat(3,1fr)}.decision-grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){.gradio-container{padding:10px!important}#hero-v169{padding:28px 24px}.kpi-grid-v169,.decision-grid{grid-template-columns:1fr}}
"""


def fmt(v, digits=2, suffix=""):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def metric_lookup(table: pd.DataFrame, metric: str, column: str):
    if table is None or len(table) == 0:
        return np.nan
    row = table[table["Metric"].astype(str).eq(metric)]
    if row.empty or column not in row.columns:
        return np.nan
    try:
        return float(row.iloc[0][column])
    except Exception:
        return np.nan


def parse_status(status_html: str):
    text = re.sub(r"<[^>]+>", " ", status_html or "")
    text = re.sub(r"\s+", " ", text).strip()
    ai_state = "N/A"
    fx_state = "N/A"
    m = re.search(r"AI:\s*([^—]+)—", text)
    if m:
        ai_state = m.group(1).strip()
    m = re.search(r"Fixed:\s*([^—]+)—", text)
    if m:
        fx_state = m.group(1).strip()
    return ai_state, fx_state


def decision_html(status_html, table):
    ai_state, fx_state = parse_status(status_html)
    ai_f = metric_lookup(table, "Switching frequency", "AI")
    fx_f = metric_lookup(table, "Switching frequency", "Fixed")
    ai_t = metric_lookup(table, "Junction temperature", "AI")
    fx_t = metric_lookup(table, "Junction temperature", "Fixed")
    ai_loss = metric_lookup(table, "MOSFET hot loss", "AI")
    fx_loss = metric_lookup(table, "MOSFET hot loss", "Fixed")
    ai_q = metric_lookup(table, "Delivered requested power", "AI")
    fx_q = metric_lookup(table, "Delivered requested power", "Fixed")

    t_saved = fx_t - ai_t if np.isfinite(ai_t) and np.isfinite(fx_t) else np.nan
    loss_saved = fx_loss - ai_loss if np.isfinite(ai_loss) and np.isfinite(fx_loss) else np.nan

    if np.isfinite(ai_f) and np.isfinite(fx_f):
        if abs(ai_f - fx_f) < 0.01:
            banner_cls = "same"
            banner = f"No adaptation required — the AI selected the same {ai_f:.2f} kHz operating point as the reference for this case."
        else:
            banner_cls = "active"
            banner = f"Adaptive action taken — switching frequency changed from {fx_f:.2f} kHz reference to {ai_f:.2f} kHz."
    else:
        banner_cls = "limit"
        banner = "Protection action detected — inspect the operating-state and delivered-power cards below."

    state_class = "good" if "VALID" in ai_state else ("warn" if "DERAT" in ai_state else "red")
    return f"""
    <div class='action-banner {banner_cls}'>{banner}</div>
    <div class='decision-grid'>
      <div class='decision-card primary'><div class='dc-label'>AI selected frequency</div><div class='dc-value'>{fmt(ai_f,3,' kHz')}</div><div class='dc-note'>Final commanded frequency after GWO + safeguards.</div></div>
      <div class='decision-card {state_class}'><div class='dc-label'>AI operating state</div><div class='dc-value' style='font-size:19px'>{ai_state.replace('_',' ')}</div><div class='dc-note'>150°C derating / 175°C protection policy.</div></div>
      <div class='decision-card red'><div class='dc-label'>Temperature saved</div><div class='dc-value'>{fmt(t_saved,2,'°C')}</div><div class='dc-note'>Fixed minus AI at the same requested case.</div></div>
      <div class='decision-card amber'><div class='dc-label'>MOSFET loss saved</div><div class='dc-value'>{fmt(loss_saved,2,' W')}</div><div class='dc-note'>Positive means lower hot MOSFET loss with AI.</div></div>
      <div class='decision-card blue'><div class='dc-label'>Delivered power</div><div class='dc-value'>{fmt(ai_q,1,'%')}</div><div class='dc-note'>Fixed reference: {fmt(fx_q,1,'%')}.</div></div>
    </div>
    """


def filter_metric_table(table, requested):
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=["Metric", "AI", "Fixed", "AI − Fixed", "Unit"])
    wanted = list(dict.fromkeys(requested or DEFAULT_DETAIL_METRICS))
    return table[table["Metric"].isin(wanted)].reset_index(drop=True)


def run_live_executive(preset_name, PF, R, Tamb, time_s, fixed_kHz, mode, detail_metrics, plot_metric, Ea, progress=gr.Progress()):
    # Always calculate the executive decision metrics, even if the user hides them.
    requested = list(dict.fromkeys(CORE_METRICS + list(detail_metrics or []) + [plot_metric]))
    status, table, fig, conv, candidates = engine.simulate_live_case(
        preset_name, PF, R, Tamb, time_s, fixed_kHz, mode,
        requested, plot_metric, Ea, progress=progress,
    )
    summary = decision_html(status, table)
    visible_table = filter_metric_table(table, detail_metrics)
    return summary, visible_table, fig, conv, candidates, status


def load_preset_exec(name):
    pf, r, ta, ts, note = engine.load_preset(name)
    return pf, r, ta, ts, note


def mc_numbers(load_band):
    if MC_BAND.empty:
        return {}
    if load_band == "All loads":
        scenarios = int(MC_BAND["scenarios"].sum())
        weights = MC_BAND["scenarios"].astype(float)
        w = weights / weights.sum()
        return {
            "scenarios": scenarios,
            "ai_service": float(np.sum(w * MC_BAND["AI_service_pct"])),
            "fx_service": float(np.sum(w * MC_BAND["Fixed_service_pct"])),
            "ai_full": float(np.sum(w * MC_BAND["AI_full_power_pct"])),
            "fx_full": float(np.sum(w * MC_BAND["Fixed_full_power_pct"])),
            "ai_energy": float(np.sum(w * MC_BAND["AI_energy_fulfillment_pct"])),
            "fx_energy": float(np.sum(w * MC_BAND["Fixed_energy_fulfillment_pct"])),
            "rescue": int(MC_BAND["AI_rescue_cases"].sum()),
        }
    row = MC_BAND[MC_BAND["load_band"].astype(str).eq(load_band)]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "scenarios": int(r["scenarios"]), "ai_service": float(r["AI_service_pct"]), "fx_service": float(r["Fixed_service_pct"]),
        "ai_full": float(r["AI_full_power_pct"]), "fx_full": float(r["Fixed_full_power_pct"]),
        "ai_energy": float(r["AI_energy_fulfillment_pct"]), "fx_energy": float(r["Fixed_energy_fulfillment_pct"]),
        "rescue": int(r["AI_rescue_cases"]),
        "loss": float(r["paired_Pmos_reduction_W"]), "temp": float(r["paired_temperature_saved_C"]),
    }


def paired_metric(metric_name, population="Heavy-load comparable scenarios"):
    if MC_PAIR.empty:
        return np.nan
    d = MC_PAIR[(MC_PAIR["population"].astype(str) == population) & (MC_PAIR["metric"].astype(str) == metric_name)]
    return float(d.iloc[0]["mean"]) if not d.empty else np.nan


def mc_kpi_html(load_band):
    n = mc_numbers(load_band)
    if not n:
        return "<div class='exec-section'>Monte Carlo summary unavailable.</div>"
    if load_band == "All loads":
        temp = paired_metric("Temperature saved (°C)", "Heavy-load comparable scenarios")
        loss = paired_metric("MOSFET loss reduction (W)", "Heavy-load comparable scenarios")
        scope_note = "300 paired scenarios"
    else:
        temp = n.get("temp", np.nan); loss = n.get("loss", np.nan); scope_note = f"{n['scenarios']} scenarios"
    return f"""
    <div class='kpi-grid-v169'>
      <div class='kpi-v169 green'><div class='kpi-label-v169'>AI service</div><div class='kpi-value-v169'>{n['ai_service']:.1f}%</div><div class='kpi-note-v169'>Fixed: {n['fx_service']:.1f}% • {scope_note}</div></div>
      <div class='kpi-v169 cyan'><div class='kpi-label-v169'>AI full power</div><div class='kpi-value-v169'>{n['ai_full']:.1f}%</div><div class='kpi-note-v169'>Fixed: {n['fx_full']:.1f}%</div></div>
      <div class='kpi-v169 purple'><div class='kpi-label-v169'>Energy fulfillment</div><div class='kpi-value-v169'>{n['ai_energy']:.1f}%</div><div class='kpi-note-v169'>Fixed: {n['fx_energy']:.1f}%</div></div>
      <div class='kpi-v169 red'><div class='kpi-label-v169'>Temperature saved</div><div class='kpi-value-v169'>{fmt(temp,2,'°C')}</div><div class='kpi-note-v169'>Comparable heavy-load operation</div></div>
      <div class='kpi-v169 amber'><div class='kpi-label-v169'>MOSFET loss saved</div><div class='kpi-value-v169'>{fmt(loss,2,' W')}</div><div class='kpi-note-v169'>Comparable heavy-load operation</div></div>
      <div class='kpi-v169'><div class='kpi-label-v169'>AI rescue cases</div><div class='kpi-value-v169'>{n['rescue']}</div><div class='kpi-note-v169'>Fixed failure while AI kept service</div></div>
    </div>
    """


def mc_plots(load_band):
    n = mc_numbers(load_band)
    if not n:
        return go.Figure(), go.Figure()
    categories = ["Service", "Full power", "Energy fulfillment"]
    ai = [n["ai_service"], n["ai_full"], n["ai_energy"]]
    fx = [n["fx_service"], n["fx_full"], n["fx_energy"]]
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name="AI adaptive", x=categories, y=ai, marker_color="#0d9b72", text=[f"{v:.1f}%" for v in ai], textposition="outside"))
    fig1.add_trace(go.Bar(name="Fixed 24 kHz", x=categories, y=fx, marker_color="#1769e0", text=[f"{v:.1f}%" for v in fx], textposition="outside"))
    fig1.update_layout(barmode="group", template="plotly_white", height=390, margin=dict(l=45,r=20,t=45,b=50), yaxis=dict(title="Percent", range=[0,108]), legend=dict(orientation="h", y=1.12), title=f"Operational outcomes — {load_band}")

    fig2 = go.Figure()
    if load_band == "All loads":
        bands = MC_BAND["load_band"].astype(str).tolist()
        temps = MC_BAND["paired_temperature_saved_C"].astype(float).tolist()
        losses = MC_BAND["paired_Pmos_reduction_W"].astype(float).tolist()
    else:
        bands = [load_band]
        temps = [n.get("temp", np.nan)]
        losses = [n.get("loss", np.nan)]
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=bands, y=temps, name="Temperature saved (°C)", marker_color="#d84949"), secondary_y=False)
    fig2.add_trace(go.Scatter(x=bands, y=losses, name="MOSFET loss saved (W)", mode="lines+markers", marker_color="#e59b18", line=dict(width=3)), secondary_y=True)
    fig2.update_yaxes(title_text="Temperature saved (°C)", secondary_y=False)
    fig2.update_yaxes(title_text="MOSFET loss saved (W)", secondary_y=True)
    fig2.update_layout(template="plotly_white", height=390, margin=dict(l=55,r=55,t=45,b=70), legend=dict(orientation="h", y=1.12), title="Thermal benefit by load band")
    return fig1, fig2


def update_mc(load_band):
    a,b = mc_plots(load_band)
    return mc_kpi_html(load_band), a, b


def build_app():
    default_preset = "V09 — headline thermal benefit" if "V09 — headline thermal benefit" in engine.CASE_PRESETS else PRESET_CHOICES[0]
    p0 = engine.CASE_PRESETS[default_preset]

    with gr.Blocks(title="Adaptive Inverter — Executive Results + Live Case", css=V169_CSS) as demo:
        gr.HTML("""
        <div id='hero-v169'>
          <span class='eyebrow'>Validated R5–120 domain · ANN + GWO supervisor</span>
          <h1>Adaptive Switching-Frequency Results Dashboard</h1>
          <p>A clean executive view of the final Monte Carlo evidence, with one added live feature: run any operating case through the real V16.8 ANN + Grey Wolf Optimization backend.</p>
          <div class='proof'>Main result: the controller preserves the high-frequency solution when it is already best, then reduces switching frequency under demanding load to lower semiconductor thermal stress and preserve useful power delivery.</div>
        </div>
        """)

        gr.HTML("<div class='exec-section'><h2>Monte Carlo results</h2><div class='exec-sub'>The same paired evidence used in the final analysis. Choose a load band to simplify the story; no live recalculation is performed in this overview.</div></div>")
        with gr.Row(elem_id="live-controls"):
            mc_band_choice = gr.Dropdown(["All loads"] + engine.STRESS_BAND_ORDER, value="All loads", label="Load view", scale=2)
        mc_kpis = gr.HTML(mc_kpi_html("All loads"))
        mc_fig1, mc_fig2 = mc_plots("All loads")
        with gr.Row():
            mc_plot1 = gr.Plot(mc_fig1, show_label=False)
            mc_plot2 = gr.Plot(mc_fig2, show_label=False)
        mc_band_choice.change(update_mc, [mc_band_choice], [mc_kpis, mc_plot1, mc_plot2])

        gr.HTML("<div class='exec-section'><h2>Live Case Lab</h2><div class='exec-sub'>Choose a reviewed case or move the sliders. Press Run once. The backend then executes the saved ANN, full GWO search, safeguards, supervisor, fixed-reference comparison, and electrothermal calculation. The progress indicator follows the real computation.</div></div>")

        with gr.Group(elem_id="live-controls"):
            with gr.Row():
                preset = gr.Dropdown(PRESET_CHOICES, value=default_preset, label="Important case", scale=4)
                load_btn = gr.Button("Load case", elem_id="load-preset-v169", scale=1)
            preset_note = gr.Markdown(f"**Why this case:** {p0['purpose']}")
            with gr.Row():
                pf = gr.Slider(engine.PF_MIN, engine.PF_MAX, value=p0["PF"], step=.001, label="Power factor")
                r = gr.Slider(engine.TRAIN_R_MIN, engine.TRAIN_R_MAX, value=p0["R_ohm"], step=.05, label="Load resistance (Ω)")
                t = gr.Slider(30, 1200, value=p0["time_s"], step=10, label="Thermal study time (s)")
                ta = gr.Slider(20, 50, value=p0["Tamb_C"], step=1, label="Ambient (°C)")

            with gr.Accordion("Customize comparison and metrics", open=False):
                with gr.Row():
                    fixed = gr.Dropdown([f/1000 for f in engine.ALLOWED_MEASURED_FREQUENCIES_HZ], value=engine.BEST_GLOBAL_FIXED_FSW/1000, label="Fixed reference (kHz)")
                    mode = gr.Radio(["Frequency only", "Complete supervisor"], value="Complete supervisor", label="Comparison mode")
                    ea = gr.Slider(.30,1.10,value=.70,step=.05,label="Activation energy Ea (eV)")
                with gr.Row():
                    details = gr.CheckboxGroup(list(engine.METRIC_DEFS.keys()), value=DEFAULT_DETAIL_METRICS, label="Metrics shown in details")
                    plot_metric = gr.Dropdown(list(engine.METRIC_DEFS.keys()), value="Junction temperature", label="Main comparison chart")

            run_btn = gr.Button("Run live case", variant="primary", elem_id="run-live-v169")

        live_summary = gr.HTML()
        with gr.Row():
            metric_plot = gr.Plot(label="Selected comparison")
            metric_table = gr.Dataframe(label="Selected metrics", interactive=False, elem_classes=["compact-table"])

        with gr.Accordion("Engineering evidence — GWO convergence and candidates", open=False):
            convergence = gr.Plot(label="GWO convergence")
            candidates = gr.Dataframe(label="Full-power candidate evidence", interactive=False)
            raw_status = gr.HTML(visible=False)

        with gr.Accordion("Manual two-frequency comparison", open=False):
            gr.Markdown("Direct ANN-electrothermal comparison without GWO — useful for the 50-second validation-style check.")
            with gr.Row():
                mpf = gr.Slider(engine.PF_MIN, engine.PF_MAX, value=.99, step=.001, label="Power factor")
                mr = gr.Slider(engine.TRAIN_R_MIN, engine.TRAIN_R_MAX, value=10, step=.05, label="Resistance (Ω)")
                mt = gr.Slider(30, 1200, value=50, step=10, label="Study time (s)")
                mta = gr.Slider(20, 50, value=25, step=1, label="Ambient (°C)")
            with gr.Row():
                mf1 = gr.Slider(engine.MIN_VALIDATED_COMMAND_FSW_HZ/1000, engine.MAX_COMMAND_FSW_HZ/1000, value=12, step=.1, label="Frequency A (kHz)")
                mf2 = gr.Slider(engine.MIN_VALIDATED_COMMAND_FSW_HZ/1000, engine.MAX_COMMAND_FSW_HZ/1000, value=24, step=.1, label="Frequency B (kHz)")
                mmetric = gr.Dropdown(list(engine.METRIC_DEFS.keys()), value="Junction temperature", label="Metric to plot")
            mrun = gr.Button("Compare frequencies")
            mstatus = gr.HTML()
            with gr.Row():
                mtable = gr.Dataframe(interactive=False, label="Manual comparison")
                mplot = gr.Plot(label="Manual comparison chart")

        load_btn.click(load_preset_exec, [preset], [pf, r, ta, t, preset_note])
        run_btn.click(
            run_live_executive,
            [preset, pf, r, ta, t, fixed, mode, details, plot_metric, ea],
            [live_summary, metric_table, metric_plot, convergence, candidates, raw_status],
        )
        mrun.click(
            engine.manual_frequency_compare,
            [mpf, mr, mta, mt, mf1, mf2, details, mmetric, ea],
            [mstatus, mtable, mplot],
        )

    return demo


APP = build_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    APP.queue().launch(server_name="0.0.0.0", server_port=port, show_error=True)
