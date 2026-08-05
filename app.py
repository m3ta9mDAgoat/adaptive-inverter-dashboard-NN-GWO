from __future__ import annotations

import math
import time
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "adaptive_inverter_v168_model.joblib"
DATASET_PATH = BASE_DIR / "normal_R5_120_full_grid_17_18_19k.csv"
MC_SUMMARY_PATH = BASE_DIR / "monte_carlo_default_paired_summary.csv"
MC_BAND_PATH = BASE_DIR / "monte_carlo_default_by_band.csv"

# -----------------------------------------------------------------------------
# V16.8 engineering configuration — kept consistent with the final workflow.
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
F_GRID_HZ = 50.0
VOUT_TARGET_RMS = 220.0
FSW_DATA_MIN_HZ = 2000.0
FSW_DATA_MAX_HZ = 24000.0
MEASURED_FREQUENCIES_HZ = [
    2000, 6000, 10000, 12000, 14000, 16000,
    17000, 18000, 19000, 20000, 22000, 24000,
]
TRAIN_R_MIN = 5.0
TRAIN_R_MAX = 120.0
PF_MIN = 0.60
PF_MAX = 0.99
RATED_POWER_W = 5000.0
RATED_CURRENT_A = 30.0
LIMIT_VOUT_ERROR_PCT = 2.0
LIMIT_THD_PCT = 5.0
LIMIT_RIPPLE_PCT = 20.0
LIMIT_RIPPLE_ABSOLUTE_A = 1.50
FILTER_L_H = 1e-3
FILTER_C_F = 30e-6
FILTER_RESONANCE_HZ = 1.0 / (2 * np.pi * np.sqrt(FILTER_L_H * FILTER_C_F))
THEORETICAL_MIN_STABLE_FSW_HZ = 2.5 * FILTER_RESONANCE_HZ
ALLOWED_MEASURED_FREQUENCIES_HZ = [
    float(f) for f in MEASURED_FREQUENCIES_HZ
    if f >= THEORETICAL_MIN_STABLE_FSW_HZ
]
MIN_VALIDATED_COMMAND_FSW_HZ = min(ALLOWED_MEASURED_FREQUENCIES_HZ)
MAX_COMMAND_FSW_HZ = max(ALLOWED_MEASURED_FREQUENCIES_HZ)
T_REF_C = 25.0
T_AMB_DEFAULT = 25.0
RTH_JH = 0.55
TAU_JH = 0.08
RTH_HA_NOM = 0.80
RTH_HA_HIGH = 1.00
CTH_HA = 250.0
ALPHA_COND_PER_C = ((75.0 / 40.0) - 1.0) / (150.0 - 25.0)
ALPHA_SW_PER_C = ((1.93 / 1.41) - 1.0) / (150.0 - 25.0)
THERMAL_MAX_ITER = 80
THERMAL_TOL_C = 1e-4
THERMAL_DAMPING = 0.45
T_VALID_MAX_C = 150.0
T_FAILURE_C = 175.0
AI_MIN_POWER_FRACTION = 0.40
AI_POWER_STEP = 0.02
DELIVERED_POWER_TIE = 0.01
FIXED_DERATING_MIN_FRACTION = 0.40
LIGHT_EFFICIENCY_TIE_PP = 0.03
MODERATE_EFFICIENCY_TIE_PP = 0.10
HEAVY_EFFICIENCY_TOLERANCE_PP = 0.25
GWO_AGENTS = 24
GWO_ITERS = 40
GWO_REPEATS = 2

STRESS_BAND_ORDER = [
    "Very light ≤25%",
    "Moderate 25–60%",
    "High 60–85%",
    "Near-rated >85%",
]

# Relative temperature-accelerated aging. This is deliberately reported as a
# multiplier, not a claim of absolute years of life.
BOLTZMANN_EV_K = 8.617333262145e-5

# Important points from the final V16.8 analysis + the 50-s validation point.
CASE_PRESETS = {
    "Custom": None,
    "Validation — 10 Ω high-PF reference": {
        "PF": 0.99, "R_ohm": 10.0, "Tamb_C": 25.0, "time_s": 50.0,
        "purpose": "Recreates the neighborhood of the 50-second thermal validation point."
    },
    "V01 — light load / AI stays at 24 kHz": {
        "PF": 0.6506693870808339, "R_ohm": 41.54228925057913,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "Checks that the controller does not change frequency when 24 kHz is already preferred."
    },
    "V04 — high-load transition / 14 kHz": {
        "PF": 0.8266354775961520, "R_ohm": 10.924513666923286,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "Shows the transition away from 24 kHz as thermal priority becomes important."
    },
    "V06 — continuous GWO interpolation challenge": {
        "PF": 0.9773287956493420, "R_ohm": 13.698200126682368,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "A continuous-frequency case where the final study selected about 8.089 kHz."
    },
    "V09 — headline thermal benefit": {
        "PF": 0.6998263853213651, "R_ohm": 5.586774892566766,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "The strongest full-power thermal and MOSFET-loss benefit case in the reviewed set."
    },
    "V11 — thermal boundary / rescue": {
        "PF": 0.7512582611138234, "R_ohm": 5.794887282366681,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "A boundary case: AI remains near the 150°C policy boundary while fixed 24 kHz is predicted beyond protection."
    },
    "V13 — complete-supervisor derating": {
        "PF": 0.7704892483602089, "R_ohm": 5.836350596814001,
        "Tamb_C": 25.0, "time_s": 600.0,
        "purpose": "Exercises the frequency-first policy and the conditional AI power search."
    },
}

METRIC_DEFS = {
    "Junction temperature": ("Tj_upper_C", "°C", "lower"),
    "MOSFET hot loss": ("Pmos_hot_W", "W", "lower"),
    "Total hot loss": ("Ptotal_hot_loss_W", "W", "lower"),
    "Electrical efficiency": ("eta_inverter_electrothermal_pct", "%", "higher"),
    "Delivered requested power": ("delivered_power_pct", "%", "higher"),
    "Switching frequency": ("commanded_fsw_Hz", "kHz", "context"),
    "THD": ("THD_target_pct", "%", "lower"),
    "Output power": ("Pac_W", "W", "context"),
    "RMS current": ("Irms_A", "A", "context"),
    "RMS voltage": ("Vout_fund_rms_V", "V", "context"),
    "Current ripple": ("IL_ripple_rms_A", "A RMS", "lower"),
    "Relative thermal lifetime": ("relative_lifetime", "×", "higher"),
}

DEFAULT_METRICS = [
    "Junction temperature", "MOSFET hot loss", "Electrical efficiency",
    "Delivered requested power", "Switching frequency", "THD",
    "Relative thermal lifetime",
]

# -----------------------------------------------------------------------------
# Load saved V16.8 surrogate package.
# -----------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model package: {MODEL_PATH}")
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")

PACKAGE = joblib.load(MODEL_PATH)
surrogate_models = PACKAGE["surrogate_models"]
target_scalers = PACKAGE["target_scalers"]
x_scaler = PACKAGE["x_scaler"]
FEATURE_COLS = list(PACKAGE["feature_cols"])
ELECTRICAL_TARGET_COLS = list(PACKAGE["electrical_target_cols"])
BEST_GLOBAL_FIXED_FSW = float(PACKAGE["best_global_fixed_fsw_Hz"])
LOG_TARGETS = {
    "Psw_4mos_W", "Pcond_4mos_W", "Ptotal_loss_W", "Pac_W",
    "THD_target_pct", "IL_ripple_rms_A", "Irms_A",
}

# Compute the two normalization constants exactly from the final 936-point
# physical dataset using the V16.8 electrothermal equations.
RAW_DATASET = pd.read_csv(DATASET_PATH)


def calculate_L_from_R_PF(R, PF, f=F_GRID_HZ):
    PF = np.clip(np.asarray(PF, dtype=float), 1e-6, 0.999999999)
    R = np.asarray(R, dtype=float)
    return R * np.tan(np.arccos(PF)) / (2 * np.pi * f)


def load_stress_from_power_current(power_W, current_A):
    return np.maximum(
        np.asarray(power_W, dtype=float) / RATED_POWER_W,
        np.asarray(current_A, dtype=float) / RATED_CURRENT_A,
    )


def add_load_band(df):
    out = df.copy()
    out["load_stress_index"] = load_stress_from_power_current(
        out["P_est_W"], out["I_est_A"]
    )
    out["load_band"] = pd.cut(
        out["load_stress_index"],
        bins=[-np.inf, 0.25, 0.60, 0.85, np.inf],
        labels=STRESS_BAND_ORDER,
        ordered=True,
    )
    out["rated_envelope"] = (
        (out["P_est_W"] <= RATED_POWER_W) &
        (out["I_est_A"] <= RATED_CURRENT_A)
    )
    return out


def filter_stability_pass(fsw_Hz):
    return np.asarray(fsw_Hz, dtype=float) >= MIN_VALIDATED_COMMAND_FSW_HZ


def derive_electrical_metrics(df):
    out = df.copy()
    for column in [
        "Psw_4mos_W", "Pcond_4mos_W", "Ptotal_loss_W", "Pac_W",
        "THD_target_pct", "Vout_fund_rms_V", "IL_ripple_rms_A", "Irms_A",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").clip(lower=0)
    out["Pac_W"] = out["Pac_W"].clip(lower=1e-6)
    out["Irms_A"] = out["Irms_A"].clip(lower=1e-6)
    out["Pmos_total_4mos_W"] = out["Psw_4mos_W"] + out["Pcond_4mos_W"]
    out["Ptotal_loss_W"] = np.maximum(out["Ptotal_loss_W"], out["Pmos_total_4mos_W"])
    out["Pother_loss_W"] = (out["Ptotal_loss_W"] - out["Pmos_total_4mos_W"]).clip(lower=0)
    out["Pdc_reconstructed_W"] = out["Pac_W"] + out["Ptotal_loss_W"]
    out["eta_inverter_total_pct"] = (
        100 * out["Pac_W"] / out["Pdc_reconstructed_W"].clip(lower=1e-6)
    ).clip(0, 100)
    out["Vout_error_pct"] = 100 * np.abs(out["Vout_fund_rms_V"] - VOUT_TARGET_RMS) / VOUT_TARGET_RMS
    out["IL_ripple_limit_A"] = np.maximum(
        LIMIT_RIPPLE_PCT / 100 * out["Irms_A"], LIMIT_RIPPLE_ABSOLUTE_A
    )
    out["ripple_index"] = out["IL_ripple_rms_A"] / out["IL_ripple_limit_A"].clip(lower=1e-9)
    return out


def thermal_temperature_constant_loss(P_total_4mos, t_s, Tamb=T_AMB_DEFAULT, Rth_ha=RTH_HA_NOM):
    P_total = np.maximum(np.asarray(P_total_4mos, dtype=float), 0.0)
    P_device = P_total / 4.0
    t = np.maximum(np.asarray(t_s, dtype=float), 0.0)
    tau_ha = Rth_ha * CTH_HA
    fast = P_device * RTH_JH * (1 - np.exp(-t / TAU_JH))
    slow = P_total * Rth_ha * (1 - np.exp(-t / tau_ha))
    return Tamb + fast + slow


def electrothermal_solution(Psw_ref, Pcond_ref, t_s, Tamb=T_AMB_DEFAULT, Rth_ha=RTH_HA_NOM):
    p_sw_ref = np.maximum(np.asarray(Psw_ref, dtype=float), 0.0)
    p_cond_ref = np.maximum(np.asarray(Pcond_ref, dtype=float), 0.0)
    shape = np.broadcast(p_sw_ref, p_cond_ref).shape
    p_sw_ref = np.broadcast_to(p_sw_ref, shape).astype(float)
    p_cond_ref = np.broadcast_to(p_cond_ref, shape).astype(float)
    tj = np.full(shape, float(Tamb), dtype=float)
    for _ in range(THERMAL_MAX_ITER):
        delta = tj - T_REF_C
        sw_multiplier = np.clip(1 + ALPHA_SW_PER_C * delta, 0.65, 2.50)
        conduction_multiplier = np.clip(1 + ALPHA_COND_PER_C * delta, 0.55, 3.50)
        p_sw_hot = p_sw_ref * sw_multiplier
        p_cond_hot = p_cond_ref * conduction_multiplier
        tj_new = thermal_temperature_constant_loss(
            p_sw_hot + p_cond_hot, t_s, Tamb=Tamb, Rth_ha=Rth_ha
        )
        tj_next = (1 - THERMAL_DAMPING) * tj + THERMAL_DAMPING * tj_new
        if np.nanmax(np.abs(tj_next - tj)) < THERMAL_TOL_C:
            tj = tj_next
            break
        tj = tj_next
    delta = tj - T_REF_C
    p_sw_hot = p_sw_ref * np.clip(1 + ALPHA_SW_PER_C * delta, 0.65, 2.50)
    p_cond_hot = p_cond_ref * np.clip(1 + ALPHA_COND_PER_C * delta, 0.55, 3.50)
    return {
        "Tj_C": tj,
        "Psw_hot_W": p_sw_hot,
        "Pcond_hot_W": p_cond_hot,
        "Pmos_hot_W": p_sw_hot + p_cond_hot,
    }


def append_electrothermal(df, time_s=600.0, Tamb=T_AMB_DEFAULT):
    out = df.copy()
    nominal = electrothermal_solution(
        out["Psw_4mos_W"].values, out["Pcond_4mos_W"].values,
        time_s, Tamb=Tamb, Rth_ha=RTH_HA_NOM,
    )
    conservative = electrothermal_solution(
        out["Psw_4mos_W"].values, out["Pcond_4mos_W"].values,
        time_s, Tamb=Tamb, Rth_ha=RTH_HA_HIGH,
    )
    out["Psw_hot_W"] = nominal["Psw_hot_W"]
    out["Pcond_hot_W"] = nominal["Pcond_hot_W"]
    out["Pmos_hot_W"] = nominal["Pmos_hot_W"]
    out["Tj_nominal_C"] = nominal["Tj_C"]
    out["Tj_conservative_C"] = conservative["Tj_C"]
    out["Tj_upper_C"] = out["Tj_conservative_C"]
    out["Ptotal_hot_loss_W"] = out["Pother_loss_W"] + out["Pmos_hot_W"]
    out["Pdc_electrothermal_W"] = out["Pac_W"] + out["Ptotal_hot_loss_W"]
    out["eta_inverter_electrothermal_pct"] = (
        100 * out["Pac_W"] / out["Pdc_electrothermal_W"].clip(lower=1e-6)
    ).clip(0, 100)
    return out


def prepare_physical_dataset(df):
    out = df.copy()
    out["THD_target_pct"] = out["THD_used_pct"]
    out["Ptotal_loss_W"] = (out["Pdc_W"] - out["Pac_W"]).clip(lower=0)
    out["X_L_ohm"] = 2 * np.pi * F_GRID_HZ * out["L_H"]
    out["Z_mag_ohm"] = np.sqrt(out["R_ohm"] ** 2 + out["X_L_ohm"] ** 2)
    out["I_est_A"] = VOUT_TARGET_RMS / out["Z_mag_ohm"]
    out["P_est_W"] = VOUT_TARGET_RMS ** 2 * out["PF"] ** 2 / out["R_ohm"]
    out = add_load_band(derive_electrical_metrics(out))
    return append_electrothermal(out, 600.0, T_AMB_DEFAULT)


_LANDSCAPE = prepare_physical_dataset(RAW_DATASET)
PMOS_NORM_DEN = max(float(np.nanmedian(_LANDSCAPE["Pmos_hot_W"])), 1e-6)
TOTAL_NORM_DEN = max(float(np.nanmedian(_LANDSCAPE["Ptotal_hot_loss_W"])), 1e-6)


def inverse_target(values, name):
    values = np.asarray(values, dtype=float)
    return np.expm1(values) if name in LOG_TARGETS else values


def make_feature_df(PF, R_ohm, fsw_Hz):
    PF_arr, R_arr, fsw_arr = np.broadcast_arrays(
        np.asarray(PF, dtype=float),
        np.asarray(R_ohm, dtype=float),
        np.asarray(fsw_Hz, dtype=float),
    )
    PF_flat = PF_arr.ravel()
    R_flat = R_arr.ravel()
    frequency_flat = fsw_arr.ravel()
    if np.any((PF_flat < PF_MIN) | (PF_flat > PF_MAX)):
        raise ValueError(f"PF must remain inside the validated domain {PF_MIN:.2f}–{PF_MAX:.2f}.")
    if np.any((R_flat < TRAIN_R_MIN) | (R_flat > TRAIN_R_MAX)):
        raise ValueError(f"R must remain inside the validated domain {TRAIN_R_MIN:g}–{TRAIN_R_MAX:g} Ω.")
    if np.any((frequency_flat < FSW_DATA_MIN_HZ) | (frequency_flat > FSW_DATA_MAX_HZ)):
        raise ValueError("Switching frequency is outside the trained domain.")

    L = calculate_L_from_R_PF(R_flat, PF_flat)
    X_L = 2 * np.pi * F_GRID_HZ * L
    Z = np.sqrt(R_flat ** 2 + X_L ** 2)
    I_est = VOUT_TARGET_RMS / Z
    P_est = VOUT_TARGET_RMS ** 2 * PF_flat ** 2 / R_flat
    frame = pd.DataFrame({
        "PF": PF_flat,
        "R_ohm": R_flat,
        "fsw_Hz": frequency_flat,
        "fsw_kHz": frequency_flat / 1000,
        "L_H": L,
        "X_L_ohm": X_L,
        "Z_mag_ohm": Z,
        "I_est_A": I_est,
        "P_est_W": P_est,
        "log_R": np.log(R_flat),
        "log_fsw": np.log(frequency_flat),
        "fsw_norm": frequency_flat / FSW_DATA_MAX_HZ,
        "inv_fsw_norm": FSW_DATA_MIN_HZ / frequency_flat,
        "log_P_est": np.log1p(P_est),
        "log_I_est": np.log1p(I_est),
    })
    return add_load_band(frame)


def predict_raw(feature_df):
    transformed = x_scaler.transform(feature_df[FEATURE_COLS])
    outputs = {}
    for target in ELECTRICAL_TARGET_COLS:
        scaled = surrogate_models[target].predict(transformed).reshape(-1, 1)
        transformed_prediction = target_scalers[target].inverse_transform(scaled).ravel()
        prediction = inverse_target(transformed_prediction, target)
        if target != "Vout_fund_rms_V":
            prediction = np.maximum(prediction, 0)
        outputs[target] = prediction
    return derive_electrical_metrics(pd.DataFrame(outputs))


def predict_metrics(PF, R_ohm, fsw_Hz, time_s=600.0, Tamb=T_AMB_DEFAULT):
    features = make_feature_df(PF, R_ohm, fsw_Hz)
    electrical = predict_raw(features)
    output = pd.concat([features.reset_index(drop=True), electrical.reset_index(drop=True)], axis=1)
    output = append_electrothermal(output, time_s=time_s, Tamb=Tamb)
    output["pass_voltage"] = output["Vout_error_pct"] <= LIMIT_VOUT_ERROR_PCT
    output["pass_THD"] = output["THD_target_pct"] <= LIMIT_THD_PCT
    output["pass_ripple"] = output["ripple_index"] <= 1.0
    output["pass_electrical"] = output["pass_voltage"] & output["pass_THD"] & output["pass_ripple"]
    output["filter_stability_constraint_pass"] = filter_stability_pass(output["fsw_Hz"])
    output["below_derating_boundary"] = output["Tj_upper_C"] <= T_VALID_MAX_C
    output["below_failure_boundary"] = output["Tj_upper_C"] < T_FAILURE_C
    return output


def load_priority_name(stress):
    stress = float(stress)
    if stress <= 0.25:
        return "LIGHT_EFFICIENCY"
    if stress <= 0.60:
        return "MODERATE_BALANCED"
    return "HEAVY_THERMAL"


def choose_candidate_by_load_priority(candidates):
    if candidates.empty:
        raise ValueError("Candidate table is empty.")
    stress = float(candidates.iloc[0]["load_stress_index"])
    priority = load_priority_name(stress)
    max_eta = float(candidates["eta_inverter_electrothermal_pct"].max())
    if priority == "LIGHT_EFFICIENCY":
        pool = candidates[candidates["eta_inverter_electrothermal_pct"] >= max_eta - LIGHT_EFFICIENCY_TIE_PP]
        selected = pool.sort_values(
            ["eta_inverter_electrothermal_pct", "Ptotal_hot_loss_W", "Pmos_hot_W", "fsw_Hz"],
            ascending=[False, True, True, True],
        ).iloc[0]
    elif priority == "MODERATE_BALANCED":
        pool = candidates[candidates["eta_inverter_electrothermal_pct"] >= max_eta - MODERATE_EFFICIENCY_TIE_PP]
        selected = pool.sort_values(
            ["Ptotal_hot_loss_W", "Tj_upper_C", "Pmos_hot_W", "eta_inverter_electrothermal_pct"],
            ascending=[True, True, True, False],
        ).iloc[0]
    else:
        pool = candidates[candidates["eta_inverter_electrothermal_pct"] >= max_eta - HEAVY_EFFICIENCY_TOLERANCE_PP]
        selected = pool.sort_values(
            ["Tj_upper_C", "Pmos_hot_W", "Ptotal_hot_loss_W", "eta_inverter_electrothermal_pct"],
            ascending=[True, True, True, False],
        ).iloc[0]
    selected = selected.copy()
    selected["load_priority"] = priority
    selected["best_candidate_efficiency_pct"] = max_eta
    selected["efficiency_sacrifice_from_best_pp"] = max_eta - float(selected["eta_inverter_electrothermal_pct"])
    return selected


def continuous_objective_cost(prediction):
    stress = prediction["load_stress_index"].to_numpy(dtype=float)
    eta_cost = np.maximum(100 - prediction["eta_inverter_electrothermal_pct"].to_numpy(dtype=float), 0)
    pmos = prediction["Pmos_hot_W"].to_numpy(dtype=float)
    total = prediction["Ptotal_hot_loss_W"].to_numpy(dtype=float)
    temperature = prediction["Tj_upper_C"].to_numpy(dtype=float)
    pmos_norm = pmos / PMOS_NORM_DEN
    total_norm = total / TOTAL_NORM_DEN
    thermal_norm = np.maximum((temperature - T_AMB_DEFAULT) / (T_FAILURE_C - T_AMB_DEFAULT), 0)
    light = 0.72 * eta_cost + 0.18 * total_norm + 0.10 * pmos_norm
    moderate = 0.48 * eta_cost + 0.30 * total_norm + 0.22 * thermal_norm
    heavy = 0.18 * eta_cost + 0.32 * pmos_norm + 0.42 * thermal_norm + 0.08 * total_norm
    soft = np.where(stress <= 0.25, light, np.where(stress <= 0.60, moderate, heavy))
    soft += 5 * np.maximum((temperature - T_VALID_MAX_C) / (T_FAILURE_C - T_VALID_MAX_C), 0) ** 2
    hard = (
        (~prediction["pass_electrical"].to_numpy(dtype=bool)) |
        (~prediction["filter_stability_constraint_pass"].to_numpy(dtype=bool)) |
        (temperature >= T_FAILURE_C)
    )
    return np.where(hard, 1e9, soft)


def gwo_continuous_frequency_live(PF, R_ohm, time_s, Tamb, n_agents, n_iters, repeats, seed, progress=None):
    best = None
    total_iterations = max(int(n_iters) * int(repeats), 1)
    completed = 0
    for repeat in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + repeat)
        positions = rng.uniform(MIN_VALIDATED_COMMAND_FSW_HZ, MAX_COMMAND_FSW_HZ, int(n_agents))
        alpha, beta, delta = map(float, positions[:3])
        alpha_score = beta_score = delta_score = np.inf
        convergence = []
        for iteration in range(int(n_iters)):
            prediction = predict_metrics(
                np.full(len(positions), float(PF)),
                np.full(len(positions), float(R_ohm)),
                positions, time_s=time_s, Tamb=Tamb,
            )
            scores = continuous_objective_cost(prediction)
            order = np.argsort(scores)
            if scores[order[0]] < alpha_score:
                alpha_score = float(scores[order[0]])
                alpha = float(positions[order[0]])
            if scores[order[1]] < beta_score:
                beta_score = float(scores[order[1]])
                beta = float(positions[order[1]])
            if scores[order[2]] < delta_score:
                delta_score = float(scores[order[2]])
                delta = float(positions[order[2]])
            a = 2 - 2 * iteration / max(int(n_iters), 1)
            new_positions = np.zeros_like(positions)
            for index, position in enumerate(positions):
                estimates = []
                for leader in [alpha, beta, delta]:
                    r1, r2 = rng.random(), rng.random()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    estimates.append(leader - A * abs(C * leader - position))
                new_positions[index] = np.mean(estimates)
            positions = np.clip(new_positions, MIN_VALIDATED_COMMAND_FSW_HZ, MAX_COMMAND_FSW_HZ)
            convergence.append(alpha_score)
            completed += 1
            if progress is not None:
                frac = 0.08 + 0.72 * completed / total_iterations
                progress(frac, desc=f"GWO search — repeat {repeat+1}/{repeats}, iteration {iteration+1}/{n_iters}")
        candidate = {"fsw_Hz": alpha, "score": alpha_score, "convergence": convergence}
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def shutdown_action(reason, full_power_temperature=np.nan):
    return {
        "service_available": False,
        "full_power_operation": False,
        "derated_operation": False,
        "valid_temperature_operation": False,
        "operating_state": "FAILURE_SHUTDOWN",
        "commanded_fsw_Hz": 0.0,
        "nominal_fsw_Hz": np.nan,
        "effective_R_ohm": np.nan,
        "delivered_power_fraction": 0.0,
        "delivered_power_pct": 0.0,
        "shutdown_required": True,
        "shutdown_reason": str(reason),
        "full_power_predicted_Tj_C": full_power_temperature,
        "metrics": pd.Series(dtype=float),
        "selection_source": "protective supervisor",
    }


def row_to_action(row, delivered_fraction, state, selection_source, full_power_temperature=None):
    temperature = float(row["Tj_upper_C"])
    service = bool(row["pass_electrical"]) and bool(row["filter_stability_constraint_pass"]) and temperature < T_FAILURE_C
    if not service:
        return shutdown_action(
            "Selected candidate violates electrical, filter, or 175°C limits.",
            temperature if full_power_temperature is None else full_power_temperature,
        )
    return {
        "service_available": True,
        "full_power_operation": bool(np.isclose(delivered_fraction, 1.0)),
        "derated_operation": bool(delivered_fraction < 1.0),
        "valid_temperature_operation": bool(temperature <= T_VALID_MAX_C),
        "operating_state": state,
        "commanded_fsw_Hz": float(row["fsw_Hz"]),
        "nominal_fsw_Hz": float(row["fsw_Hz"]),
        "effective_R_ohm": float(row["R_ohm"]),
        "delivered_power_fraction": float(delivered_fraction),
        "delivered_power_pct": 100 * float(delivered_fraction),
        "shutdown_required": False,
        "shutdown_reason": "",
        "full_power_predicted_Tj_C": temperature if full_power_temperature is None else float(full_power_temperature),
        "metrics": row.copy(),
        "selection_source": selection_source,
    }


def build_ai_full_power_action(PF, R_ohm, time_s, Tamb, continuous):
    frequencies = np.unique(np.append(np.asarray(ALLOWED_MEASURED_FREQUENCIES_HZ, dtype=float), continuous["fsw_Hz"]))
    candidates = predict_metrics(
        np.full(len(frequencies), float(PF)),
        np.full(len(frequencies), float(R_ohm)),
        frequencies, time_s=time_s, Tamb=Tamb,
    )
    candidates["selection_source"] = np.where(
        np.isclose(candidates["fsw_Hz"], continuous["fsw_Hz"], atol=1e-7, rtol=0),
        "continuous GWO candidate", "measured-frequency safeguard",
    )
    operational = candidates[
        candidates["pass_electrical"] &
        candidates["filter_stability_constraint_pass"] &
        (candidates["Tj_upper_C"] < T_FAILURE_C)
    ]
    if operational.empty:
        action = shutdown_action("No AI frequency satisfies full-power electrical/filter/175°C limits.")
        action["continuous_gwo_fsw_Hz"] = continuous["fsw_Hz"]
        action["convergence"] = continuous["convergence"]
        action["candidate_table"] = candidates
        return action
    valid = operational[operational["Tj_upper_C"] <= T_VALID_MAX_C]
    selected = choose_candidate_by_load_priority(valid if not valid.empty else operational)
    state = "VALID_FULL_POWER" if float(selected["Tj_upper_C"]) <= T_VALID_MAX_C else "DERATING_REQUIRED_AT_FULL_POWER"
    action = row_to_action(selected, 1.0, state, str(selected["selection_source"]), float(selected["Tj_upper_C"]))
    action["continuous_gwo_fsw_Hz"] = continuous["fsw_Hz"]
    action["convergence"] = continuous["convergence"]
    action["candidate_table"] = candidates
    action["load_priority"] = selected["load_priority"]
    return action


def ai_frequency_power_search(PF, requested_R_ohm, time_s, Tamb, continuous_frequency=np.nan):
    fractions = np.arange(0.98, AI_MIN_POWER_FRACTION - AI_POWER_STEP / 2, -AI_POWER_STEP)
    frequencies = np.asarray(ALLOWED_MEASURED_FREQUENCIES_HZ, dtype=float)
    if np.isfinite(continuous_frequency):
        frequencies = np.unique(np.append(frequencies, continuous_frequency))
    fraction_grid, frequency_grid = np.meshgrid(fractions, frequencies, indexing="ij")
    fractions_flat = fraction_grid.ravel()
    frequencies_flat = frequency_grid.ravel()
    effective_R = float(requested_R_ohm) / fractions_flat
    in_domain = effective_R <= TRAIN_R_MAX + 1e-9
    fractions_flat = fractions_flat[in_domain]
    frequencies_flat = frequencies_flat[in_domain]
    effective_R = effective_R[in_domain]
    if not len(fractions_flat):
        return pd.DataFrame()
    prediction = predict_metrics(
        np.full(len(fractions_flat), float(PF)), effective_R, frequencies_flat,
        time_s=time_s, Tamb=Tamb,
    )
    prediction["requested_R_ohm"] = float(requested_R_ohm)
    prediction["effective_R_ohm"] = effective_R
    prediction["delivered_power_fraction"] = fractions_flat
    prediction["delivered_power_pct"] = 100 * fractions_flat
    prediction["selection_source"] = np.where(
        np.isclose(frequencies_flat, continuous_frequency, atol=1e-7, rtol=0),
        "continuous GWO frequency + AI power search",
        "measured frequency + AI power search",
    )
    return prediction


def select_ai_supervisor_from_full(PF, R_ohm, time_s, Tamb, full_power_action):
    if full_power_action.get("service_available", False) and full_power_action.get("valid_temperature_operation", False):
        result = dict(full_power_action)
        result["selection_source"] = str(result["selection_source"]) + " — full power retained"
        return result
    search = ai_frequency_power_search(
        PF, R_ohm, time_s, Tamb,
        full_power_action.get("continuous_gwo_fsw_Hz", np.nan),
    )
    if search.empty:
        return shutdown_action("AI power search leaves the validated resistance domain.", full_power_action.get("full_power_predicted_Tj_C", np.nan))
    service = search[
        search["pass_electrical"] & search["filter_stability_constraint_pass"] & (search["Tj_upper_C"] < T_FAILURE_C)
    ]
    if service.empty:
        return shutdown_action("AI could not find a frequency/power pair below 175°C.", full_power_action.get("full_power_predicted_Tj_C", np.nan))
    valid = service[service["Tj_upper_C"] <= T_VALID_MAX_C]
    pool = valid if not valid.empty else service
    maximum_fraction = float(pool["delivered_power_fraction"].max())
    near_maximum = pool[pool["delivered_power_fraction"] >= maximum_fraction - DELIVERED_POWER_TIE]
    selected = choose_candidate_by_load_priority(near_maximum)
    state = "AI_DERATED_VALID" if float(selected["Tj_upper_C"]) <= T_VALID_MAX_C else "AI_DERATED_CONDITIONAL"
    action = row_to_action(
        selected, float(selected["delivered_power_fraction"]), state,
        str(selected["selection_source"]), full_power_action.get("full_power_predicted_Tj_C", np.nan),
    )
    action["effective_R_ohm"] = float(selected["effective_R_ohm"])
    action["candidate_table"] = search
    action["load_priority"] = selected["load_priority"]
    return action


def fixed_derating_fraction(full_power_temperature_C):
    temperature = float(full_power_temperature_C)
    if temperature <= T_VALID_MAX_C:
        return 1.0
    fraction = 1.0 - (1.0 - FIXED_DERATING_MIN_FRACTION) * (temperature - T_VALID_MAX_C) / (T_FAILURE_C - T_VALID_MAX_C)
    return float(np.clip(fraction, FIXED_DERATING_MIN_FRACTION, 1.0))


def evaluate_fixed_full_power(PF, R_ohm, fixed_fsw_Hz, time_s, Tamb):
    row = predict_metrics(float(PF), float(R_ohm), [float(fixed_fsw_Hz)], time_s=time_s, Tamb=Tamb).iloc[0]
    if (not bool(row["pass_electrical"])) or (not bool(row["filter_stability_constraint_pass"])) or float(row["Tj_upper_C"]) >= T_FAILURE_C:
        return shutdown_action("Fixed full-power state fails electrical/filter/175°C limits.", float(row["Tj_upper_C"]))
    state = "VALID_FULL_POWER" if float(row["Tj_upper_C"]) <= T_VALID_MAX_C else "DERATING_REQUIRED_AT_FULL_POWER"
    return row_to_action(row, 1.0, state, "fixed frequency — full-power study", float(row["Tj_upper_C"]))


def evaluate_fixed_supervisor(PF, R_ohm, fixed_fsw_Hz, time_s, Tamb, full_power_action):
    if full_power_action.get("service_available", False) and full_power_action.get("valid_temperature_operation", False):
        result = dict(full_power_action)
        result["selection_source"] = "fixed frequency — full power retained"
        return result
    full_temperature = full_power_action.get("full_power_predicted_Tj_C", np.nan)
    if not np.isfinite(full_temperature):
        return shutdown_action("Fixed full-power temperature is unavailable.", full_temperature)
    fraction = fixed_derating_fraction(full_temperature)
    effective_R = float(R_ohm) / fraction
    if effective_R > TRAIN_R_MAX + 1e-9:
        return shutdown_action("Predetermined fixed derating leaves the validated domain.", full_temperature)
    row = predict_metrics(float(PF), effective_R, [float(fixed_fsw_Hz)], time_s=time_s, Tamb=Tamb).iloc[0]
    if (not bool(row["pass_electrical"])) or (not bool(row["filter_stability_constraint_pass"])) or float(row["Tj_upper_C"]) >= T_FAILURE_C:
        return shutdown_action("Predetermined fixed derating does not produce a valid state.", full_temperature)
    state = "FIXED_DERATED_VALID" if float(row["Tj_upper_C"]) <= T_VALID_MAX_C else "FIXED_DERATED_CONDITIONAL"
    action = row_to_action(row, fraction, state, "fixed predetermined thermal-derating curve", full_temperature)
    action["effective_R_ohm"] = effective_R
    return action


def action_to_row(action):
    metrics = action.get("metrics", pd.Series(dtype=float))
    row = metrics.to_dict() if isinstance(metrics, pd.Series) else {}
    for key, value in action.items():
        if key in {"metrics", "candidate_table", "convergence"}:
            continue
        if np.isscalar(value) or value is None:
            row[key] = value
    return row


def case_physical_summary(PF, R_ohm):
    L = float(calculate_L_from_R_PF(R_ohm, PF))
    X_L = 2 * np.pi * F_GRID_HZ * L
    Z = float(np.sqrt(R_ohm ** 2 + X_L ** 2))
    current = VOUT_TARGET_RMS / Z
    power = VOUT_TARGET_RMS ** 2 * PF ** 2 / R_ohm
    stress = max(power / RATED_POWER_W, current / RATED_CURRENT_A)
    if stress <= 0.25:
        band = STRESS_BAND_ORDER[0]
    elif stress <= 0.60:
        band = STRESS_BAND_ORDER[1]
    elif stress <= 0.85:
        band = STRESS_BAND_ORDER[2]
    else:
        band = STRESS_BAND_ORDER[3]
    return {
        "L_H": L, "I_est_A": current, "P_est_W": power,
        "load_stress_index": stress, "load_band": band,
        "inside_rated_envelope": bool(power <= RATED_POWER_W and current <= RATED_CURRENT_A),
    }


def relative_thermal_lifetime_ratio(T_ai_C, T_fixed_C, Ea_eV):
    if not (np.isfinite(T_ai_C) and np.isfinite(T_fixed_C)):
        return np.nan
    if T_ai_C >= T_FAILURE_C or T_fixed_C >= T_FAILURE_C:
        return np.nan
    T_ai_K = float(T_ai_C) + 273.15
    T_fixed_K = float(T_fixed_C) + 273.15
    exponent = float(Ea_eV) / BOLTZMANN_EV_K * (1.0 / T_ai_K - 1.0 / T_fixed_K)
    return float(np.exp(np.clip(exponent, -50, 50)))


def make_convergence_plot(convergence):
    fig = go.Figure()
    if convergence:
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(convergence) + 1), y=convergence,
            mode="lines", name="Best objective",
        ))
    fig.update_layout(
        title="GWO Convergence", xaxis_title="Iteration", yaxis_title="Best objective",
        template="plotly_white", height=380, margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def metric_table_and_plot(ai_row, fixed_row, selected_metrics, plot_metric, Ea_eV):
    ai_t = float(ai_row.get("Tj_upper_C", np.nan))
    fx_t = float(fixed_row.get("Tj_upper_C", np.nan))
    lifetime_ratio = relative_thermal_lifetime_ratio(ai_t, fx_t, Ea_eV)
    rows = []
    for metric in selected_metrics:
        key, unit, preference = METRIC_DEFS[metric]
        if metric == "Relative thermal lifetime":
            ai_value = lifetime_ratio
            fixed_value = 1.0 if np.isfinite(lifetime_ratio) else np.nan
        else:
            ai_value = float(ai_row.get(key, np.nan))
            fixed_value = float(fixed_row.get(key, np.nan))
            if metric == "Switching frequency":
                ai_value /= 1000.0 if np.isfinite(ai_value) else 1.0
                fixed_value /= 1000.0 if np.isfinite(fixed_value) else 1.0
        difference = ai_value - fixed_value if np.isfinite(ai_value) and np.isfinite(fixed_value) else np.nan
        rows.append({
            "Metric": metric,
            "AI": ai_value,
            "Fixed": fixed_value,
            "AI − Fixed": difference,
            "Unit": unit,
        })
    table = pd.DataFrame(rows)

    key, unit, _ = METRIC_DEFS[plot_metric]
    if plot_metric == "Relative thermal lifetime":
        values = [lifetime_ratio, 1.0 if np.isfinite(lifetime_ratio) else np.nan]
    else:
        values = [float(ai_row.get(key, np.nan)), float(fixed_row.get(key, np.nan))]
        if plot_metric == "Switching frequency":
            values = [v / 1000.0 if np.isfinite(v) else np.nan for v in values]
    fig = go.Figure(go.Bar(
        x=["AI dynamic", "Fixed reference"], y=values,
        text=[f"{v:.3f}" if np.isfinite(v) else "N/A" for v in values],
        textposition="outside",
    ))
    if plot_metric == "Junction temperature":
        fig.add_hline(y=T_VALID_MAX_C, line_dash="dot", annotation_text="150°C derating")
        fig.add_hline(y=T_FAILURE_C, line_dash="dash", annotation_text="175°C protection")
    fig.update_layout(
        title=plot_metric, yaxis_title=unit, template="plotly_white", height=430,
        margin=dict(l=55, r=30, t=65, b=50), showlegend=False,
    )
    return table, fig, lifetime_ratio


def load_preset(name):
    p = CASE_PRESETS.get(name)
    if not p:
        return gr.update(), gr.update(), gr.update(), gr.update(), "**Custom case.** Move the sliders and run the live engine."
    return p["PF"], p["R_ohm"], p["Tamb_C"], p["time_s"], f"**Preset purpose:** {p['purpose']}"


def simulate_live_case(
    preset_name, PF, R_ohm, Tamb, time_s, fixed_kHz, mode,
    selected_metrics, plot_metric, Ea_eV,
    progress=gr.Progress(),
):
    started = time.perf_counter()
    progress(0.01, desc="Checking the requested operating point")
    PF = float(PF)
    R_ohm = float(R_ohm)
    Tamb = float(Tamb)
    time_s = float(time_s)
    fixed_fsw = float(fixed_kHz) * 1000.0
    selected_metrics = list(selected_metrics or DEFAULT_METRICS)
    if plot_metric not in METRIC_DEFS:
        plot_metric = "Junction temperature"
    if not (PF_MIN <= PF <= PF_MAX):
        raise gr.Error(f"PF must be between {PF_MIN} and {PF_MAX}.")
    if not (TRAIN_R_MIN <= R_ohm <= TRAIN_R_MAX):
        raise gr.Error(f"R must be between {TRAIN_R_MIN:g} and {TRAIN_R_MAX:g} Ω.")
    if not (30 <= time_s <= 1200):
        raise gr.Error("Study time must be between 30 and 1200 s.")
    physical = case_physical_summary(PF, R_ohm)

    progress(0.05, desc="Loading V16.8 ANN surrogate and preparing GWO")
    continuous = gwo_continuous_frequency_live(
        PF, R_ohm, time_s, Tamb,
        n_agents=GWO_AGENTS, n_iters=GWO_ITERS, repeats=GWO_REPEATS,
        seed=RANDOM_STATE, progress=progress,
    )

    progress(0.82, desc="Evaluating measured-frequency safeguards")
    ai_full = build_ai_full_power_action(PF, R_ohm, time_s, Tamb, continuous)

    progress(0.88, desc="Applying the frequency-first supervisory logic")
    if mode == "Complete supervisor":
        ai_action = select_ai_supervisor_from_full(PF, R_ohm, time_s, Tamb, ai_full)
    else:
        ai_action = ai_full

    progress(0.92, desc="Evaluating the fixed-frequency reference")
    fixed_full = evaluate_fixed_full_power(PF, R_ohm, fixed_fsw, time_s, Tamb)
    if mode == "Complete supervisor":
        fixed_action = evaluate_fixed_supervisor(PF, R_ohm, fixed_fsw, time_s, Tamb, fixed_full)
    else:
        fixed_action = fixed_full

    progress(0.96, desc="Calculating the selected engineering metrics")
    ai_row = action_to_row(ai_action)
    fixed_row = action_to_row(fixed_action)
    metrics_table, metric_plot, lifetime_ratio = metric_table_and_plot(
        ai_row, fixed_row, selected_metrics, plot_metric, float(Ea_eV)
    )

    progress(0.985, desc="Building convergence and candidate evidence")
    convergence_plot = make_convergence_plot(continuous.get("convergence", []))
    candidate_table = ai_full.get("candidate_table", pd.DataFrame()).copy()
    candidate_cols = [
        "fsw_Hz", "eta_inverter_electrothermal_pct", "Pmos_hot_W",
        "Ptotal_hot_loss_W", "Tj_upper_C", "THD_target_pct",
        "pass_electrical", "filter_stability_constraint_pass", "selection_source",
    ]
    if not candidate_table.empty:
        candidate_table = candidate_table[[c for c in candidate_cols if c in candidate_table.columns]].copy()
        if "fsw_Hz" in candidate_table:
            candidate_table["fsw_kHz"] = candidate_table["fsw_Hz"] / 1000.0
            candidate_table = candidate_table.drop(columns=["fsw_Hz"])
        candidate_table = candidate_table.sort_values("fsw_kHz").reset_index(drop=True)

    elapsed = time.perf_counter() - started
    ai_state = ai_action.get("operating_state", "N/A")
    fx_state = fixed_action.get("operating_state", "N/A")
    ai_f = ai_action.get("commanded_fsw_Hz", np.nan)
    fx_f = fixed_action.get("commanded_fsw_Hz", np.nan)
    ai_q = ai_action.get("delivered_power_pct", 0.0)
    fx_q = fixed_action.get("delivered_power_pct", 0.0)
    status = f"""
<div class="result-card">
<h3>Live V16.8 simulation complete</h3>
<p><b>This result was calculated now.</b> No Monte Carlo row or nearest-case approximation was used.</p>
<p><b>Requested point:</b> PF {PF:.4f}, R {R_ohm:.4f} Ω, estimated {physical['P_est_W']:.1f} W / {physical['I_est_A']:.2f} A, {physical['load_band']}.</p>
<p><b>AI:</b> {ai_state} — {ai_f/1000 if np.isfinite(ai_f) else 0:.3f} kHz — {ai_q:.1f}% delivered.</p>
<p><b>Fixed:</b> {fx_state} — {fx_f/1000 if np.isfinite(fx_f) else 0:.3f} kHz — {fx_q:.1f}% delivered.</p>
<p><b>GWO:</b> {GWO_AGENTS} wolves × {GWO_ITERS} iterations × {GWO_REPEATS} repeats. Continuous candidate: {continuous['fsw_Hz']/1000:.4f} kHz.</p>
<p><b>Runtime:</b> {elapsed:.2f} s on the current Python backend.</p>
</div>
"""
    if np.isfinite(lifetime_ratio):
        status += (
            f"<p><b>Relative thermal-aging multiplier:</b> AI / Fixed = {lifetime_ratio:.3f}× "
            f"at Ea = {float(Ea_eV):.2f} eV. This is a temperature-accelerated aging comparison, not an absolute lifetime in years.</p>"
        )
    else:
        status += "<p><b>Relative thermal-aging multiplier:</b> not evaluated when a compared state reaches or exceeds the 175°C protection boundary.</p>"
    if not physical["inside_rated_envelope"]:
        status += "<p><b>Rated-envelope notice:</b> this requested point exceeds the 5 kW / 30 A study envelope and should be interpreted as a protection demonstration.</p>"

    progress(1.0, desc="Done")
    return status, metrics_table, metric_plot, convergence_plot, candidate_table


def manual_frequency_compare(PF, R_ohm, Tamb, time_s, f1_kHz, f2_kHz, selected_metrics, plot_metric, Ea_eV, progress=gr.Progress()):
    started = time.perf_counter()
    progress(0.10, desc="Validating manual frequency inputs")
    PF = float(PF); R_ohm = float(R_ohm); Tamb = float(Tamb); time_s = float(time_s)
    f1 = float(f1_kHz) * 1000.0; f2 = float(f2_kHz) * 1000.0
    for f in [f1, f2]:
        if not (MIN_VALIDATED_COMMAND_FSW_HZ <= f <= MAX_COMMAND_FSW_HZ):
            raise gr.Error(f"Manual frequencies must be between {MIN_VALIDATED_COMMAND_FSW_HZ/1000:g} and {MAX_COMMAND_FSW_HZ/1000:g} kHz.")
    progress(0.35, desc="Running ANN-electrothermal prediction at frequency A")
    r1 = predict_metrics(PF, R_ohm, [f1], time_s=time_s, Tamb=Tamb).iloc[0]
    progress(0.65, desc="Running ANN-electrothermal prediction at frequency B")
    r2 = predict_metrics(PF, R_ohm, [f2], time_s=time_s, Tamb=Tamb).iloc[0]
    a1 = row_to_action(r1, 1.0, "MANUAL_FREQUENCY_A", "manual frequency test", float(r1["Tj_upper_C"]))
    a2 = row_to_action(r2, 1.0, "MANUAL_FREQUENCY_B", "manual frequency test", float(r2["Tj_upper_C"]))
    row1 = action_to_row(a1); row2 = action_to_row(a2)
    progress(0.80, desc="Calculating selected metrics")
    table, fig, lifetime = metric_table_and_plot(row1, row2, list(selected_metrics or DEFAULT_METRICS), plot_metric, float(Ea_eV))
    # Rename presentation labels for manual test.
    table = table.rename(columns={"AI": f"{f1_kHz:g} kHz", "Fixed": f"{f2_kHz:g} kHz", "AI − Fixed": "A − B"})
    fig.data[0].x = [f"{f1_kHz:g} kHz", f"{f2_kHz:g} kHz"]
    elapsed = time.perf_counter() - started
    status = f"""
<div class="result-card">
<h3>Manual frequency simulation complete</h3>
<p>PF {PF:.4f}, R {R_ohm:.4f} Ω, study time {time_s:g} s, ambient {Tamb:g}°C.</p>
<p>The ANN-electrothermal model was evaluated directly at {f1_kHz:g} kHz and {f2_kHz:g} kHz. No Monte Carlo lookup was used.</p>
<p><b>Runtime:</b> {elapsed:.2f} s.</p>
</div>
"""
    progress(1.0, desc="Done")
    return status, table, fig


def build_mc_summary_plot():
    if not MC_BAND_PATH.exists():
        return go.Figure()
    d = pd.read_csv(MC_BAND_PATH)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["load_band"], y=d["AI_energy_fulfillment_pct"], name="AI"))
    fig.add_trace(go.Bar(x=d["load_band"], y=d["Fixed_energy_fulfillment_pct"], name="Fixed 24 kHz"))
    fig.update_layout(
        barmode="group", title="Requested-energy fulfillment by load band",
        yaxis_title="Fulfillment (%)", template="plotly_white", height=430,
        margin=dict(l=55, r=30, t=65, b=90),
    )
    return fig


CUSTOM_CSS = """
.gradio-container {max-width: 1500px !important;}
.hero {padding: 22px 26px; border-radius: 18px; background: linear-gradient(135deg,#0b1736,#193b6b); color: white; margin-bottom: 14px;}
.hero h1 {margin:0 0 8px 0; font-size: 32px;}
.hero p {margin:0; opacity:.92; font-size: 15px;}
.result-card {padding: 16px 18px; border: 1px solid #dbe4f0; border-radius: 14px; background: #f8fbff;}
.engine-note {padding:12px 16px; border-left:4px solid #2f75b5; background:#f7faff; border-radius:8px;}
"""


def build_app():
    with gr.Blocks(title="Adaptive Inverter V16.12 — Live Simulation Dashboard") as demo:
        gr.HTML("""
        <div class="hero">
          <h1>Adaptive Inverter V16.12 — Live Simulation Dashboard</h1>
          <p>Interactive ANN + Grey Wolf Optimization + electrothermal supervisor. Live case results are computed on demand from the saved V16.8 model; they are not approximated from Monte Carlo scenarios.</p>
        </div>
        """)

        with gr.Tab("Live AI + GWO case"):
            gr.HTML("<div class='engine-note'><b>Live engine:</b> press Run to execute the full V16.8 GWO search (24 wolves × 40 iterations × 2 repeats), measured-frequency safeguard, supervisor, fixed-reference comparison, electrothermal calculation, and selected metrics. The progress indicator follows the actual GWO iterations; there is no artificial waiting delay.</div>")
            with gr.Row():
                preset = gr.Dropdown(choices=list(CASE_PRESETS.keys()), value="V09 — headline thermal benefit", label="Important case / preset")
                load_button = gr.Button("Load preset")
            preset_note = gr.Markdown(CASE_PRESETS["V09 — headline thermal benefit"]["purpose"])
            with gr.Row():
                PF_input = gr.Slider(PF_MIN, PF_MAX, value=CASE_PRESETS["V09 — headline thermal benefit"]["PF"], step=0.001, label="Power factor")
                R_input = gr.Slider(TRAIN_R_MIN, TRAIN_R_MAX, value=CASE_PRESETS["V09 — headline thermal benefit"]["R_ohm"], step=0.05, label="Load resistance (Ω)")
                Tamb_input = gr.Slider(20, 50, value=25, step=1, label="Ambient temperature (°C)")
                time_input = gr.Slider(30, 1200, value=600, step=10, label="Thermal study time (s)")
            with gr.Row():
                fixed_input = gr.Dropdown(
                    choices=[f / 1000 for f in ALLOWED_MEASURED_FREQUENCIES_HZ],
                    value=BEST_GLOBAL_FIXED_FSW / 1000,
                    label="Fixed reference (kHz)",
                )
                mode_input = gr.Radio(["Frequency only", "Complete supervisor"], value="Complete supervisor", label="Comparison mode")
                Ea_input = gr.Slider(0.30, 1.10, value=0.70, step=0.05, label="Activation energy Ea (eV) — relative thermal aging")
            with gr.Row():
                metric_choices = gr.CheckboxGroup(
                    choices=list(METRIC_DEFS.keys()), value=DEFAULT_METRICS,
                    label="Metrics to include in the result table",
                )
                plot_metric = gr.Dropdown(choices=list(METRIC_DEFS.keys()), value="Junction temperature", label="Metric to plot")
            run_button = gr.Button("Run live simulation", variant="primary", size="lg")
            status = gr.HTML()
            metric_table = gr.Dataframe(label="Selected metrics — freshly calculated", interactive=False)
            with gr.Row():
                metric_plot = gr.Plot(label="Selected metric")
                convergence_plot = gr.Plot(label="GWO convergence")
            candidate_table = gr.Dataframe(label="Full-power candidate evidence", interactive=False)

            load_button.click(load_preset, inputs=[preset], outputs=[PF_input, R_input, Tamb_input, time_input, preset_note])
            run_button.click(
                simulate_live_case,
                inputs=[preset, PF_input, R_input, Tamb_input, time_input, fixed_input, mode_input, metric_choices, plot_metric, Ea_input],
                outputs=[status, metric_table, metric_plot, convergence_plot, candidate_table],
            )

        with gr.Tab("Manual frequency test"):
            gr.Markdown("Use this tab to reproduce direct frequency comparisons, such as the 50-second 12 kHz vs 24 kHz validation-style test, without running GWO.")
            with gr.Row():
                m_pf = gr.Slider(PF_MIN, PF_MAX, value=0.99, step=0.001, label="Power factor")
                m_r = gr.Slider(TRAIN_R_MIN, TRAIN_R_MAX, value=10.0, step=0.05, label="Load resistance (Ω)")
                m_tamb = gr.Slider(20, 50, value=25, step=1, label="Ambient temperature (°C)")
                m_time = gr.Slider(30, 1200, value=50, step=10, label="Study time (s)")
            with gr.Row():
                m_f1 = gr.Slider(MIN_VALIDATED_COMMAND_FSW_HZ/1000, MAX_COMMAND_FSW_HZ/1000, value=12, step=0.1, label="Frequency A (kHz)")
                m_f2 = gr.Slider(MIN_VALIDATED_COMMAND_FSW_HZ/1000, MAX_COMMAND_FSW_HZ/1000, value=24, step=0.1, label="Frequency B (kHz)")
                m_ea = gr.Slider(0.30, 1.10, value=0.70, step=0.05, label="Activation energy Ea (eV)")
            with gr.Row():
                m_metrics = gr.CheckboxGroup(choices=list(METRIC_DEFS.keys()), value=DEFAULT_METRICS, label="Metrics to include")
                m_plot_metric = gr.Dropdown(choices=list(METRIC_DEFS.keys()), value="Junction temperature", label="Metric to plot")
            m_run = gr.Button("Run manual frequency simulation", variant="primary")
            m_status = gr.HTML()
            m_table = gr.Dataframe(label="Manual comparison", interactive=False)
            m_plot = gr.Plot(label="Selected metric")
            m_run.click(
                manual_frequency_compare,
                inputs=[m_pf, m_r, m_tamb, m_time, m_f1, m_f2, m_metrics, m_plot_metric, m_ea],
                outputs=[m_status, m_table, m_plot],
            )

        with gr.Tab("Monte Carlo summary"):
            gr.Markdown("This tab preserves the published 300-scenario summary. It is separate from the live case engine above.")
            if MC_SUMMARY_PATH.exists():
                gr.Dataframe(pd.read_csv(MC_SUMMARY_PATH), label="Paired Monte Carlo summary", interactive=False)
            if MC_BAND_PATH.exists():
                gr.Dataframe(pd.read_csv(MC_BAND_PATH), label="Results by load band", interactive=False)
                gr.Plot(build_mc_summary_plot(), label="Energy fulfillment by load band")

        with gr.Tab("Model boundaries"):
            gr.Markdown(f"""
### Validated domain and supervisory policy

- PF: **{PF_MIN:.2f}–{PF_MAX:.2f}**
- R: **{TRAIN_R_MIN:g}–{TRAIN_R_MAX:g} Ω**
- Trained switching-frequency domain: **2–24 kHz**
- Commandable switching-frequency domain: **{MIN_VALIDATED_COMMAND_FSW_HZ/1000:g}–{MAX_COMMAND_FSW_HZ/1000:g} kHz**
- LC filter: **{FILTER_L_H*1000:.3f} mH / {FILTER_C_F*1e6:.1f} µF**, resonance **{FILTER_RESONANCE_HZ:.1f} Hz**
- Full-power valid thermal region: **Tj ≤ 150°C**
- Derating region: **150°C < Tj < 175°C**
- Protection boundary: **Tj ≥ 175°C**

The relative thermal-lifetime metric uses an Arrhenius temperature-acceleration ratio with a user-selectable activation energy. It is shown only as a **relative aging multiplier** and is not presented as an absolute service life in years.
            """)
    return demo


APP = build_app()

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "10000"))

    APP.queue().launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
        show_error=True
    )
    
