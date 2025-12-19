import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# App configuration
# =========================================================
st.set_page_config(
    page_title="Untreated Type 1 Diabetes – Physiological Simulator",
    layout="wide"
)

# =========================================================
# Sidebar – Parameters
# =========================================================
st.sidebar.title("Physiology – Untreated T1DM")

weight = st.sidebar.slider("Body weight (kg)", 40, 120, 70)

egp_baseline = st.sidebar.slider(
    "Baseline endogenous glucose production (mg/kg/min)",
    1.5, 3.5, 2.4,
    help="Normal basal EGP ≈1.5-2.0 mg/kg/min. In untreated T1DM, "
         "EGP increases up to 60% (to ~2.7 mg/kg/min) due to unopposed "
         "glucagon and counter-regulatory hormones. Use 2.4-2.9 to simulate insulin withdrawal."
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose uptake (mg/kg/min)",
    0.8, 1.6, 1.2,
    help="Insulin-independent uptake ≈1.0-1.5 mg/kg/min. "
         "Includes brain, RBC, and other tissues with GLUT1/3 transporters."
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    160, 220, 180,
    help="Normal renal threshold ≈180 mg/dL. Glucosuria begins above this level."
)

GFR = st.sidebar.slider(
    "GFR (mL/kg/min)",
    1.0, 2.5, 1.7,
    help="Normal adult GFR ≈1.5–2.0 mL/kg/min. Lower values simulate renal impairment."
)

st.sidebar.divider()
st.sidebar.title("Meal & Simulation")

meal_carbs = st.sidebar.slider("Meal carbohydrates (g)", 0, 150, 0)

meal_absorption_fraction = st.sidebar.slider(
    "Fraction absorbed by 5h",
    0.6, 0.9, 0.8,
    help="Mixed meals ≈60%; pure carbohydrate meals may reach 80–90%"
)

meal_time = st.sidebar.slider("Meal time (hours)", 0, 24, 8)

simulation_hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)

noise_level = st.sidebar.slider(
    "Physiological variability (mg/dL)",
    0.0, 5.0, 1.5,
    help="Models CGM-like measurement noise and biological variability"
)

# =========================================================
# Physiological constants & ketone model params
# =========================================================
DT = 1.0                    # min
Vd = 0.22                   # L/kg (glucose distribution volume)
TMAX_GLUCOSE = 375          # mg/min (max tubular reabsorption)

# Insulin-independent uptake saturation (Michaelis-Menten)
U_II_MAX = 2.0              # mg/kg/min, max insulin-independent uptake
KM_UPTAKE = 180.0           # mg/dL, half-saturation

# Ketone model parameters (physiologically constrained)
KETONE_BASELINE = 0.127     # mmol/L baseline
KETONE_DKA_THRESHOLD = 3.0  # mmol/L diagnostic threshold
KETONE_TYPICAL_MAX = 10.0    # mmol/L typical DKA max
KETONE_ABSOLUTE_MAX = 25.0  # mmol/L hard ceiling to prevent overflow

# Production & utilization parameters (units: mmol/L/min)
# Base production observed in literature ≈ 0.002 - 0.003 mmol/L/min
KETONE_BASE_PROD = 0.006    # base production rate at max insulin deficiency
KM_KETONE = 8.0             # mmol/L, half-saturation constant for production slowdown
MAX_UTILIZATION = 0.002     # mmol/L/min, maximal utilization (brain + kidney)
# Clearance half-life when glucose normalizes
KETONE_CLEARANCE_HALF_LIFE = 240.0  # minutes (~4 hours)
KETONE_CLEARANCE_RATE = np.log(2.0) / KETONE_CLEARANCE_HALF_LIFE  # per minute

# =========================================================
# Time grid
# =========================================================
time = np.arange(0, simulation_hours * 60, DT)
n_steps = len(time)

# =========================================================
# Meal absorption
# =========================================================
def meal_absorption(t, meal_time_h, carbs_g, weight, fraction):
    """Models carbohydrate absorption as Gaussian-like distribution (peak ~1h, duration 5h)"""
    if carbs_g == 0:
        return np.zeros_like(t, dtype=float)
    duration = 300
    std = 60
    peak = meal_time_h * 60 + 60
    x = np.arange(duration, dtype=float)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel /= kernel.sum()
    absorbed = carbs_g * 1000.0 * fraction * kernel  # mg total over duration
    signal = np.zeros_like(t, dtype=float)
    start = int(peak - duration / 2)
    end = start + duration
    if start >= 0 and end < len(t):
        signal[start:end] = absorbed
    elif start < len(t):
        signal[start:] = absorbed[:len(t)-start]
    # return mg/kg/min
    return signal / weight

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight, meal_absorption_fraction)

# =========================================================
# State variables (unchanged interface)
# =========================================================
G = np.zeros(n_steps, dtype=float)
G[0] = 100.0
ketones = np.zeros(n_steps, dtype=float)
ketones[0] = KETONE_BASELINE
urinary_glucose = np.zeros(n_steps, dtype=float)
urine_volume = np.zeros(n_steps, dtype=float)
glucose_roc = np.zeros(n_steps, dtype=float)
cumulative_egp = np.zeros(n_steps, dtype=float)
egp_array = np.zeros(n_steps, dtype=float)
uptake_array = np.zeros(n_steps, dtype=float)
renal_array = np.zeros(n_steps, dtype=float)

# safeguard flag
numerical_instability = False
instability_message = ""

# =========================================================
# Simulation loop (with corrected ketone model and safeguards)
# =========================================================
for i in range(1, n_steps):
    glucose = G[i - 1]

    # quick numerical checks on previous ketone to avoid carrying NaN/inf
    if not np.isfinite(ketones[i-1]) or ketones[i-1] < 0 or ketones[i-1] > 1e6:
        numerical_instability = True
        instability_message = f"Numerical instability detected at t={time[i-1]/60:.2f} h: ketones={ketones[i-1]:.3e}"
        break

    # Progressive EGP
    hours_hyper = np.sum(G[:i] > 180) / 60.0
    egp_multiplier = min(1.0 + 0.1 * hours_hyper, 1.6)

    # Dawn phenomenon
    hour = (time[i] / 60.0) % 24
    dawn_multiplier = 1.15 if 4 <= hour <= 8 else 1.0

    egp = egp_baseline * egp_multiplier * dawn_multiplier  # mg/kg/min
    egp_array[i] = egp
    cumulative_egp[i] = cumulative_egp[i-1] + egp * DT * weight / 1000.0  # grams

    # Insulin-independent uptake (Michaelis-Menten style)
    # convert to mg/kg/min units via saturation:
    u_ii = U_II_MAX * glucose / (KM_UPTAKE + glucose)  # mg/kg/min
    uptake_array[i] = u_ii

    # Renal glucose handling (mg/dL -> mg/min correctly)
    if glucose > renal_threshold:
        GFR_total = GFR * weight  # mL/min
        filtered_mg_min = (GFR_total * glucose) / 100.0  # mg/min (because 1 dL = 100 mL)
        reabsorbed_mg_min = min(filtered_mg_min, TMAX_GLUCOSE)  # mg/min
        excreted_mg_min = max(filtered_mg_min - reabsorbed_mg_min, 0.0)  # mg/min
        renal_mg_kg_min = excreted_mg_min / weight  # mg/kg/min
    else:
        renal_mg_kg_min = 0.0
    renal_array[i] = renal_mg_kg_min

    # Cumulative urinary glucose and urine volume
    urinary_glucose[i] = urinary_glucose[i-1] + renal_mg_kg_min * weight * DT  # mg total
    urine_volume[i] = (urinary_glucose[i] / 1000.0) * 0.018  # L (18 mL per gram)

    # -------------------------------
    # Corrected, physiologic Ketogenesis model
    # -------------------------------
    # Approach:
    # - Production driven by insulin deficiency proxy (hyperglycemia index)
    # - Production saturates as ketones accumulate (Km-like)
    # - Utilization (brain + kidney) increases with ketone level and saturates
    # - Hard absolute cap prevents overflow
    if glucose > 200.0:
        # insulin deficiency index: 0 at 200 mg/dL, 1.0 at >=500 mg/dL
        insulin_def_index = min(max((glucose - 200.0) / 300.0, 0.0), 1.0)
        # base production scales with insulin deficiency
        base_production = KETONE_BASE_PROD * insulin_def_index  # mmol/L/min
        # saturation factor: production decreases as ketones accumulate (Km formulation)
        saturation_factor = KM_KETONE / (KM_KETONE + ketones[i-1])  # in (0,1]
        production_rate = base_production * saturation_factor
        # utilization (brain + kidney) saturable with ketone level
        utilization_rate = MAX_UTILIZATION * (ketones[i-1] / (3.0 + ketones[i-1]))
        # net rate (mmol/L/min)
        net_rate = production_rate - utilization_rate
        ketones[i] = ketones[i-1] + net_rate * DT
        # enforce physiological bounds
        if ketones[i] < 0.0:
            ketones[i] = 0.0
        if ketones[i] > KETONE_ABSOLUTE_MAX:
            ketones[i] = KETONE_ABSOLUTE_MAX
    else:
        # glucose normalized: first-order clearance
        clearance = KETONE_CLEARANCE_RATE * ketones[i-1]
        ketones[i] = max(ketones[i-1] - clearance * DT, 0.0)

    # Safety numeric check after ketone update
    if not np.isfinite(ketones[i]) or ketones[i] > 1e6:
        numerical_instability = True
        instability_message = f"Numerical instability after ketone update at t={time[i]/60:.2f} h: ketones={ketones[i]:.3e}"
        break

    # Glucose mass balance (mg/kg/min)
    dG_mass = egp + meal_signal[i] - u_ii - renal_mg_kg_min
    # convert to concentration change (mg/dL/min)
    dG_conc = dG_mass / (Vd * 10.0)
    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG_conc + noise, 40.0)

    if i >= 60:
        glucose_roc[i] = G[i] - G[i-60]

# if instability detected, fill remaining arrays with NaN (keeps shapes consistent)
if numerical_instability:
    for j in range(i+1, n_steps):
        G[j] = np.nan
        ketones[j] = np.nan
        urinary_glucose[j] = np.nan
        urine_volume[j] = np.nan
        glucose_roc[j] = np.nan
        cumulative_egp[j] = cumulative_egp[i]
        egp_array[j] = egp_array[i]
        uptake_array[j] = uptake_array[i]
        renal_array[j] = renal_array[i]

# =========================================================
# Output DataFrame
# =========================================================
df = pd.DataFrame({
    "Time (h)": time / 60.0,
    "Glucose (mg/dL)": G,
    "EGP (mg/kg/min)": egp_array,
    "Uptake (mg/kg/min)": uptake_array,
    "Renal loss (mg/kg/min)": renal_array,
    "Ketones (mmol/L)": ketones,
    "Urinary glucose (g)": urinary_glucose / 1000.0,
    "Urine volume (L)": urine_volume,
    "Glucose ROC (mg/dL/h)": glucose_roc,
    "Cumulative EGP (g)": cumulative_egp
})

# =========================================================
# Visualization (unchanged)
# =========================================================
st.title("Untreated Type 1 Diabetes – Advanced Physiological Simulator")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["Time (h)"], df["Glucose (mg/dL)"], linewidth=2.5, color='#2E86AB', label='Blood Glucose')
ax.axhline(70, linestyle=":", color="gray", alpha=0.5, label="Normal fasting")
ax.axhline(180, linestyle="--", color="orange", linewidth=2, label="Hyperglycemia (>180 mg/dL)")
ax.axhline(250, linestyle="--", color="red", linewidth=2, label="DKA Risk (>250 mg/dL)")
ax.fill_between(df["Time (h)"], 70, 180, alpha=0.1, color='green', label='Target range')
ax.set_xlabel("Time (hours)", fontsize=12)
ax.set_ylabel("Glucose (mg/dL)", fontsize=12)
ax.set_title("Continuous Glucose Monitor Simulation", fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# =========================================================
# Clinical indicators (unchanged)
# =========================================================
dka_time = (np.argmax(ketones >= KETONE_DKA_THRESHOLD) / 60.0) if np.any(ketones >= KETONE_DKA_THRESHOLD) else None
time_in_target = np.mean((G >= 70) & (G <= 180)) * 100 if np.any(np.isfinite(G)) else 0.0
time_above_180 = np.mean(G > 180) * simulation_hours if np.any(np.isfinite(G)) else 0.0
time_above_250 = np.mean(G > 250) * simulation_hours if np.any(np.isfinite(G)) else 0.0
max_roc = np.nanmax(np.abs(glucose_roc)) if np.any(np.isfinite(glucose_roc)) else np.nan

st.subheader("Clinical & Educational Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Peak glucose", f"{np.nanmax(G):.0f} mg/dL")
c2.metric("Time in range", f"{time_in_target:.0f}%")
c3.metric("Max glucose ROC", f"{max_roc:.0f} mg/dL/h" if np.isfinite(max_roc) else "N/A")
c4.metric("Urinary glucose loss", f"{(urinary_glucose[-1]/1000.0):.1f} g" if np.isfinite(urinary_glucose[-1]) else "N/A")
c5.metric("Cumulative EGP", f"{cumulative_egp[-1]:.0f} g")
c6.metric("Time to DKA", f"{dka_time:.1f} h" if dka_time else "Not reached")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Urine volume", f"{urine_volume[-1]:.1f} L" if np.isfinite(urine_volume[-1]) else "N/A")
c8.metric("Peak ketones", f"{np.nanmax(ketones):.2f} mmol/L" if np.any(np.isfinite(ketones)) else "N/A")
c9.metric("Time >180 mg/dL", f"{time_above_180:.1f} h")
c10.metric("Time >250 mg/dL", f"{time_above_250:.1f} h")

# =========================================================
# Interpretation & Teaching Points (unchanged)
# =========================================================
st.subheader("Physiological Interpretation")
with st.expander("📚 Key Teaching Points"):
    st.markdown("""
**Pathophysiology of Untreated T1DM:**

- Absolute insulin deficiency → Unopposed hepatic glucose production
- Progressive EGP elevation → Counter-regulatory hormones
- Renal compensation limits → Glucosuria cannot prevent hyperglycemia above ~250 mg/dL
- Insulin-independent uptake → Brain glucose utilization saturates (~2 mg/kg/min)
- Ketogenesis → β-hydroxybutyrate rises towards DKA threshold (3.0 mmol/L) over ~8–18 h depending on severity
- Osmotic diuresis → Each gram of glucose excreted carries ~18 mL of water
""")

interpretations = []
if dka_time:
    interpretations.append(("error",
        f"β-hydroxybutyrate exceeds {KETONE_DKA_THRESHOLD:.1f} mmol/L at {dka_time:.1f} hours, meeting clinical criteria for DKA."
    ))
if np.isfinite(max_roc) and max_roc > 56:
    interpretations.append(("warning",
        f"Glucose rate of rise exceeds 99th percentile (>56 mg/dL/h). Maximum observed: {max_roc:.0f} mg/dL/h."
    ))

for level, msg in interpretations:
    if level == "error":
        st.error(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.info(msg)

# =========================================================
# Numerical & physiological validation messages
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Model validation")
if numerical_instability:
    st.sidebar.error(instability_message)

# warn if ketones exceeded model cap
if np.nanmax(ketones) > KETONE_ABSOLUTE_MAX:
    st.sidebar.error(f"Ketones reached absolute cap ({KETONE_ABSOLUTE_MAX} mmol/L). Check parameters.")

# quick plausibility checks
if np.nanmax(ketones) > 50:
    st.sidebar.error("Ketone values exceeded physiological maximum (>50 mmol/L) — numerical instability suspected.")

# =========================================================
# Export CSV
# =========================================================
st.subheader("Download simulation data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name="untreated_t1dm_simulation.csv",
    mime="text/csv"
)
