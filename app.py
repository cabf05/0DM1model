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
    1.5, 3.0, 1.6,
    help=(
        "Normal basal EGP ≈1.5–2.0 mg/kg/min. In untreated T1DM, "
        "EGP increases up to ~60% (≈2.7 mg/kg/min) due to unopposed "
        "glucagon and counter-regulatory hormones. Progressive increase "
        "simulates insulin withdrawal studies (Voss et al., Diabetologia 2019)."
    )
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose uptake (mg/kg/min)",
    1.0, 1.6, 1.2,
    help="Primarily reflects cerebral glucose uptake, which is insulin-independent."
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    160, 220, 180,
    help="Plasma glucose above which glucose appears in urine due to transporter saturation."
)

GFR = st.sidebar.slider(
    "GFR (mL/kg/min)",
    1.0, 2.5, 1.7,
    help=(
        "Normal adult GFR ≈1.5–2.0 mL/kg/min. "
        "Lower values simulate renal impairment."
    )
)

st.sidebar.divider()
st.sidebar.title("Meal & Simulation")

meal_carbs = st.sidebar.slider("Meal carbohydrates (g)", 0, 150, 60)

meal_absorption_fraction = st.sidebar.slider(
    "Fraction absorbed by 5 hours",
    0.6, 0.9, 0.8,
    help=(
        "Mixed meals ≈60% absorbed by 5h. "
        "Pure carbohydrate meals may reach 80–90%."
    )
)

meal_time = st.sidebar.slider("Meal time (hours)", 0, 24, 8)
simulation_hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)

noise_level = st.sidebar.slider("Physiological variability", 0.0, 5.0, 1.0)

# =========================================================
# Physiological constants
# =========================================================

DT = 1
Vd = 0.20
TMAX_GLUCOSE = 375

# =========================================================
# Time grid
# =========================================================

time = np.arange(0, simulation_hours * 60, DT)

# =========================================================
# Meal absorption
# =========================================================

def meal_absorption(t, meal_time_h, carbs_g, weight, fraction):
    if carbs_g == 0:
        return np.zeros_like(t)

    duration = 300
    std = 60
    peak = meal_time_h * 60 + 60

    x = np.arange(duration)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel /= kernel.sum()

    absorbed = carbs_g * 1000 * fraction * kernel
    signal = np.zeros_like(t)

    start = int(peak - duration / 2)
    end = start + duration

    if start >= 0 and end < len(t):
        signal[start:end] = absorbed

    return signal / weight

meal_signal = meal_absorption(
    time, meal_time, meal_carbs, weight, meal_absorption_fraction
)

# =========================================================
# State variables
# =========================================================

G = np.zeros_like(time)
G[0] = 100

ketones = np.zeros_like(time)
urinary_glucose = np.zeros_like(time)
urine_volume = np.zeros_like(time)
glucose_roc = np.zeros_like(time)
cumulative_egp = np.zeros_like(time)

# =========================================================
# Simulation loop
# =========================================================

for i in range(1, len(time)):
    glucose = G[i - 1]

    hours_hyper = np.sum(G[:i] > 180) / 60
    egp_multiplier = min(1 + 0.1 * hours_hyper, 1.6)

    hour = (time[i] / 60) % 24
    dawn_multiplier = 1.15 if 4 <= hour <= 8 else 1.0

    egp = egp_baseline * egp_multiplier * dawn_multiplier
    cumulative_egp[i] = cumulative_egp[i - 1] + egp * DT * weight / 1000

    if glucose > 180:
        u_ii = brain_uptake * (1 + 0.01 * (glucose - 180))
    else:
        u_ii = brain_uptake

    if glucose > renal_threshold:
        GFR_total = GFR * weight
        filtered = (GFR_total / 1000) * glucose
        reabsorbed = min(filtered, TMAX_GLUCOSE / 1000)
        excreted = max(filtered - reabsorbed, 0)
        renal_mg_kg_min = (excreted * 1000) / weight
    else:
        renal_mg_kg_min = 0

    urinary_glucose[i] = urinary_glucose[i - 1] + renal_mg_kg_min * DT
    urine_volume[i] = (urinary_glucose[i] / 1000) * 0.018

    if glucose > 200:
        ketone_rate = 0.0015 * (glucose - 200) * (1 + ketones[i - 1] / 10)
        ketones[i] = ketones[i - 1] + ketone_rate * DT
    else:
        ketones[i] = max(ketones[i - 1] - 0.002 * DT, 0)

    dG_mass = egp + meal_signal[i] - u_ii - renal_mg_kg_min
    dG_conc = (dG_mass / Vd) / 10

    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG_conc + noise, 40)

    if i >= 60:
        glucose_roc[i] = G[i] - G[i - 60]

# =========================================================
# Output DataFrame
# =========================================================

df = pd.DataFrame({
    "Time (h)": time / 60,
    "Glucose (mg/dL)": G,
    "Ketones (mmol/L)": ketones,
    "Urinary glucose (g)": urinary_glucose / 1000,
    "Urine volume (L)": urine_volume,
    "Glucose ROC (mg/dL/h)": glucose_roc,
    "Cumulative EGP (g)": cumulative_egp
})

# =========================================================
# Visualization
# =========================================================

st.title("Untreated Type 1 Diabetes – Advanced Physiological Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (h)"], df["Glucose (mg/dL)"], linewidth=2)
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia (>180 mg/dL)")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk (>250 mg/dL)")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =========================================================
# Teaching Points
# =========================================================

with st.expander("📚 Key Teaching Points"):
    st.markdown("""
**Pathophysiology of Untreated Type 1 Diabetes**

- **Absolute insulin deficiency** → Unopposed hepatic glucose production  
- **Progressive EGP elevation** → Counter-regulatory hormone dominance  
- **Renal compensation limits** → Glucosuria cannot prevent severe hyperglycemia  
- **Insulin-independent uptake** → Brain glucose utilization increases but is insufficient  
- **Ketogenesis** → β-hydroxybutyrate ≥3.0 mmol/L defines DKA  
- **Osmotic diuresis** → Each gram of glucose excreted carries ~18 mL of water  

**Clinical Correlates**

- Glucose ROC >56 mg/dL/h exceeds the 99th percentile for health  
- DKA typically develops within 14–18 hours of complete insulin withdrawal  
- Polyuria reflects osmotic diuresis secondary to glucosuria  
""")

# =========================================================
# Export
# =========================================================

st.subheader("Download simulation data")

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name="untreated_t1dm_advanced_simulation.csv",
    mime="text/csv"
)
