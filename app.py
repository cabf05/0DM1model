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
    help="Basal EGP before insulin withdrawal"
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose uptake (mg/kg/min)",
    1.0, 1.6, 1.2
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    160, 220, 180
)

st.sidebar.divider()
st.sidebar.title("Meal & Simulation")

meal_carbs = st.sidebar.slider("Meal carbohydrates (g)", 0, 150, 60)
meal_time = st.sidebar.slider("Meal time (hours)", 0, 24, 8)
simulation_hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)

noise_level = st.sidebar.slider("Physiological variability", 0.0, 5.0, 1.0)

# =========================================================
# Physiological constants
# =========================================================

DT = 1                      # min
Vd = 0.20                   # L/kg (glucose distribution volume)
GFR = 1.7                   # mL/kg/min
TMAX_GLUCOSE = 375          # mg/min (max tubular reabsorption)

# =========================================================
# Time grid
# =========================================================

time = np.arange(0, simulation_hours * 60, DT)

# =========================================================
# Meal absorption (slow, incomplete)
# =========================================================

def meal_absorption(t, meal_time_h, carbs_g, weight):
    if carbs_g == 0:
        return np.zeros_like(t)

    duration = 300  # 5 hours
    std = 60
    peak = meal_time_h * 60 + 60

    x = np.arange(duration)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel /= kernel.sum()

    absorbed = carbs_g * 1000 * 0.6 * kernel
    signal = np.zeros_like(t)

    start = int(peak - duration / 2)
    end = start + duration

    if start >= 0 and end < len(t):
        signal[start:end] = absorbed

    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight)

# =========================================================
# State variables
# =========================================================

G = np.zeros_like(time)
G[0] = 100

ketones = np.zeros_like(time)        # mmol/L (β-hydroxybutyrate)
urinary_glucose = np.zeros_like(time)  # mg
urine_volume = np.zeros_like(time)     # L
glucose_roc = np.zeros_like(time)      # mg/dL/h

# =========================================================
# Simulation loop
# =========================================================

for i in range(1, len(time)):
    glucose = G[i - 1]

    # ---------------------------------------------
    # Progressive EGP increase during insulin lack
    # ---------------------------------------------
    hours_hyper = np.sum(G[:i] > 180) / 60
    egp_multiplier = min(1 + 0.1 * hours_hyper, 1.6)

    # Dawn phenomenon (4–8 AM)
    hour = (time[i] / 60) % 24
    dawn_multiplier = 1.15 if 4 <= hour <= 8 else 1.0

    egp = egp_baseline * egp_multiplier * dawn_multiplier

    # ---------------------------------------------
    # Insulin-independent glucose uptake
    # ---------------------------------------------
    if glucose > 180:
        u_ii = brain_uptake * (1 + 0.01 * (glucose - 180))
    else:
        u_ii = brain_uptake

    # ---------------------------------------------
    # Renal glucose handling (with Tmax)
    # ---------------------------------------------
    if glucose > renal_threshold:
        GFR_total = GFR * weight  # mL/min
        filtered = (GFR_total / 1000) * glucose  # g/min
        reabsorbed = min(filtered, TMAX_GLUCOSE / 1000)
        excreted = max(filtered - reabsorbed, 0)
        renal_mg_kg_min = (excreted * 1000) / weight
    else:
        renal_mg_kg_min = 0

    urinary_glucose[i] = urinary_glucose[i - 1] + renal_mg_kg_min * DT
    urine_volume[i] = urinary_glucose[i] * 0.018 / 1000  # L

    # ---------------------------------------------
    # Ketogenesis (nonlinear, NEFA-driven)
    # ---------------------------------------------
    if glucose > 200:
        ketone_rate = 0.0015 * (glucose - 200) * (1 + ketones[i - 1] / 10)
        ketones[i] = ketones[i - 1] + ketone_rate * DT
    else:
        ketones[i] = max(ketones[i - 1] - 0.002 * DT, 0)

    # ---------------------------------------------
    # Glucose mass balance
    # ---------------------------------------------
    dG_mass = egp + meal_signal[i] - u_ii - renal_mg_kg_min

    # Convert mass (mg/kg/min) → concentration (mg/dL/min)
    dG_conc = (dG_mass / Vd) / 10

    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG_conc + noise, 40)

    # ---------------------------------------------
    # Rate of glucose rise (mg/dL/h)
    # ---------------------------------------------
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
    "Glucose ROC (mg/dL/h)": glucose_roc
})

# =========================================================
# Visualization
# =========================================================

st.title("Untreated Type 1 Diabetes – Advanced Physiological Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (h)"], df["Glucose (mg/dL)"], linewidth=2)
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =========================================================
# Clinical indicators
# =========================================================

dka_time = (
    np.argmax(ketones >= 3.0) / 60
    if np.any(ketones >= 3.0)
    else None
)

st.subheader("Clinical & Educational Indicators")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Peak glucose", f"{np.max(G):.0f} mg/dL")
c2.metric("Max glucose ROC", f"{np.max(glucose_roc):.0f} mg/dL/h")
c3.metric("Urinary glucose loss", f"{urinary_glucose[-1]/1000:.1f} g")
c4.metric("Urine volume", f"{urine_volume[-1]:.1f} L")
c5.metric(
    "Time to DKA",
    f"{dka_time:.1f} h" if dka_time else "Not reached"
)

# =========================================================
# Interpretation
# =========================================================

st.subheader("Physiological Interpretation")

if dka_time:
    st.error(
        "β-hydroxybutyrate exceeds 3.0 mmol/L, meeting clinical criteria for DKA. "
        "This reflects unchecked lipolysis and hepatic ketogenesis in the absence of insulin."
    )
elif np.max(glucose_roc) > 56:
    st.warning(
        "Rate of glucose rise exceeds physiological limits, illustrating "
        "the instability of glucose control without insulin."
    )
else:
    st.info(
        "Progressive hyperglycemia emerges from delayed but sustained counter-regulatory responses."
    )

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
