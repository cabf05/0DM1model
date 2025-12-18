import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# App Configuration
# =========================================================

st.set_page_config(
    page_title="Untreated Type 1 Diabetes Simulator (Physiology-Based)",
    layout="wide"
)

# =========================================================
# Sidebar – Parameters
# =========================================================

st.sidebar.title("Physiological Parameters")

weight = st.sidebar.slider("Body weight (kg)", 40, 120, 70)

egp_base = st.sidebar.slider(
    "Basal endogenous glucose production (mg/kg/min)",
    2.0, 3.5, 2.9
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose utilization (mg/kg/min)",
    1.0, 1.5, 1.2
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    150, 220, 180
)

st.sidebar.divider()
st.sidebar.title("Meal & Simulation")

meal_carbs = st.sidebar.slider("Meal carbohydrates (g)", 0, 150, 60)
meal_time = st.sidebar.slider("Meal time (hours)", 0, 24, 8)
simulation_hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)
noise_level = st.sidebar.slider("Physiological variability", 0.0, 4.0, 1.0)

# =========================================================
# Constants
# =========================================================

dt = 1  # min
time = np.arange(0, simulation_hours * 60, dt)

Vd = 0.20  # L/kg
GFR = 1.7  # mL/kg/min

# =========================================================
# Meal Absorption (slower, physiologic)
# =========================================================

def meal_absorption(t, meal_time_h, carbs_g, weight):
    if carbs_g == 0:
        return np.zeros_like(t)

    peak = meal_time_h * 60 + 60
    duration = 300  # 5 hours
    std = 60

    x = np.arange(duration)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel = kernel / kernel.sum()

    absorbed = carbs_g * 1000 * kernel  # mg
    signal = np.zeros_like(t)

    start = int(peak - duration / 2)
    end = start + duration

    if start >= 0 and end < len(t):
        signal[start:end] = absorbed

    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight)

# =========================================================
# Simulation
# =========================================================

G = np.zeros_like(time, dtype=float)
G[0] = 100

urinary_glucose = np.zeros_like(time)
ketones = np.zeros_like(time)

for i in range(1, len(time)):
    glucose = G[i - 1]

    # Dawn phenomenon (4–8 AM)
    hour = (time[i] / 60) % 24
    egp = egp_base * (1.15 if 4 <= hour <= 8 else 1.0)

    # Insulin-independent uptake (nonlinear)
    if glucose > 180:
        multiplier = 1 + 0.01 * (glucose - 180)
        u_ii = brain_uptake * multiplier
    else:
        u_ii = brain_uptake

    # Renal glucose excretion
    if glucose > renal_threshold:
        filtered = GFR * glucose
        reabsorbed = GFR * renal_threshold
        excreted_mg_min = max(filtered - reabsorbed, 0)
        renal_mgkgmin = excreted_mg_min / weight
    else:
        renal_mgkgmin = 0

    # Net glucose mass balance
    dG_mass = egp + meal_signal[i] - u_ii - renal_mgkgmin

    # Convert to concentration
    dG = dG_mass / (Vd * 10)

    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG + noise, 40)

    urinary_glucose[i] = urinary_glucose[i - 1] + renal_mgkgmin * weight

    # Simple ketone accumulation
    if glucose > 250:
        ketones[i] = ketones[i - 1] + 0.01
    else:
        ketones[i] = max(ketones[i - 1] - 0.005, 0)

# =========================================================
# Results
# =========================================================

df = pd.DataFrame({
    "Time (hours)": time / 60,
    "Glucose (mg/dL)": G,
    "Cumulative urinary glucose (g)": urinary_glucose / 1000,
    "Ketone index (a.u.)": ketones
})

# =========================================================
# Visualization
# =========================================================

st.title("Untreated Type 1 Diabetes – Physiology-Based Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (hours)"], df["Glucose (mg/dL)"], linewidth=2)
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =========================================================
# Metrics
# =========================================================

st.subheader("Clinical & Physiological Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Peak glucose (mg/dL)", f"{np.max(G):.0f}")
col2.metric("Urinary glucose loss (g)", f"{urinary_glucose[-1] / 1000:.1f}")
col3.metric("Ketone accumulation", f"{ketones[-1]:.2f}")

# =========================================================
# Interpretation
# =========================================================

st.subheader("Educational Interpretation")

if ketones[-1] > 1.0:
    st.error(
        "Sustained hyperglycemia with progressive ketone accumulation indicates "
        "high risk of diabetic ketoacidosis without insulin."
    )
elif np.mean(G > 250) * simulation_hours > 2:
    st.warning(
        "Prolonged glucose above 250 mg/dL demonstrates failure of renal compensation."
    )
else:
    st.info(
        "Even with renal glucose loss, insulin absence leads to progressive dysregulation."
    )

st.markdown("""
### Teaching Points
- Endogenous glucose production remains unopposed
- Renal glucose excretion delays but cannot prevent hyperglycemia
- Insulin-independent uptake saturates at high glucose
- Dawn phenomenon worsens morning hyperglycemia
- Ketogenesis reflects insulin deficiency, not glucose alone
""")

# =========================================================
# Export
# =========================================================

st.subheader("Download Simulation Data")

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False),
    file_name="untreated_t1dm_physiology_simulation.csv",
    mime="text/csv"
)
