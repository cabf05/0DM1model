import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# App Configuration
# =========================================================

st.set_page_config(
    page_title="Untreated Type 1 Diabetes Simulator",
    layout="wide"
)

# =========================================================
# Sidebar – Model Parameters
# =========================================================

st.sidebar.title("Physiological Parameters")

weight = st.sidebar.slider(
    "Body weight (kg)",
    min_value=40,
    max_value=120,
    value=70
)

egp = st.sidebar.slider(
    "Endogenous glucose production (mg/kg/min)",
    min_value=1.5,
    max_value=3.5,
    value=2.4
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose utilization (mg/kg/min)",
    min_value=0.8,
    max_value=2.5,
    value=1.2
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    min_value=150,
    max_value=220,
    value=180
)

renal_clearance = st.sidebar.slider(
    "Renal glucose clearance above threshold (mg/dL/min)",
    min_value=0.5,
    max_value=5.0,
    value=2.0
)

st.sidebar.divider()
st.sidebar.title("Meal & Simulation")

meal_carbs = st.sidebar.slider(
    "Meal carbohydrates (g)",
    min_value=0,
    max_value=150,
    value=60
)

meal_time = st.sidebar.slider(
    "Meal time (hours)",
    min_value=0,
    max_value=24,
    value=8
)

simulation_hours = st.sidebar.slider(
    "Simulation duration (hours)",
    min_value=6,
    max_value=48,
    value=24
)

noise_level = st.sidebar.slider(
    "Physiological variability",
    min_value=0.0,
    max_value=5.0,
    value=1.0
)

# =========================================================
# Time Grid
# =========================================================

dt = 1  # minute
time = np.arange(0, simulation_hours * 60, dt)

# =========================================================
# Meal Absorption Model (Gaussian-like, no SciPy)
# =========================================================

def meal_absorption(t, meal_time_h, carbs_g, weight):
    if carbs_g == 0:
        return np.zeros_like(t)

    peak_time = meal_time_h * 60 + 45
    duration = 240  # minutes
    std = 40

    x = np.arange(duration)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel = kernel / kernel.sum()

    absorption = carbs_g * 1000 * kernel  # mg
    signal = np.zeros_like(t)

    start = int(peak_time - duration / 2)
    end = start + duration

    if start >= 0 and end < len(t):
        signal[start:end] = absorption

    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight)

# =========================================================
# Glucose Dynamics Simulation
# =========================================================

G = np.zeros_like(time, dtype=float)
G[0] = 100  # initial glucose mg/dL

for i in range(1, len(time)):
    glucose = G[i - 1]

    # Insulin-independent utilization (nonlinear in hyperglycemia)
    u_ii = brain_uptake * (1 + 0.002 * max(glucose - 180, 0))

    # Renal glucose excretion
    renal = renal_clearance * max(glucose - renal_threshold, 0)

    # Net glucose change (mg/kg/min)
    dG = (
        egp
        + meal_signal[i]
        - u_ii
        - renal
    )

    # Convert to mg/dL/min (simplified distribution volume)
    dG = dG / 10

    # Add stochastic variability
    noise = np.random.normal(0, noise_level)

    G[i] = max(glucose + dG + noise, 40)

# =========================================================
# Results DataFrame
# =========================================================

df = pd.DataFrame({
    "Time (hours)": time / 60,
    "Glucose (mg/dL)": G
})

# =========================================================
# Visualization
# =========================================================

st.title("Untreated Type 1 Diabetes – Educational Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (hours)"], df["Glucose (mg/dL)"], linewidth=2)
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia (>180)")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk (>250)")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =========================================================
# Clinical Metrics
# =========================================================

time_hyper_180 = np.mean(G > 180) * simulation_hours
time_hyper_250 = np.mean(G > 250) * simulation_hours

st.subheader("Clinical Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Time > 180 mg/dL (h)", f"{time_hyper_180:.1f}")
col2.metric("Time > 250 mg/dL (h)", f"{time_hyper_250:.1f}")
col3.metric("Peak glucose (mg/dL)", f"{np.max(G):.0f}")

# =========================================================
# Educational Interpretation
# =========================================================

st.subheader("Educational Interpretation")

if time_hyper_250 > 2:
    st.error(
        "Sustained glucose above 250 mg/dL indicates high risk of diabetic ketoacidosis. "
        "Without insulin, hepatic glucose production remains unopposed."
    )
elif time_hyper_180 > 4:
    st.warning(
        "Persistent hyperglycemia demonstrates the absolute insulin requirement in type 1 diabetes."
    )
else:
    st.info(
        "Even in short simulations, glucose rises progressively without insulin."
    )

st.markdown("""
### Key Teaching Points
- No endogenous insulin secretion (β-cell function = 0)
- Hepatic glucose production is not suppressed
- Brain glucose uptake alone cannot normalize glycemia
- Renal glucose loss delays but does not prevent hyperglycemia
- Variability exists even under identical physiological conditions
""")

# =========================================================
# Data Export
# =========================================================

st.subheader("Download Simulation Data")
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="untreated_t1dm_simulation.csv",
    mime="text/csv"
)
