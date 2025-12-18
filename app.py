import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import gaussian

st.set_page_config(page_title="Untreated T1DM Simulator", layout="wide")

# =============================
# Sidebar – Model Parameters
# =============================

st.sidebar.title("Model Parameters")

weight = st.sidebar.slider("Body weight (kg)", 40, 120, 70)

egp = st.sidebar.slider(
    "Endogenous Glucose Production (mg/kg/min)",
    1.5, 3.5, 2.4
)

brain_uptake = st.sidebar.slider(
    "Insulin-independent glucose utilization (mg/kg/min)",
    0.8, 2.5, 1.2
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    150, 220, 180
)

renal_clearance = st.sidebar.slider(
    "Renal glucose clearance above threshold (mg/dL/min)",
    0.5, 5.0, 2.0
)

meal_carbs = st.sidebar.slider(
    "Meal carbohydrates (g)",
    0, 150, 60
)

meal_time = st.sidebar.slider(
    "Meal time (hours)",
    0, 24, 8
)

simulation_hours = st.sidebar.slider(
    "Simulation duration (hours)",
    6, 48, 24
)

noise_level = st.sidebar.slider(
    "Physiological variability (stochasticity)",
    0.0, 5.0, 1.0
)

# =============================
# Time Grid
# =============================

dt = 1  # minute
time = np.arange(0, simulation_hours * 60, dt)

# =============================
# Meal Absorption Function
# =============================

def meal_absorption(t, meal_time_h, carbs_g):
    if carbs_g == 0:
        return np.zeros_like(t)
    peak_time = meal_time_h * 60 + 45
    kernel = gaussian(240, std=40)
    kernel = kernel / kernel.sum()
    absorption = carbs_g * 1000 * kernel  # mg
    signal = np.zeros_like(t)
    start = int(peak_time - len(kernel) / 2)
    if start > 0 and start + len(kernel) < len(t):
        signal[start:start + len(kernel)] = absorption
    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs)

# =============================
# Glucose Dynamics
# =============================

G = np.zeros_like(time, dtype=float)
G[0] = 100  # initial glucose mg/dL

for i in range(1, len(time)):
    glucose = G[i - 1]

    # Insulin-independent utilization (nonlinear)
    u_ii = brain_uptake * (1 + 0.002 * max(glucose - 180, 0))

    # Renal excretion
    renal = renal_clearance * max(glucose - renal_threshold, 0)

    dG = (
        egp
        + meal_signal[i]
        - u_ii
        - renal
    )

    # Convert mg/kg/min → mg/dL/min (distribution volume approx.)
    dG = dG / 10

    # Stochastic noise
    noise = np.random.normal(0, noise_level)

    G[i] = max(glucose + dG + noise, 40)

# =============================
# Output Data
# =============================

df = pd.DataFrame({
    "Time (hours)": time / 60,
    "Glucose (mg/dL)": G
})

# =============================
# Visualization
# =============================

st.title("Untreated Type 1 Diabetes – Educational Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (hours)"], df["Glucose (mg/dL)"])
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =============================
# Educational Metrics
# =============================

time_hyper_180 = np.mean(G > 180) * simulation_hours
time_hyper_250 = np.mean(G > 250) * simulation_hours

st.subheader("Clinical Indicators")

st.metric("Time > 180 mg/dL (hours)", f"{time_hyper_180:.1f}")
st.metric("Time > 250 mg/dL (hours)", f"{time_hyper_250:.1f}")
st.metric("Peak glucose (mg/dL)", f"{np.max(G):.0f}")

# =============================
# Educational Feedback
# =============================

st.subheader("Educational Interpretation")

if time_hyper_250 > 2:
    st.error(
        "Sustained glucose >250 mg/dL suggests high risk of ketoacidosis. "
        "Without insulin, hepatic glucose production remains unopposed."
    )
elif time_hyper_180 > 4:
    st.warning(
        "Persistent hyperglycemia demonstrates the absolute insulin requirement in T1DM."
    )
else:
    st.info(
        "Even short simulations show progressive glucose rise without insulin."
    )

st.markdown("""
### Key Teaching Points
- No endogenous insulin → no suppression of hepatic glucose production  
- Brain glucose uptake alone cannot compensate  
- Renal excretion delays but does not prevent hyperglycemia  
- Variability exists even under identical parameters  
""")
