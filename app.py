import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# App configuration
# =========================================================

st.set_page_config(
    page_title="Untreated Type 1 Diabetes Simulator",
    layout="wide"
)

# =========================================================
# Sidebar – Parameters
# =========================================================

st.sidebar.title("Physiology (Untreated T1DM)")

weight = st.sidebar.slider("Body weight (kg)", 40, 120, 70)

egp = st.sidebar.slider(
    "Endogenous glucose production – EGP (mg/kg/min)",
    2.0, 3.8, 2.9,
    help="Typically elevated (≈40–60%) in untreated T1DM"
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

noise_level = st.sidebar.slider(
    "Physiological variability",
    0.0, 5.0, 1.0
)

# =========================================================
# Constants (physiology-based)
# =========================================================

Vd = 0.20        # L/kg – glucose distribution volume
GFR = 1.7        # mL/kg/min – glomerular filtration rate
DT = 1           # minute time step

# =========================================================
# Time grid
# =========================================================

time = np.arange(0, simulation_hours * 60, DT)

# =========================================================
# Meal absorption model (extended, slow absorption)
# =========================================================

def meal_absorption(t, meal_time_h, carbs_g, weight):
    if carbs_g == 0:
        return np.zeros_like(t)

    duration = 300  # 5 hours
    std = 60
    peak = meal_time_h * 60 + 60

    x = np.arange(duration)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel = kernel / kernel.sum()

    absorbed = carbs_g * 1000 * 0.6 * kernel  # only ~60% absorbed in 5h
    signal = np.zeros_like(t)

    start = int(peak - duration / 2)
    end = start + duration

    if start >= 0 and end < len(t):
        signal[start:end] = absorbed

    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight)

# =========================================================
# Simulation variables
# =========================================================

G = np.zeros_like(time, dtype=float)
G[0] = 100

urinary_glucose = np.zeros_like(time)
ketones = np.zeros_like(time)

# =========================================================
# Simulation loop
# =========================================================

for i in range(1, len(time)):
    glucose = G[i - 1]

    # Dawn phenomenon (4–8 AM)
    hour = (time[i] / 60) % 24
    dawn_multiplier = 1.15 if 4 <= hour <= 8 else 1.0
    egp_effective = egp * dawn_multiplier

    # Insulin-independent uptake (strong nonlinear increase)
    if glucose > 180:
        multiplier = 1 + 0.01 * (glucose - 180)
        u_ii = brain_uptake * multiplier
    else:
        u_ii = brain_uptake

    # Renal glucose excretion (physiologic)
    if glucose > renal_threshold:
        filtered = GFR * glucose           # mg/kg/min
        reabsorbed = GFR * renal_threshold
        renal_mg_kg_min = filtered - reabsorbed
    else:
        renal_mg_kg_min = 0

    urinary_glucose[i] = urinary_glucose[i - 1] + renal_mg_kg_min * DT

    # Net glucose mass change
    dG_mass = (
        egp_effective
        + meal_signal[i]
        - u_ii
        - renal_mg_kg_min
    )

    # Convert mass → concentration
    dG_conc = dG_mass / (Vd * 10)

    # Ketone accumulation (simplified)
    if glucose > 250:
        ketones[i] = ketones[i - 1] + 0.01 * DT
    else:
        ketones[i] = max(ketones[i - 1] - 0.005 * DT, 0)

    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG_conc + noise, 40)

# =========================================================
# DataFrame
# =========================================================

df = pd.DataFrame({
    "Time (hours)": time / 60,
    "Glucose (mg/dL)": G,
    "Cumulative urinary glucose (g)": urinary_glucose / 1000,
    "Ketone index": ketones
})

# =========================================================
# Visualization
# =========================================================

st.title("Untreated Type 1 Diabetes – Physiological Simulator")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Time (hours)"], df["Glucose (mg/dL)"], linewidth=2)
ax.axhline(180, linestyle="--", color="orange", label="Hyperglycemia")
ax.axhline(250, linestyle="--", color="red", label="DKA Risk")
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Glucose (mg/dL)")
ax.legend()
st.pyplot(fig)

# =========================================================
# Clinical indicators
# =========================================================

time_hyper_250 = np.mean(G > 250) * simulation_hours
dka_time = np.argmax(ketones > 1.0) / 60 if np.any(ketones > 1.0) else None

st.subheader("Clinical & Educational Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Peak glucose (mg/dL)", f"{np.max(G):.0f}")
c2.metric("Time >250 mg/dL (h)", f"{time_hyper_250:.1f}")
c3.metric("Urinary glucose loss (g)", f"{urinary_glucose[-1]/1000:.1f}")
c4.metric(
    "Estimated time to DKA (h)",
    f"{dka_time:.1f}" if dka_time else "Not reached"
)

# =========================================================
# Educational interpretation
# =========================================================

st.subheader("Physiological Interpretation")

if dka_time and dka_time < simulation_hours:
    st.error(
        "Progressive hyperglycemia with rising ketone production illustrates "
        "the inevitability of diabetic ketoacidosis without insulin."
    )
else:
    st.warning(
        "Even without reaching overt DKA, glucose rises continuously due to "
        "unopposed hepatic glucose production."
    )

st.markdown("""
### Key Teaching Points
- Endogenous insulin = 0 → hepatic glucose production remains elevated  
- Renal glucose excretion delays but cannot prevent hyperglycemia  
- Insulin-independent uptake increases but is insufficient  
- Dawn phenomenon worsens morning hyperglycemia  
- Ketone production reflects uncontrolled lipolysis  
""")

# =========================================================
# Export
# =========================================================

st.subheader("Download simulation data")

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name="untreated_t1dm_simulation.csv",
    mime="text/csv"
)
