import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Untreated Type 1 Diabetes Simulator", layout="wide")

# =========================
# Sidebar – Parameters
# =========================
st.sidebar.header("Patient & Physiological Parameters")

weight = st.sidebar.slider(
    "Body weight (kg)",
    30, 120, 70
)

egp_baseline = st.sidebar.slider(
    "Baseline endogenous glucose production (mg/kg/min)",
    1.5, 3.0, 1.6,
    help="Normal basal EGP ~1.5–2.0 mg/kg/min. In untreated T1DM, "
         "EGP may increase by ~60% due to unopposed glucagon and "
         "counter-regulatory hormones."
)

gfr = st.sidebar.slider(
    "Glomerular filtration rate (mL/kg/min)",
    1.0, 2.5, 1.7,
    help="Normal adult GFR ≈ 1.5–2.0 mL/kg/min. "
         "Lower values simulate renal impairment."
)

vd = st.sidebar.slider(
    "Glucose distribution volume (L/kg)",
    0.15, 0.25, 0.20,
    help="Physiological glucose Vd ≈ 0.18–0.23 L/kg, representing "
         "the extracellular fluid space."
)

renal_threshold = st.sidebar.slider(
    "Renal glucose threshold (mg/dL)",
    160, 220, 180
)

tmax = st.sidebar.slider(
    "Max tubular glucose reabsorption (mg/min)",
    250, 450, 375
)

brain_util = st.sidebar.slider(
    "Insulin-independent glucose utilization (mg/kg/min)",
    0.8, 2.5, 1.5,
    help="Primarily brain uptake. Increases nonlinearly at "
         "very high glucose concentrations."
)

st.sidebar.header("Meal Simulation")

carbs_g = st.sidebar.slider(
    "Carbohydrate load (grams)",
    0, 150, 60
)

meal_absorption = st.sidebar.slider(
    "Fraction absorbed by 5h",
    0.6, 0.9, 0.8,
    help="Mixed meals ≈60%; pure carbohydrate loads may reach 80–90%."
)

st.sidebar.header("Simulation")

hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)
dt = 5  # minutes

# =========================
# Time Vector
# =========================
time = np.arange(0, hours * 60 + dt, dt)
n = len(time)

# =========================
# State Variables
# =========================
glucose = np.zeros(n)
ketones = np.zeros(n)
urinary_glucose = np.zeros(n)
urine_volume = np.zeros(n)
egp_cumulative = np.zeros(n)

glucose[0] = 100
ketones[0] = 0.3

# =========================
# Meal Absorption Curve
# =========================
meal_absorption_curve = np.zeros(n)
if carbs_g > 0:
    absorbed_total_mg = carbs_g * 1000 * meal_absorption
    tau = 90  # minutes
    for i, t in enumerate(time):
        meal_absorption_curve[i] = absorbed_total_mg * (t / tau) * np.exp(1 - t / tau)
    meal_absorption_curve = meal_absorption_curve / meal_absorption_curve.sum()

# =========================
# Simulation Loop
# =========================
for i in range(1, n):
    hours_elapsed = time[i] / 60

    # Progressive EGP increase (max +60%)
    egp_multiplier = min(1 + 0.1 * hours_elapsed, 1.6)
    egp = egp_baseline * egp_multiplier

    egp_mg = egp * weight * dt
    egp_cumulative[i] = egp_cumulative[i-1] + egp_mg / 1000

    # Insulin-independent utilization (nonlinear)
    util = brain_util * weight * dt
    if glucose[i-1] > 250:
        util *= 1.2

    # Meal input
    meal_input = meal_absorption_curve[i] if carbs_g > 0 else 0

    # Renal glucose handling
    filtered = glucose[i-1] * gfr * weight * dt / 100
    reabsorbed = min(filtered, tmax * dt)
    urinary = max(filtered - reabsorbed, 0)

    urinary_glucose[i] = urinary

    # Osmotic diuresis (18 mL per gram)
    urine_volume[i] = (urinary / 1000) * 0.018

    # Net glucose mass balance
    dG = egp_mg + meal_input - util - urinary

    # Convert mass → concentration
    glucose[i] = glucose[i-1] + (dG / (vd * weight)) / 10

    # Ketogenesis (nonlinear, autocatalytic)
    if glucose[i] > 200:
        ketone_rate = 0.0015 * (glucose[i] - 200) * (1 + ketones[i-1] / 10)
        ketones[i] = ketones[i-1] + ketone_rate * dt / 60
    else:
        ketones[i] = max(ketones[i-1] - 0.01, 0.1)

# =========================
# Metrics
# =========================
roc = np.diff(glucose) / (dt / 60)
max_roc = roc.max() if len(roc) > 0 else 0

# =========================
# Visualization
# =========================
st.title("Untreated Type 1 Diabetes Educational Simulator")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Plasma Glucose (mg/dL)")
    fig, ax = plt.subplots()
    ax.plot(time / 60, glucose)
    ax.axhline(180, linestyle="--")
    ax.axhline(250, linestyle="--")
    ax.set_xlabel("Hours")
    ax.set_ylabel("mg/dL")
    st.pyplot(fig)

with col2:
    st.subheader("Ketones (β-hydroxybutyrate, mmol/L)")
    fig, ax = plt.subplots()
    ax.plot(time / 60, ketones)
    ax.axhline(3.0, color="red", linestyle="--")
    ax.set_xlabel("Hours")
    ax.set_ylabel("mmol/L")
    st.pyplot(fig)

st.subheader("Educational Indicators")

st.metric(
    "Max glucose rate of rise (mg/dL/hour)",
    f"{max_roc:.1f}",
    delta="⚠️ Severe dysregulation" if max_roc > 60 else None
)

st.metric(
    "Total endogenous glucose produced (g)",
    f"{egp_cumulative[-1]:.1f}"
)

st.metric(
    "Total urinary glucose loss (g)",
    f"{urinary_glucose.sum() / 1000:.1f}"
)

st.metric(
    "Total urine volume from glycosuria (L)",
    f"{urine_volume.sum():.2f}"
)

# =========================
# Educational Messages
# =========================
st.subheader("Physiological Interpretation")

if glucose.max() > 250:
    st.warning(
        "Sustained hyperglycemia >250 mg/dL demonstrates "
        "unopposed hepatic glucose production and inadequate "
        "renal compensation without insulin."
    )

if ketones.max() >= 3.0:
    st.error(
        "Ketone levels reached the DKA threshold (≥3.0 mmol/L), "
        "illustrating the absolute requirement for insulin in T1DM."
    )

st.info(
    "This simulator is for educational purposes only. It demonstrates "
    "why insulin is physiologically mandatory in type 1 diabetes and "
    "why renal glucose loss and insulin-independent utilization "
    "cannot prevent metabolic decompensation."
)
