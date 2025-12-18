import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Untreated T1DM Simulator", layout="wide")

# ===============================
# Sidebar – Parameters
# ===============================
st.sidebar.header("Patient Parameters")

weight = st.sidebar.slider("Body weight (kg)", 30, 120, 70)

egp_baseline = st.sidebar.slider(
    "Baseline EGP (mg/kg/min)",
    1.5, 3.0, 1.6,
    help="Normal basal EGP ≈1.5–2.0 mg/kg/min. "
         "In untreated T1DM, EGP increases up to ~60% due to unopposed glucagon."
)

gfr = st.sidebar.slider(
    "GFR (mL/kg/min)",
    1.0, 2.5, 1.7,
    help="Normal adult GFR ≈1.5–2.0 mL/kg/min. "
         "Lower values simulate renal impairment."
)

vd = st.sidebar.slider(
    "Glucose distribution volume (L/kg)",
    0.15, 0.25, 0.20
)

renal_threshold = st.sidebar.slider("Renal threshold (mg/dL)", 160, 220, 180)
tmax = st.sidebar.slider("Tubular max reabsorption (mg/min)", 250, 450, 375)

brain_util = st.sidebar.slider(
    "Insulin-independent glucose utilization (mg/kg/min)",
    0.8, 2.5, 1.5
)

st.sidebar.header("Meal")

carbs_g = st.sidebar.slider("Carbohydrate load (g)", 0, 150, 60)

meal_fraction = st.sidebar.slider(
    "Fraction absorbed by 5h",
    0.6, 0.9, 0.8,
    help="Mixed meals ≈60%; pure carbohydrate loads may reach 80–90%."
)

hours = st.sidebar.slider("Simulation duration (hours)", 6, 48, 24)

# ===============================
# Time
# ===============================
dt = 5  # minutes
time = np.arange(0, hours * 60 + dt, dt)
n = len(time)

# ===============================
# State variables
# ===============================
glucose = np.zeros(n)
ketones = np.zeros(n)
urinary_glucose = np.zeros(n)
urine_volume = np.zeros(n)
egp_cumulative = np.zeros(n)

glucose[0] = 100
ketones[0] = 0.3

# ===============================
# Meal absorption
# ===============================
meal_curve = np.zeros(n)
if carbs_g > 0:
    total_absorbed = carbs_g * 1000 * meal_fraction
    tau = 90
    for i, t in enumerate(time):
        meal_curve[i] = total_absorbed * (t / tau) * np.exp(1 - t / tau)
    meal_curve /= meal_curve.sum()

# ===============================
# Simulation loop
# ===============================
for i in range(1, n):
    h = time[i] / 60

    # Progressive EGP (+60% max)
    egp_multiplier = min(1 + 0.1 * h, 1.6)
    egp = egp_baseline * egp_multiplier
    egp_mg = egp * weight * dt

    egp_cumulative[i] = egp_cumulative[i - 1] + egp_mg / 1000

    # Insulin-independent utilization
    util = brain_util * weight * dt
    if glucose[i - 1] > 250:
        util *= 1.2

    # Meal input
    meal = meal_curve[i] if carbs_g > 0 else 0

    # Renal glucose handling
    filtered = glucose[i - 1] * gfr * weight * dt / 100
    reabsorbed = min(filtered, tmax * dt)
    urinary = max(filtered - reabsorbed, 0)

    urinary_glucose[i] = urinary

    # Osmotic diuresis (18 mL per gram)
    urine_volume[i] = (urinary / 1000) * 0.018

    # Glucose balance
    dG = egp_mg + meal - util - urinary
    glucose[i] = glucose[i - 1] + (dG / (vd * weight)) / 10

    # Ketogenesis (autocatalytic)
    if glucose[i] > 200:
        ketone_rate = 0.0015 * (glucose[i] - 200) * (1 + ketones[i - 1] / 10)
        ketones[i] = ketones[i - 1] + ketone_rate * dt / 60
    else:
        ketones[i] = max(ketones[i - 1] - 0.01, 0.1)

# ===============================
# Metrics
# ===============================
roc = np.diff(glucose) / (dt / 60)
max_roc = roc.max() if len(roc) > 0 else 0

# ===============================
# Plots
# ===============================
st.title("Untreated Type 1 Diabetes Simulator")

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

# ===============================
# Educational indicators
# ===============================
st.subheader("Educational Indicators")

st.metric("Max glucose rate of rise (mg/dL/hour)", f"{max_roc:.1f}")
st.metric("Cumulative endogenous glucose production (g)", f"{egp_cumulative[-1]:.1f}")
st.metric("Total urinary glucose loss (g)", f"{urinary_glucose.sum() / 1000:.1f}")
st.metric("Urine volume from glycosuria (L)", f"{urine_volume.sum():.2f}")

# ===============================
# Interpretation
# ===============================
st.subheader("Physiological Interpretation")

if max_roc > 60:
    st.warning(
        "Glucose rate of rise exceeds the 99th percentile (>60 mg/dL/hour), "
        "indicating severe metabolic dysregulation."
    )

if glucose.max() > 250:
    st.warning(
        "Sustained hyperglycemia demonstrates unopposed hepatic glucose production "
        "and insufficient renal compensation."
    )

if ketones.max() >= 3.0:
    st.error(
        "Ketones reached ≥3.0 mmol/L, consistent with diabetic ketoacidosis risk."
    )

st.info(
    "This simulator is intended for educational use. It demonstrates why insulin "
    "is physiologically mandatory in type 1 diabetes."
)
