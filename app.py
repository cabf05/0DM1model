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
         "glucagon and counter-regulatory hormones (Voss 2019). "
         "Use 2.4-2.9 to simulate insulin withdrawal."
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
# Physiological constants
# =========================================================
DT = 1.0                    # min
Vd = 0.20                   # L/kg (glucose distribution volume)
TMAX_GLUCOSE = 375          # mg/min (max tubular reabsorption)

# =========================================================
# Time grid
# =========================================================
time = np.arange(0, simulation_hours * 60, DT)
n_steps = len(time)

# =========================================================
# Meal absorption
# =========================================================
def meal_absorption(t, meal_time_h, carbs_g, weight, fraction):
    """Models carbohydrate absorption as Gaussian distribution (peak ~1h, duration 5h)"""
    if carbs_g == 0:
        return np.zeros_like(t, dtype=float)
    duration = 300  # 5 hours in minutes
    std = 60
    peak = meal_time_h * 60 + 60
    x = np.arange(duration, dtype=float)
    kernel = np.exp(-0.5 * ((x - duration / 2) / std) ** 2)
    kernel /= kernel.sum()
    absorbed = carbs_g * 1000 * fraction * kernel
    signal = np.zeros_like(t, dtype=float)
    start = int(peak - duration / 2)
    end = start + duration
    if start >= 0 and end < len(t):
        signal[start:end] = absorbed
    elif start < len(t):
        signal[start:] = absorbed[:len(t)-start]
    return signal / weight  # mg/kg/min

meal_signal = meal_absorption(time, meal_time, meal_carbs, weight, meal_absorption_fraction)

# =========================================================
# State variables
# =========================================================
G = np.zeros(n_steps, dtype=float)
G[0] = 100.0

ketones = np.zeros(n_steps, dtype=float)
urinary_glucose = np.zeros(n_steps, dtype=float)   # mg total
urine_volume = np.zeros(n_steps, dtype=float)      # L total
glucose_roc = np.zeros(n_steps, dtype=float)       # mg/dL/h
cumulative_egp = np.zeros(n_steps, dtype=float)    # g total
egp_array = np.zeros(n_steps, dtype=float)         # mg/kg/min at each time
uptake_array = np.zeros(n_steps, dtype=float)      # mg/kg/min at each time
renal_array = np.zeros(n_steps, dtype=float)       # mg/kg/min at each time

# =========================================================
# Simulation loop
# =========================================================
for i in range(1, n_steps):
    glucose = G[i - 1]

    # Progressive EGP increase
    hours_hyper = np.sum(G[:i] > 180) / 60.0
    egp_multiplier = min(1.0 + 0.1 * hours_hyper, 1.6)

    # Dawn phenomenon (4–8 AM)
    hour = (time[i] / 60.0) % 24
    dawn_multiplier = 1.15 if 4 <= hour <= 8 else 1.0

    egp = egp_baseline * egp_multiplier * dawn_multiplier
    egp_array[i] = egp
    cumulative_egp[i] = cumulative_egp[i-1] + egp * DT * weight / 1000.0  # grams

    # Insulin-independent glucose uptake
    if glucose > 180:
        u_ii = brain_uptake * (1.0 + 0.01 * (glucose - 180))
    else:
        u_ii = brain_uptake
    uptake_array[i] = u_ii

    # Renal glucose handling
    if glucose > renal_threshold:
        GFR_total = GFR * weight
        filtered = (GFR_total / 1000.0) * glucose
        reabsorbed = min(filtered, TMAX_GLUCOSE / 1000.0)
        excreted = max(filtered - reabsorbed, 0.0)
        renal_mg_kg_min = (excreted * 1000.0) / weight
    else:
        renal_mg_kg_min = 0.0
    renal_array[i] = renal_mg_kg_min

    urinary_glucose[i] = urinary_glucose[i-1] + renal_mg_kg_min * weight * DT
    urine_volume[i] = (urinary_glucose[i] / 1000.0) * 0.018  # 18 mL per gram

    # Ketogenesis
    if glucose > 200:
        ketone_rate = 0.0015 * (glucose - 200) * (1 + ketones[i-1]/10.0)
        ketones[i] = ketones[i-1] + ketone_rate * DT
    else:
        ketones[i] = max(ketones[i-1] - 0.002 * DT, 0.0)

    # Glucose mass balance
    dG_mass = egp + meal_signal[i] - u_ii - renal_mg_kg_min
    dG_conc = (dG_mass / Vd) / 10.0
    noise = np.random.normal(0, noise_level)
    G[i] = max(glucose + dG_conc + noise, 40.0)

    if i >= 60:
        glucose_roc[i] = G[i] - G[i-60]

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
# Visualization
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
# Clinical indicators
# =========================================================
dka_time = (np.argmax(ketones >= 3.0) / 60.0) if np.any(ketones >= 3.0) else None
time_in_target = np.mean((G >= 70) & (G <= 180)) * 100
time_above_180 = np.mean(G > 180) * simulation_hours
time_above_250 = np.mean(G > 250) * simulation_hours
max_roc = np.max(np.abs(glucose_roc))

st.subheader("Clinical & Educational Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Peak glucose", f"{np.max(G):.0f} mg/dL")
c2.metric("Time in range", f"{time_in_target:.0f}%")
c3.metric("Max glucose ROC", f"{max_roc:.0f} mg/dL/h")
c4.metric("Urinary glucose loss", f"{urinary_glucose[-1]/1000:.1f} g")
c5.metric("Cumulative EGP", f"{cumulative_egp[-1]:.0f} g")
c6.metric("Time to DKA", f"{dka_time:.1f} h" if dka_time else "Not reached")

c7, c8, c9, c10 = st.columns(4)
c7.metric("Urine volume", f"{urine_volume[-1]:.1f} L")
c8.metric("Peak ketones", f"{np.max(ketones):.2f} mmol/L")
c9.metric("Time >180 mg/dL", f"{time_above_180:.1f} h")
c10.metric("Time >250 mg/dL", f"{time_above_250:.1f} h")

# =========================================================
# Interpretation & Teaching Points
# =========================================================
st.subheader("Physiological Interpretation")
with st.expander("📚 Key Teaching Points"):
    st.markdown("""
**Pathophysiology of Untreated T1DM:**

- **Absolute insulin deficiency** → Unopposed hepatic glucose production
- **Progressive EGP elevation** → Simulates counter-regulatory hormone effects (glucagon, cortisol)
- **Renal compensation limits** → Glucosuria cannot prevent hyperglycemia above ~250 mg/dL
- **Insulin-independent uptake** → Brain glucose utilization increases but is insufficient
- **Ketogenesis** → Uncontrolled lipolysis produces β-hydroxybutyrate (DKA threshold: 3.0 mmol/L)
- **Osmotic diuresis** → Each gram of glucose excreted carries ~18 mL of water

**Clinical Correlates:**

- Glucose ROC >56 mg/dL/h exceeds 99th percentile for health
- DKA develops within 14-18 hours of complete insulin withdrawal
- Polyuria (>14 cc/kg/h) reflects osmotic diuresis from glucosuria
""")

interpretations = []
if dka_time:
    interpretations.append(("error",
        f"β-hydroxybutyrate exceeds 3.0 mmol/L at {dka_time:.1f} hours, meeting clinical criteria for DKA. "
        "Reflects uncontrolled lipolysis and hepatic ketogenesis."
    ))
if max_roc > 56:
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
# Export CSV
# =========================================================
st.subheader("Download simulation data")
st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    file_name="untreated_t1dm_simulation.csv",
    mime="text/csv"
)
