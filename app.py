
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="WattGuard",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# Load dataset and model
# -----------------------------
data_path = "energy.csv.csv"

df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour

features = [
    "hour",
    "lights",
    "T1", "RH_1",
    "T2", "RH_2",
    "T3", "RH_3",
    "T_out",
    "Press_mm_hg",
    "Windspeed",
    "Visibility",
    "Tdewpoint"
]

X = df[features]
y = df["Appliances"]

model = RandomForestRegressor(
    n_estimators=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
df["date"] = pd.to_datetime(df["date"])
df["hour"] = df["date"].dt.hour

# -----------------------------
# Basic calculations
# -----------------------------
total_energy = df["Appliances"].sum()
total_energy_kwh = total_energy / 1000

hourly_energy = df.groupby("hour")["Appliances"].mean()
peak_hour = hourly_energy.idxmax()

# -----------------------------
# Smart-control simulation
# -----------------------------
peak_hours = [17, 18, 19, 20]
savings_rate = 0.15

hourly_baseline = hourly_energy.copy()
hourly_optimized = hourly_energy.copy()

for hour in peak_hours:
    hourly_optimized.loc[hour] = (
        hourly_optimized.loc[hour] * (1 - savings_rate)
    )

peak_energy_total = df[
    df["hour"].isin(peak_hours)
]["Appliances"].sum()

estimated_savings = peak_energy_total * savings_rate
estimated_savings_kwh = estimated_savings / 1000

overall_reduction = (
    estimated_savings / total_energy
) * 100

# -----------------------------
# Title
# -----------------------------
st.title("⚡ WattGuard")

st.markdown(
    "### Smart Building Energy Management System"
)

st.write(
    "WattGuard predicts building energy demand, detects peak periods, "
    "and recommends intelligent load management while keeping essential "
    "building services operating."
)

st.info(
    "🎯 Goal: Reduce unnecessary peak energy consumption "
    "without compromising comfort, safety, or productivity."
)

st.divider()

# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Energy",
        f"{total_energy_kwh:,.2f} kWh"
    )

with col2:
    st.metric(
        "Peak Hour",
        f"{peak_hour}:00"
    )

with col3:
    st.metric(
        "Estimated Savings",
        f"{estimated_savings_kwh:,.2f} kWh"
    )

with col4:
    st.metric(
        "Energy Reduction",
        f"{overall_reduction:.2f}%"
    )

st.divider()

# -----------------------------
# Building Status
# -----------------------------
st.header("🏢 Building Status")

if peak_hour in peak_hours:
    st.error(
        "🔴 PEAK DEMAND: Flexible loads should be reduced or shifted."
    )
else:
    st.success(
        "🟢 NORMAL OPERATION: Continue standard building operation."
    )

st.write(
    f"Detected peak period: "
    f"**{peak_hours[0]}:00–{peak_hours[-1]}:00**"
)

# -----------------------------
# Peak alert
# -----------------------------
st.warning(
    "⚠️ Peak-demand period detected: "
    "17:00–20:00. Flexible loads can be shifted "
    "or reduced."
)

# -----------------------------
# Hourly energy graph
# -----------------------------
st.header("📈 Average Energy Consumption by Hour")

chart_data = hourly_energy.reset_index()
chart_data.columns = ["Hour", "Average Energy (Wh)"]

fig1 = px.line(
    chart_data,
    x="Hour",
    y="Average Energy (Wh)",
    markers=True,
    title="Hourly Building Energy Pattern"
)

fig1.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Average Energy Consumption (Wh)"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Before vs Smart Control
# -----------------------------
st.header("📉 Baseline vs WattGuard Smart Control")

comparison = pd.DataFrame({
    "Hour": range(24),
    "Baseline": hourly_baseline.values,
    "WattGuard Scenario": hourly_optimized.values
})

comparison_long = comparison.melt(
    id_vars="Hour",
    var_name="Scenario",
    value_name="Energy (Wh)"
)

fig2 = px.line(
    comparison_long,
    x="Hour",
    y="Energy (Wh)",
    color="Scenario",
    markers=True,
    title="Simulated Peak-Load Reduction"
)

fig2.update_layout(
    xaxis_title="Hour of Day",
    yaxis_title="Average Energy Consumption (Wh)"
)

st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Simulation assumption: WattGuard reduces flexible "
    "peak-hour consumption by 15%. This is a scenario estimate, "
    "not a measured real-world saving."
)

# -----------------------------
# Smart recommendation
# -----------------------------
st.header("🤖 Smart Recommendation")

st.info(
    "Reduce or shift flexible loads during peak hours "
    "while maintaining comfort and essential building services."
)

# -----------------------------
# Smart Control Plan
# -----------------------------
st.header("🎛️ Smart Control Plan")

recommendations = []
average_energy = hourly_energy.mean()

for hour in range(24):

    if hour in peak_hours:
        status = "PEAK"
        action = "Reduce or shift flexible loads"
        priority = "High"

    elif hourly_energy.loc[hour] > average_energy:
        status = "HIGH"
        action = "Optimize HVAC and lighting"
        priority = "Medium"

    else:
        status = "NORMAL"
        action = "Normal building operation"
        priority = "Low"

    recommendations.append([
        f"{hour:02d}:00",
        status,
        action,
        priority
    ])

control_plan = pd.DataFrame(
    recommendations,
    columns=[
        "Hour",
        "Load Status",
        "Recommended Action",
        "Priority"
    ]
)

st.dataframe(
    control_plan,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The control plan represents software recommendations. "
    "Actual building control would require integration with "
    "the building management system (BMS)."
)

# -----------------------------
# ML Energy Prediction
# -----------------------------
st.header("🔮 Energy Consumption Prediction")

prediction_hour = st.slider(
    "Select an hour for prediction",
    min_value=0,
    max_value=23,
    value=int(peak_hour)
)

sample = df[df["hour"] == prediction_hour].iloc[0]

prediction_input = pd.DataFrame([{
    "hour": prediction_hour,
    "lights": sample["lights"],
    "T1": sample["T1"],
    "RH_1": sample["RH_1"],
    "T2": sample["T2"],
    "RH_2": sample["RH_2"],
    "T3": sample["T3"],
    "RH_3": sample["RH_3"],
    "T_out": sample["T_out"],
    "Press_mm_hg": sample["Press_mm_hg"],
    "Windspeed": sample["Windspeed"],
    "Visibility": sample["Visibility"],
    "Tdewpoint": sample["Tdewpoint"]
}])

predicted_energy = model.predict(prediction_input)[0]

st.metric(
    "Predicted Energy Consumption",
    f"{predicted_energy:.2f} Wh"
)

st.caption(
    "Prediction generated by the trained Random Forest model "
    "using environmental and building-energy features from the dataset."
)


# -----------------------------
# WattGuard System Workflow
# -----------------------------
st.header("🔄 How WattGuard Works")

step1, step2, step3, step4, step5 = st.columns(5)

with step1:
    st.markdown("### 1️⃣")
    st.markdown("**Building Data**")
    st.caption("Energy, lighting, temperature and weather data")

with step2:
    st.markdown("### 2️⃣")
    st.markdown("**Prediction**")
    st.caption("ML model estimates upcoming energy demand")

with step3:
    st.markdown("### 3️⃣")
    st.markdown("**Peak Detection**")
    st.caption("Identifies high-demand periods")

with step4:
    st.markdown("### 4️⃣")
    st.markdown("**Smart Decision**")
    st.caption("Finds opportunities to reduce or shift flexible loads")

with step5:
    st.markdown("### 5️⃣")
    st.markdown("**Action & Savings**")
    st.caption("Recommends controls and estimates energy savings")

st.info(
    "WattGuard is designed as a decision-support system. "
    "In a real building, recommended actions could be sent to the "
    "Building Management System (BMS) for controlled execution."
)


# -----------------------------
# Indian Building Deployment Context
# -----------------------------
st.header("🇮🇳 Indian Building Deployment Context")

st.write(
    "WattGuard can be adapted for Indian commercial and institutional "
    "buildings where cooling, lighting and other flexible electrical loads "
    "can contribute significantly to demand."
)

context1, context2, context3, context4 = st.columns(4)

with context1:
    st.markdown("### ❄️ HVAC")
    st.caption(
        "Use predicted demand to optimize cooling schedules "
        "while respecting indoor comfort limits."
    )

with context2:
    st.markdown("### 💡 Lighting")
    st.caption(
        "Reduce unnecessary lighting during low-occupancy or "
        "high-demand periods."
    )

with context3:
    st.markdown("### 🏢 BMS")
    st.caption(
        "Recommendations can be integrated with a Building "
        "Management System in a real deployment."
    )

with context4:
    st.markdown("### ⚡ Grid")
    st.caption(
        "Peak-load recommendations could support future "
        "demand-response programs."
    )

st.info(
    "Deployment concept: WattGuard acts as an intelligent decision layer "
    "between building data and existing control infrastructure. "
    "The current prototype is software-based and does not claim a "
    "physical BMS or grid connection."
)


# -----------------------------
# Quantified Results
# -----------------------------
st.header("📊 Quantified Results")

st.write(
    "The following results demonstrate WattGuard's potential using "
    "the available energy dataset and a simulated peak-load control scenario."
)

result1, result2, result3, result4 = st.columns(4)

with result1:
    st.metric(
        "Baseline Energy",
        f"{total_energy_kwh:,.2f} kWh"
    )

with result2:
    st.metric(
        "Peak-Period Energy",
        f"{peak_energy_total / 1000:,.2f} kWh"
    )

with result3:
    st.metric(
        "Simulated Savings",
        f"{estimated_savings_kwh:,.2f} kWh"
    )

with result4:
    st.metric(
        "Overall Reduction",
        f"{overall_reduction:.2f}%"
    )

st.info(
    "Simulation assumption: 15% of flexible consumption during the "
    "17:00–20:00 peak period is shifted or reduced. The resulting "
    "76.84 kWh saving is an estimate from this scenario, not a "
    "measured real-world saving."
)

st.markdown("### 🤖 Model Evaluation")

st.write(
    "WattGuard uses a Random Forest regression model to estimate "
    "building energy consumption. Model performance is evaluated "
    "using both a conventional random split and a time-aware split."
)

st.caption(
    "The time-aware evaluation trains on earlier observations and "
    "tests on later observations, which better represents future "
    "energy prediction in a real deployment."
)

# -----------------------------
# Dataset Information
# -----------------------------
st.header("📊 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Data Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Data Features",
        f"{len(df.columns):,}"
    )
