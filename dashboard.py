import streamlit as st
import matplotlib.pyplot as plt
from simulation import generate_data, apply_baseline, apply_kli_model, calculate_metrics

st.set_page_config(layout="wide")
st.title("📦 Kitchen Prep Time Simulation Dashboard")

# Sidebar Controls
st.sidebar.header("Simulation Controls")

num_orders = st.sidebar.slider("Number of Orders", 1000, 30000, 15000)
avg_active_orders = st.sidebar.slider("Average Active Orders", 1, 15, 6)
peak_ratio = st.sidebar.slider("Peak Hour Probability", 0.0, 1.0, 0.35)
kli_weight = st.sidebar.slider("KLI Weight", 0.1, 1.0, 0.7)

# Generate Data
data = generate_data(num_orders, avg_active_orders, peak_ratio)
data = apply_baseline(data)
data = apply_kli_model(data, kli_weight)

metrics = calculate_metrics(data)

# KPI Display
col1, col2, col3 = st.columns(3)

col1.metric("MAE Current", round(metrics["mae_current"], 2))
col2.metric("MAE Proposed", round(metrics["mae_proposed"], 2))
col3.metric("Improvement %",
            round((metrics["mae_current"] - metrics["mae_proposed"]) /
                  metrics["mae_current"] * 100, 1))

col4, col5, col6 = st.columns(3)

col4.metric("Avg Rider Wait Current",
            round(metrics["avg_wait_current"], 2))
col5.metric("Avg Rider Wait Proposed",
            round(metrics["avg_wait_proposed"], 2))
col6.metric("P90 Error Reduction",
            round(metrics["p90_current"] - metrics["p90_proposed"], 2))

# Charts
st.subheader("MAE Comparison")

plt.figure()
plt.bar(["Current", "Proposed"],
        [metrics["mae_current"], metrics["mae_proposed"]])
st.pyplot(plt)

st.subheader("Rider Wait Distribution")

plt.figure()
plt.hist(data["wait_current"], bins=40)
st.pyplot(plt)

plt.figure()
plt.hist(data["wait_proposed"], bins=40)
st.pyplot(plt)

st.subheader("System Insight")
st.write("""
This dashboard demonstrates congestion-aware correction of merchant-reported
prep times using a Kitchen Load Index (KLI).
Adjust parameters to simulate rush-hour and congestion impact.
""")