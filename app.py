import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bottle Filling Quality Control Dashboard", layout="wide")

st.title("Bottle Filling Quality Control Dashboard")
st.write("Monitor a simulated bottle filling process using control charts and process capability analysis.")

# Sidebar inputs
st.sidebar.header("Process Settings")

target = st.sidebar.number_input("Target Fill Volume (ml)", value=500.0)
lsl = st.sidebar.number_input("Lower Specification Limit - LSL (ml)", value=495.0)
usl = st.sidebar.number_input("Upper Specification Limit - USL (ml)", value=505.0)

process_mean = st.sidebar.slider("Actual Process Mean (ml)", 490.0, 510.0, 500.0)
process_std = st.sidebar.slider("Process Standard Deviation (ml)", 0.5, 5.0, 2.0)

num_samples = st.sidebar.slider("Number of Samples", 10, 50, 25)
sample_size = st.sidebar.slider("Sample Size per Sample", 2, 10, 5)

simulate_shift = st.sidebar.checkbox("Simulate Process Shift")
if simulate_shift:
    shift_point = st.sidebar.slider("Shift Starts After Sample", 5, num_samples, 15)
    shift_amount = st.sidebar.slider("Shift Amount (ml)", -5.0, 5.0, 2.0)
else:
    shift_point = None
    shift_amount = 0

# Generate data
np.random.seed(42)

data = []
for i in range(num_samples):
    mean = process_mean
    if simulate_shift and i >= shift_point:
        mean += shift_amount

    sample = np.random.normal(mean, process_std, sample_size)
    for value in sample:
        data.append([i + 1, value])

df = pd.DataFrame(data, columns=["Sample", "Fill Volume"])

sample_stats = df.groupby("Sample").agg(
    xbar=("Fill Volume", "mean"),
    range=("Fill Volume", lambda x: x.max() - x.min())
).reset_index()

# Control chart constants for Xbar-R charts
constants = {
    2: (1.880, 0, 3.267),
    3: (1.023, 0, 2.574),
    4: (0.729, 0, 2.282),
    5: (0.577, 0, 2.114),
    6: (0.483, 0, 2.004),
    7: (0.419, 0.076, 1.924),
    8: (0.373, 0.136, 1.864),
    9: (0.337, 0.184, 1.816),
    10: (0.308, 0.223, 1.777)
}

A2, D3, D4 = constants[sample_size]

xbar_bar = sample_stats["xbar"].mean()
r_bar = sample_stats["range"].mean()

ucl_x = xbar_bar + A2 * r_bar
lcl_x = xbar_bar - A2 * r_bar

ucl_r = D4 * r_bar
lcl_r = D3 * r_bar

# Capability analysis
overall_mean = df["Fill Volume"].mean()
overall_std = df["Fill Volume"].std(ddof=1)

cp = (usl - lsl) / (6 * overall_std)
cpk = min((usl - overall_mean) / (3 * overall_std),
          (overall_mean - lsl) / (3 * overall_std))

def status_color(value):
    if value >= 1.33:
        return "Good"
    elif value >= 1.00:
        return "Acceptable"
    else:
        return "Poor"

# KPI cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Average Fill Volume", f"{overall_mean:.2f} ml")
col2.metric("Standard Deviation", f"{overall_std:.2f} ml")
col3.metric("Cp", f"{cp:.2f}", status_color(cp))
col4.metric("Cpk", f"{cpk:.2f}", status_color(cpk))

st.divider()

# Charts
left, right = st.columns(2)

with left:
    st.subheader("X-bar Control Chart")

    fig, ax = plt.subplots()
    ax.plot(sample_stats["Sample"], sample_stats["xbar"], marker="o")
    ax.axhline(xbar_bar, linestyle="--", label="Center Line")
    ax.axhline(ucl_x, linestyle="--", label="UCL")
    ax.axhline(lcl_x, linestyle="--", label="LCL")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Average Fill Volume (ml)")
    ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("R Control Chart")

    fig, ax = plt.subplots()
    ax.plot(sample_stats["Sample"], sample_stats["range"], marker="o")
    ax.axhline(r_bar, linestyle="--", label="Center Line")
    ax.axhline(ucl_r, linestyle="--", label="UCL")
    ax.axhline(lcl_r, linestyle="--", label="LCL")
    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Range (ml)")
    ax.legend()
    st.pyplot(fig)

st.divider()

left2, right2 = st.columns(2)

with left2:
    st.subheader("Histogram with Specification Limits")

    fig, ax = plt.subplots()
    ax.hist(df["Fill Volume"], bins=15, edgecolor="black")
    ax.axvline(lsl, linestyle="--", label="LSL")
    ax.axvline(usl, linestyle="--", label="USL")
    ax.axvline(target, linestyle="-", label="Target")
    ax.set_xlabel("Fill Volume (ml)")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

with right2:
    st.subheader("Process Capability Summary")

    st.write(f"**Target:** {target:.2f} ml")
    st.write(f"**LSL:** {lsl:.2f} ml")
    st.write(f"**USL:** {usl:.2f} ml")
    st.write(f"**Cp:** {cp:.2f}")
    st.write(f"**Cpk:** {cpk:.2f}")

    if cp < 1:
        st.error("Cp is below 1. The process variation is too high for the specification limits.")
    elif cp < 1.33:
        st.warning("Cp is acceptable but not excellent. Variation should be reduced.")
    else:
        st.success("Cp is strong. The process has good potential capability.")

    if cpk < 1:
        st.error("Cpk is below 1. The process is not reliably meeting specifications.")
    elif cpk < 1.33:
        st.warning("Cpk is acceptable, but the process may need better centering.")
    else:
        st.success("Cpk is strong. The process is capable and well-centered.")

st.divider()

# Out-of-control detection
st.subheader("AI-Style Quality Engineer Insights")

out_x = sample_stats[
    (sample_stats["xbar"] > ucl_x) | (sample_stats["xbar"] < lcl_x)
]

out_r = sample_stats[
    (sample_stats["range"] > ucl_r) | (sample_stats["range"] < lcl_r)
]

defect_count = ((df["Fill Volume"] < lsl) | (df["Fill Volume"] > usl)).sum()
defect_rate = defect_count / len(df) * 100

if len(out_x) == 0 and len(out_r) == 0:
    st.success("The process appears statistically stable based on the X-bar and R charts.")
else:
    st.error("Out-of-control signals were detected.")

    if len(out_x) > 0:
        st.write("Samples outside X-bar control limits:")
        st.write(out_x[["Sample", "xbar"]])

    if len(out_r) > 0:
        st.write("Samples outside R-chart control limits:")
        st.write(out_r[["Sample", "range"]])

st.write(f"**Defect Rate:** {defect_rate:.2f}%")

if overall_mean < target:
    st.warning("The process is leaning toward underfilling, which may cause customer complaints.")
elif overall_mean > target:
    st.warning("The process is leaning toward overfilling, which may increase product waste and cost.")
else:
    st.success("The process average is centered close to the target.")

if defect_rate > 5:
    st.error("High defect rate detected. Recommended action: adjust the filling machine and reduce variability.")
elif defect_rate > 1:
    st.warning("Moderate defect rate detected. Recommended action: monitor the process closely.")
else:
    st.success("Low defect rate. The process is performing well.")

st.divider()

st.subheader("Raw Simulated Data")
st.dataframe(df, use_container_width=True)
