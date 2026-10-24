# Author identifier: 72427_abdulrafay
import asyncio
import sys

# Fix for Windows asyncio socket connection drops on Streamlit
if sys.platform == "win32":
  asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import joblib
import numpy as np
import streamlit as st

# (Rest of your app.py code continues below...)

# Load artifacts
model = joblib.load("maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")
optimal_threshold = joblib.load("optimal_threshold.pkl")

st.title("Industrial Equipment Failure Predictor")
st.write(
    "Input live operational sensor metrics below to evaluate machine failure"
    " risk."
)

air_temp = st.slider(
    "Air Temperature [K]", 295.0, 305.0, 298.0, step=0.1
)
process_temp = st.slider(
    "Process Temperature [K]", 305.0, 315.0, 308.0, step=0.1
)
rot_speed = st.slider("Rotational Speed [rpm]", 1200, 2800, 1500, step=10)
torque = st.slider("Torque [Nm]", 10.0, 80.0, 40.0, step=0.5)
tool_wear = st.slider("Tool Wear [min]", 0, 250, 100, step=1)

if st.button("Run Diagnostic Analysis"):
  input_data = np.array(
      [[air_temp, process_temp, rot_speed, torque, tool_wear]]
  )
  input_scaled = scaler.transform(input_data)
  failure_probability = model.predict_proba(input_scaled)[:, 1][0]
  is_failure = failure_probability >= optimal_threshold

  st.markdown("---")
  st.subheader("Diagnostic Results")
  st.metric(
      label="Predicted Failure Probability", value=f"{failure_probability:.4f}"
  )
  st.text(f"Active Decision Threshold: {optimal_threshold:.4f}")

  if is_failure:
    st.error(
        f"CRITICAL ALERT: High risk of failure! Probability"
        f" ({failure_probability:.4f}) meets or exceeds the threshold."
    )
  else:
    st.success(
        "NORMAL: Equipment operating safely below the critical failure"
        " threshold."
    )