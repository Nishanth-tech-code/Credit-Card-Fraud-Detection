from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="🛡️")

st.title("🛡️ Credit Card Fraud Detector")
st.markdown("Real-time transaction risk scoring using **XGBoost**.")

# 1. Load files using absolute path
BASE_DIR = Path(__file__).parent
model_file = BASE_DIR / "fraud_model.pkl"
data_file = BASE_DIR / "demo_data.csv"

if not model_file.exists() or not data_file.exists():
  st.error("Missing 'fraud_model.pkl' or 'demo_data.csv' in repository root.")
  st.stop()

pkg = joblib.load(model_file)
model = pkg["model"]
features = pkg.get("features") or pkg.get("feature_names")
df = pd.read_csv(data_file)

# 2. UI - Pick a transaction
st.subheader("Select a Test Transaction")
label_col = "Actual_Label" if "Actual_Label" in df.columns else "Label"

idx = st.selectbox(
    "Choose Record:",
    range(len(df)),
    format_func=lambda i: f"Transaction #{i+1} — Ground Truth: {df.iloc[i].get(label_col, 'N/A')}",
)

row = df.iloc[idx]

col1, col2 = st.columns(2)
with col1:
  st.metric("Scaled Amount", f"{row.get('scaled_amount', 0):.2f}")
with col2:
  st.metric("Recorded Hour", f"{int(row.get('Hour', 0))}:00")

# 3. Predict Button
if st.button("Evaluate Fraud Risk", type="primary", use_container_width=True):
  X_input = row[features].to_frame().T
  prob = float(model.predict_proba(X_input)[0][1])

  st.write("---")
  st.progress(prob)

  if prob >= 0.30:
    st.error(f"🚨 **FLAGGED AS FRAUD** (Risk Score: {prob*100:.2f}%)")
    st.caption("Action: Block transaction and request OTP authentication.")
  else:
    st.success(f"✅ **TRANSACTION APPROVED** (Risk Score: {prob*100:.2f}%)")
    st.caption("Action: Normal transaction density verified. Proceed.")
