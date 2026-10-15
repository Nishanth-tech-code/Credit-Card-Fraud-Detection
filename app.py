
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Checker", page_icon="💳")

st.title("💳 Credit Card Transaction Checker")
st.markdown(
    "Enter transaction details to analyze risk using the trained **XGBoost**"
    " model."
)

# 1. Load Model, Scaler, and Demo Data
BASE_DIR = Path(__file__).parent
model_file = BASE_DIR / "fraud_model.pkl"
demo_file = BASE_DIR / "demo_data.csv"

if not model_file.exists() or not demo_file.exists():
  st.error("Missing 'fraud_model.pkl' or 'demo_data.csv' in the app folder.")
  st.stop()

pkg = joblib.load(model_file)
model = pkg["model"]
scaler = pkg["scaler"]
features = pkg.get("features") or pkg.get("feature_names")
demo_df = pd.read_csv(demo_file)

# 2. Form Inputs
with st.form("transaction_form"):
  col1, col2 = st.columns(2)

  with col1:
    amount = st.number_input(
        "Transaction Amount ($):",
        min_value=0.01,
        max_value=50000.0,
        value=25.00,
        step=5.0,
    )

  with col2:
    hour = st.slider(
        "Hour of Transaction (0 to 23):", min_value=0, max_value=23, value=14
    )

  profile_type = st.radio(
      "Card & Terminal Signal Profile:",
      ["Standard Verified Card", "Compromised / Suspicious Card Signature"],
      help=(
          "Simulates PCA behavioral features (V1-V28) from verified normal vs."
          " compromised transactions."
      ),
  )

  submit = st.form_submit_button(
      "Analyze Transaction", type="primary", use_container_width=True
  )

# 3. Prediction
if submit:
  scaled_amt = scaler.transform([[amount]])[0][0]

  # Find available label column dynamically
  label_col = None
  for candidate in ["Actual_Label", "Label", "Class", "class"]:
    if candidate in demo_df.columns:
      label_col = candidate
      break

  # Select representative normal or fraud row
  if profile_type == "Compromised / Suspicious Card Signature":
    if label_col:
      fraud_rows = demo_df[
          demo_df[label_col].astype(str).str.contains("1|fraud", case=False)
      ]
      chosen_row = (
          fraud_rows.iloc[0] if len(fraud_rows) > 0 else demo_df.iloc[-1]
      )
    else:
      chosen_row = demo_df.iloc[-1]  # Fallback: last row
  else:
    if label_col:
      normal_rows = demo_df[
          demo_df[label_col]
          .astype(str)
          .str.contains("0|legit|normal", case=False)
      ]
      chosen_row = (
          normal_rows.iloc[0] if len(normal_rows) > 0 else demo_df.iloc[0]
      )
    else:
      chosen_row = demo_df.iloc[0]  # Fallback: first row

  # Prepare feature vector
  input_data = {feat: chosen_row[feat] for feat in features}
  input_data["scaled_amount"] = scaled_amt
  if "Hour" in input_data:
    input_data["Hour"] = hour

  X_input = pd.DataFrame([input_data])[features]
  prob = float(model.predict_proba(X_input)[0][1])

  st.write("---")
  st.subheader("Verdict")

  col_a, col_b = st.columns(2)
  with col_a:
    st.metric("Scaled Amount", f"{scaled_amt:.4f}")
  with col_b:
    st.metric("Fraud Probability", f"{prob * 100:.2f}%")

  if prob >= 0.30:
    st.error(f"🚨 **FLAGGED AS FRAUD** (Risk Score: {prob * 100:.1f}%)")
    st.warning("Action: Block transaction and request OTP verification.")
  else:
    st.success(f"✅ **TRANSACTION APPROVED** (Risk Score: {prob * 100:.1f}%)")
    st.info("Action: Legitimate pattern verified. Proceed to settlement.")
