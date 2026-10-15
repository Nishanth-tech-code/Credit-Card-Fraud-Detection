import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Credit Card Fraud Checker", page_icon="💳")

st.title("💳 Credit Card Transaction Checker")
st.markdown("Enter transaction details to analyze risk using the trained **XGBoost** model.")

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
            step=5.0
        )
        
    with col2:
        hour = st.slider(
            "Hour of Transaction (0 to 23):",
            min_value=0,
            max_value=23,
            value=14
        )

    # Behavior profile selector to provide realistic latent V-features
    profile_type = st.radio(
        "Card & Terminal Signal Profile:",
        ["Standard Verified Card", "Compromised / Suspicious Card Signature"],
        help="In production, payment networks derive PCA features (V1-V28) from IP address, device fingerprint, and velocity."
    )

    submit = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)

# 3. Prediction
if submit:
    # Scale Amount using fitted RobustScaler
    scaled_amt = scaler.transform([[amount]])[0][0]

    # Select representative PCA features from actual data
    if profile_type == "Compromised / Suspicious Card Signature":
        # Extract a real fraud transaction's PCA vector from demo_data
        fraud_sample = demo_df[demo_df['Actual_Label'].str.contains('Fraud', case=False, na=False)].iloc[0]
        input_data = {feat: fraud_sample[feat] for feat in features}
    else:
        # Extract a real normal transaction's PCA vector from demo_data
        normal_sample = demo_df[demo_df['Actual_Label'].str.contains('Legitimate|Normal', case=False, na=False)].iloc[0]
        input_data = {feat: normal_sample[feat] for feat in features}

    # Override user-specified values
    input_data['scaled_amount'] = scaled_amt
    if 'Hour' in input_data:
        input_data['Hour'] = hour

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
