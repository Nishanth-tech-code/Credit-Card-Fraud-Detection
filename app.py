from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Checker", page_icon="💳")

st.title("💳 Credit Card Transaction Checker")
st.markdown("Enter transaction details below to verify if the charge is legitimate or fraudulent.")

# 1. Load Model and Scaler
BASE_DIR = Path(__file__).parent
model_file = BASE_DIR / "fraud_model.pkl"

if not model_file.exists():
    st.error("Cannot find 'fraud_model.pkl' in the repository root directory.")
    st.stop()

pkg = joblib.load(model_file)
model = pkg["model"]
scaler = pkg["scaler"]
features = pkg.get("features") or pkg.get("feature_names")

# 2. User Input Form
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input(
            "Transaction Amount ($):",
            min_value=0.01,
            max_value=50000.0,
            value=25.00,
            step=5.0,
            help="Enter the dollar value of the transaction."
        )
        
    with col2:
        hour = st.slider(
            "Hour of Transaction (0 to 23):",
            min_value=0,
            max_value=23,
            value=14,
            help="0 is Midnight, 12 is Noon, 18 is 6:00 PM."
        )

    submit = st.form_submit_button("Check Transaction", type="primary", use_container_width=True)

# 3. Process & Predict
if submit:
    # Scale Amount using the fitted RobustScaler
    scaled_amt = scaler.transform([[amount]])[0][0]

    # Initialize all PCA features (V1-V28) to 0.0 (baseline neutral value)
    input_data = {feat: 0.0 for feat in features}
    
    # Assign scaled amount and hour
    input_data['scaled_amount'] = scaled_amt
    if 'Hour' in input_data:
        input_data['Hour'] = hour

    # Build DataFrame in exact feature order
    X_input = pd.DataFrame([input_data])[features]

    # Model Prediction
    prob = float(model.predict_proba(X_input)[0][1])

    st.write("---")
    st.subheader("Transaction Analysis Result")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Raw Amount", f"${amount:.2f}")
        st.metric("Transformed Scaled Amount", f"{scaled_amt:.4f}")
    with col_res2:
        st.metric("Time of Day", f"{hour:02d}:00")
        st.metric("Calculated Fraud Risk", f"{prob * 100:.2f}%")

    # Decision Threshold (0.30)
    if prob >= 0.30:
        st.error("🚨 **STATUS: FRAUDULENT TRANSACTION DETECTED**")
        st.warning(f"The risk score of {prob * 100:.1f}% exceeds the security threshold (30%). Transaction has been blocked.")
    else:
        st.success("✅ **STATUS: TRANSACTION IS LEGITIMATE**")
        st.info(f"The risk score is {prob * 100:.1f}%, which is well within normal operating parameters. Approved for settlement.")
