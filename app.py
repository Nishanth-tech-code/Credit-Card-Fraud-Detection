import joblib
import pandas as pd
import streamlit as st

st.title("🛡️ Credit Card Fraud Detector")

# Load saved package and demo records
pkg = joblib.load("fraud_model_package.pkl")
model, features = pkg["model"], pkg["feature_names"]
df = pd.read_csv("demo_transactions.csv")

# Pick a test transaction
idx = st.selectbox(
    "Select Transaction:",
    range(len(df)),
    format_func=lambda i: f"Row {i+1} — Ground Truth: {df.iloc[i].get('Actual_Label', 'N/A')}",
)
row = df.iloc[idx]

st.write(
    f"**Amount:** {row.get('scaled_amount', 0):.2f} | **Hour:** {int(row.get('Hour', 0))}:00"
)

# Predict
if st.button("Check Fraud Risk", type="primary"):
    X_input = row[features].to_frame().T
    prob = model.predict_proba(X_input)[0][1]

    if prob >= 0.30:
        st.error(f"🚨 Flagged as FRAUD (Risk: {prob*100:.1f}%)")
    else:
        st.success(f"✅ Transaction Approved (Risk: {prob*100:.1f}%)")
