
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os

@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model       = tf.keras.models.load_model(os.path.join(base, "aml_model.h5"))
    with open(os.path.join(base, "scaler.pkl"),       "rb") as f: scaler       = pickle.load(f)
    with open(os.path.join(base, "encoders.pkl"),     "rb") as f: encoders     = pickle.load(f)
    with open(os.path.join(base, "feature_cols.pkl"), "rb") as f: feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols

model, scaler, encoders, feature_cols = load_artifacts()

st.set_page_config(page_title="AML Detection", page_icon="💰", layout="wide")
st.title("💰 Money Laundering Detection System")
st.markdown("Fill in the details below to assess transaction risk.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Info")
    customer_type  = st.selectbox("Customer Type",    list(encoders["Customer_Type"].classes_))
    account_status = st.selectbox("Account Status",   list(encoders["Account_Status"].classes_))
    kyc_compliance = st.selectbox("KYC Compliance",   list(encoders["KYC_Compliance"].classes_))
    occupation     = st.selectbox("Occupation",       list(encoders["Occupation"].classes_))
    customer_age   = st.number_input("Customer Age",  min_value=18, max_value=100, value=35)
    account_tenure = st.number_input("Account Tenure (years)", min_value=0.0, value=2.0, step=0.1)
    income         = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)

with col2:
    st.subheader("💳 Transaction Info")
    transaction_type = st.selectbox("Transaction Type", list(encoders["Transaction_Type"].classes_))
    industry_risk    = st.selectbox("Industry Risk",    list(encoders["Industry_Risk"].classes_))
    txn_amount       = st.number_input("Transaction Amount ($)", min_value=0.0, value=5000.0, step=100.0)
    txn_count        = st.number_input("Transaction Count", min_value=1, value=10, step=1)
    country_risk     = st.slider("Country Risk Score", min_value=1, max_value=10, value=3)

with col3:
    st.subheader("🚩 Risk Flags")
    st.markdown("<br><br>", unsafe_allow_html=True)
    is_high_risk_country   = st.checkbox("🌍 High Risk Country")
    international_transfer = st.checkbox("✈️  International Transfer")
    unusual_timing         = st.checkbox("⏰ Unusual Timing")
    round_amount_flag      = st.checkbox("🔢 Round Amount Flag")

st.markdown("---")

if st.button("🔍 Run Prediction", use_container_width=True):
    try:
        ct  = encoders["Customer_Type"].transform([customer_type])[0]
        ac  = encoders["Account_Status"].transform([account_status])[0]
        kyc = encoders["KYC_Compliance"].transform([kyc_compliance])[0]
        ir  = encoders["Industry_Risk"].transform([industry_risk])[0]
        tt  = encoders["Transaction_Type"].transform([transaction_type])[0]
        occ = encoders["Occupation"].transform([occupation])[0]

        raw = {
            "Customer_Type":           ct,
            "Account_Status":          ac,
            "KYC_Compliance":          kyc,
            "Industry_Risk":           ir,
            "Transaction_Amount":      txn_amount,
            "Transaction_Count":       txn_count,
            "Country_Risk":            country_risk,
            "Is_High_Risk_Country":    int(is_high_risk_country),
            "International_Transfer":  int(international_transfer),
            "Unusual_Timing":          int(unusual_timing),
            "Round_Amount_Flag":       int(round_amount_flag),
            "Transaction_Type":        tt,
            "Customer_Age":            customer_age,
            "Customer_Account_Tenure": account_tenure,
            "Occupation":              occ,
            "Income":                  income,
        }

        features        = np.array([[raw[col] for col in feature_cols]])
        features_scaled = scaler.transform(features)
        prob            = float(model.predict(features_scaled)[0][0])

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            if prob > 0.5:
                st.error("🚨 HIGH RISK — Likely Money Laundering")
            else:
                st.success("✅ LOW RISK — Transaction Appears Legitimate")
        with r2:
            st.metric("Fraud Probability", f"{prob:.2%}")
        with r3:
            st.metric("Confidence", f"{max(prob, 1-prob):.2%}")

        st.progress(prob)

        with st.expander("📋 View Full Input Summary"):
            import pandas as pd
            summary = pd.DataFrame({
                "Feature":    feature_cols,
                "Raw Value":  [raw[col] for col in feature_cols]
            })
            st.dataframe(summary, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
