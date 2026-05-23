import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
from datetime import datetime
import time

# ── Pure NumPy Neural Network (No TensorFlow needed) ─────────────────────────
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_numpy(weights, x):
    # Forward pass through Dense layers
    # Layer order: Dense(64) → Dense(32) → Dense(16) → Dense(1)
    out = x
    layer_idx = 0
    activations = [relu, relu, relu, sigmoid]
    for i, act in enumerate(activations):
        W, b = weights[layer_idx], weights[layer_idx + 1]
        out = act(out @ W + b)
        layer_idx += 2
    return out

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, "model_weights.pkl"), "rb") as f:
        weights_nested = pickle.load(f)
    # Flatten: [[W0,b0],[W1,b1],...] → [W0,b0,W1,b1,...]
    weights = []
    for layer_weights in weights_nested:
        weights.extend(layer_weights)

    with open(os.path.join(base, "scaler.pkl"),       "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(base, "encoders.pkl"),     "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(base, "feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    return weights, scaler, encoders, feature_cols

weights, scaler, encoders, feature_cols = load_artifacts()

# ── RBI EWS Alert Rules ───────────────────────────────────────────────────────
def check_rbi_ews_alerts(data):
    alerts = []

    if data["Transaction_Amount"] >= 1000000:
        alerts.append({
            "level":   "🔴 CRITICAL",
            "rule":    "CTR — RBI/2023-24/73",
            "message": "Cash transaction ≥ ₹10 Lakh — Mandatory CTR filing with FIU-IND",
            "action":  "File Cash Transaction Report (CTR) within 15 days"
        })
    if data["Transaction_Amount"] >= 500000 and data["Unusual_Timing"]:
        alerts.append({
            "level":   "🔴 CRITICAL",
            "rule":    "STR — PMLA 2002 Section 12",
            "message": "High value transaction ≥ ₹5 Lakh at unusual hours",
            "action":  "File Suspicious Transaction Report (STR) within 7 days"
        })
    if data["KYC_Status"] in ["Incomplete", "Pending"]:
        alerts.append({
            "level":   "🟠 HIGH",
            "rule":    "KYC — RBI Master Direction 2016",
            "message": "Transaction by non-KYC compliant customer",
            "action":  "Freeze account and complete KYC within 30 days"
        })
    if data["Round_Amount_Flag"] and data["Transaction_Amount"] >= 100000:
        alerts.append({
            "level":   "🟠 HIGH",
            "rule":    "EWS — Structuring Alert",
            "message": "Round-figure transaction ≥ ₹1 Lakh — possible structuring",
            "action":  "Investigate for transaction structuring"
        })
    if data["International_Transfer"] and data["Is_High_Risk_Country"]:
        alerts.append({
            "level":   "🔴 CRITICAL",
            "rule":    "FEMA 1999 + FATF High-Risk",
            "message": "International transfer to/from FATF high-risk jurisdiction",
            "action":  "Enhanced Due Diligence required. Report to RBI FEMA cell"
        })
    if data["Account_Status"] == "Dormant" and data["Transaction_Amount"] >= 50000:
        alerts.append({
            "level":   "🟠 HIGH",
            "rule":    "EWS — Dormant Account Alert",
            "message": "High value activity on dormant account",
            "action":  "Verify customer identity before processing"
        })
    if data["Country_Risk"] >= 8:
        alerts.append({
            "level":   "🟡 MEDIUM",
            "rule":    "EWS — Geographic Risk",
            "message": f"High country risk score: {data['Country_Risk']}/10",
            "action":  "Apply Enhanced Due Diligence per RBI Risk-Based Approach"
        })
    if data["Transaction_Count"] >= 20:
        alerts.append({
            "level":   "🟡 MEDIUM",
            "rule":    "EWS — Velocity Alert",
            "message": f"High transaction frequency: {data['Transaction_Count']} transactions",
            "action":  "Review for layering/smurfing activity"
        })
    if data["Transaction_Amount"] > (data["Income"] * 0.5):
        alerts.append({
            "level":   "🟠 HIGH",
            "rule":    "EWS — Income Mismatch",
            "message": "Transaction exceeds 50% of annual income",
            "action":  "Request source of funds documentation"
        })
    if data["Account_Status"] == "Closed":
        alerts.append({
            "level":   "🔴 CRITICAL",
            "rule":    "EWS — Closed Account",
            "message": "Transaction attempted on closed account",
            "action":  "Block transaction immediately. Escalate to compliance team"
        })
    return alerts

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="AML Dashboard", page_icon="🏦", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1a1a2e, #16213e);
    padding: 20px; border-radius: 10px;
    color: white; text-align: center; margin-bottom: 20px;
}
.alert-critical {
    background:#fff5f5; border-left:5px solid #e53e3e;
    padding:12px; border-radius:8px; margin:8px 0;
}
.alert-high {
    background:#fffaf0; border-left:5px solid #dd6b20;
    padding:12px; border-radius:8px; margin:8px 0;
}
.alert-medium {
    background:#fffff0; border-left:5px solid #d69e2e;
    padding:12px; border-radius:8px; margin:8px 0;
}
.rbi-badge {
    background:#2d3748; color:white;
    padding:3px 10px; border-radius:20px;
    font-size:12px; font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏦 AML Monitoring Dashboard</h1>
    <p>Real-Time Transaction Surveillance | RBI EWS & PMLA Compliance</p>
    <p style="font-size:12px;opacity:0.7;">
        PMLA 2002 | RBI KYC Master Directions 2016 |
        FEMA 1999 | FIU-IND Guidelines | FATF Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

if "transaction_log" not in st.session_state:
    st.session_state.transaction_log = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 RBI AML System")
    st.markdown("---")
    st.markdown("**Regulatory Framework:**")
    for rule in ["📋 PMLA 2002","📋 RBI KYC Directions 2016",
                 "📋 FEMA 1999","📋 FIU-IND Guidelines",
                 "📋 FATF 40 Recommendations","📋 RBI EWS Circular 2023"]:
        st.markdown(f"- {rule}")
    st.markdown("---")
    st.markdown("**Alert Thresholds:**")
    st.markdown("- CTR: ≥ ₹10 Lakh")
    st.markdown("- STR: Suspicious activity")
    st.markdown("- EDD: High-risk countries")
    st.markdown("---")
    total = len(st.session_state.transaction_log)
    if total > 0:
        flagged = sum(1 for t in st.session_state.transaction_log
                      if t["risk"] in ["HIGH","CRITICAL"])
        st.metric("Total Transactions", total)
        st.metric("Flagged", flagged, delta=f"{flagged/total*100:.0f}% rate")
    if st.button("🗑️ Clear Log", use_container_width=True):
        st.session_state.transaction_log = []
        st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Transaction Screening",
    "📊 Analytics Dashboard",
    "📋 Compliance Log"
])

# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Real-Time Transaction Screening")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Customer Information**")
        customer_type  = st.selectbox("Customer Type",
                             list(encoders["Customer_Type"].classes_))
        account_status = st.selectbox("Account Status",
                             list(encoders["Account_Status"].classes_))
        kyc_status     = st.selectbox("KYC Status",
                             list(encoders["KYC_Compliance"].classes_))
        occupation     = st.selectbox("Occupation",
                             list(encoders["Occupation"].classes_))
        customer_age   = st.number_input("Customer Age",
                             min_value=18, max_value=100, value=35)
        account_tenure = st.number_input("Account Tenure (years)",
                             min_value=0.0, value=2.0, step=0.1)
        income         = st.number_input("Annual Income (₹)",
                             min_value=0.0, value=500000.0, step=10000.0)

    with col2:
        st.markdown("**💳 Transaction Details**")
        transaction_type = st.selectbox("Transaction Type",
                               list(encoders["Transaction_Type"].classes_))
        industry_risk    = st.selectbox("Industry Risk",
                               list(encoders["Industry_Risk"].classes_))
        txn_amount  = st.number_input("Transaction Amount (₹)",
                          min_value=0.0, value=50000.0, step=1000.0)
        txn_count   = st.number_input("Transaction Count",
                          min_value=1, value=10, step=1)
        country_risk = st.slider("Country Risk Score (1-10)",
                          min_value=1, max_value=10, value=3)
        txn_ref = st.text_input("Transaction Reference",
                      value=f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}")

    with col3:
        st.markdown("**🚩 Risk Indicators**")
        st.markdown("<br>", unsafe_allow_html=True)
        is_high_risk_country   = st.checkbox("🌍 High Risk Country (FATF)")
        international_transfer = st.checkbox("✈️ International Transfer")
        unusual_timing         = st.checkbox("⏰ Unusual Transaction Timing")
        round_amount_flag      = st.checkbox("🔢 Round Amount (Structuring Risk)")
        st.markdown(
            f"<br>**⏱️ Time:** `{datetime.now().strftime('%d %b %Y, %H:%M:%S')}`",
            unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚨 Screen Transaction Now",
                 use_container_width=True, type="primary"):
        with st.spinner("Running ML model + RBI EWS checks..."):
            time.sleep(1)

            ct  = encoders["Customer_Type"].transform([customer_type])[0]
            ac  = encoders["Account_Status"].transform([account_status])[0]
            kyc = encoders["KYC_Compliance"].transform([kyc_status])[0]
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
            prob = float(predict_numpy(weights, features_scaled)[0][0])
            ml_risk = "HIGH" if prob > 0.5 else "LOW"

            ews_data = {
                "Transaction_Amount":     txn_amount,
                "Unusual_Timing":         unusual_timing,
                "KYC_Status":             kyc_status,
                "Round_Amount_Flag":      round_amount_flag,
                "International_Transfer": international_transfer,
                "Is_High_Risk_Country":   is_high_risk_country,
                "Account_Status":         account_status,
                "Country_Risk":           country_risk,
                "Transaction_Count":      txn_count,
                "Income":                 income,
            }
            alerts = check_rbi_ews_alerts(ews_data)

            critical_alerts = [a for a in alerts if "CRITICAL" in a["level"]]
            high_alerts     = [a for a in alerts if "HIGH"     in a["level"]]
            overall_risk    = ("CRITICAL" if critical_alerts
                               else "HIGH" if (ml_risk=="HIGH" or high_alerts)
                               else "LOW")

            st.session_state.transaction_log.append({
                "ref":      txn_ref,
                "time":     datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "amount":   txn_amount,
                "type":     transaction_type,
                "risk":     overall_risk,
                "ml_prob":  prob,
                "alerts":   len(alerts),
                "customer": customer_type,
            })

        st.markdown("---")
        st.subheader("📊 Screening Results")
        m1,m2,m3,m4 = st.columns(4)
        with m1:
            color = ("🔴" if overall_risk=="CRITICAL"
                     else "🟠" if overall_risk=="HIGH" else "🟢")
            st.metric("Overall Risk", f"{color} {overall_risk}")
        with m2: st.metric("ML Fraud Probability", f"{prob:.2%}")
        with m3: st.metric("RBI EWS Alerts", f"{len(alerts)} triggered")
        with m4: st.metric("Transaction Ref", txn_ref)

        st.progress(prob)

        if ml_risk == "HIGH":
            st.error(f"🤖 ML Model: HIGH RISK — {prob:.2%} fraud probability")
        else:
            st.success(f"🤖 ML Model: LOW RISK — {prob:.2%} fraud probability")

        if alerts:
            st.markdown("---")
            st.subheader(f"🚨 RBI EWS Alerts ({len(alerts)} triggered)")
            for alert in alerts:
                css = ("alert-critical" if "CRITICAL" in alert["level"]
                       else "alert-high" if "HIGH" in alert["level"]
                       else "alert-medium")
                st.markdown(f"""
                <div class="{css}">
                    <b>{alert["level"]}</b> &nbsp;
                    <span class="rbi-badge">{alert["rule"]}</span><br>
                    <b>⚠️ Alert:</b> {alert["message"]}<br>
                    <b>✅ Action:</b> {alert["action"]}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ No RBI EWS alerts triggered")

        st.markdown("---")
        st.subheader("📋 Recommended Compliance Actions")
        if overall_risk == "CRITICAL":
            st.markdown("""
| Priority | Action | Deadline |
|----------|--------|----------|
| 🔴 Immediate | Block/hold transaction | Now |
| 🔴 Immediate | Escalate to Compliance Officer | Within 1 hour |
| 🔴 Urgent | File STR with FIU-IND | Within 7 days |
| 🔴 Urgent | File CTR if cash ≥ ₹10L | Within 15 days |
| 🟠 High | Enhanced Due Diligence (EDD) | Within 24 hours |
""")
        elif overall_risk == "HIGH":
            st.markdown("""
| Priority | Action | Deadline |
|----------|--------|----------|
| 🟠 High | Flag for manual review | Within 4 hours |
| 🟠 High | Customer due diligence | Within 24 hours |
| 🟡 Medium | Request source of funds | Within 48 hours |
""")
        else:
            st.markdown("""
| Priority | Action | Deadline |
|----------|--------|----------|
| 🟢 Routine | Standard monitoring | Ongoing |
| 🟢 Routine | Periodic KYC review | Annually |
""")

# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📊 Transaction Analytics")
    if not st.session_state.transaction_log:
        st.info("📭 No transactions yet. Use Tab 1 to screen transactions.")
    else:
        log_df = pd.DataFrame(st.session_state.transaction_log)
        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("Total Screened", len(log_df))
        with k2: st.metric("🔴 Critical",
                           len(log_df[log_df["risk"]=="CRITICAL"]))
        with k3: st.metric("🟠 High Risk",
                           len(log_df[log_df["risk"]=="HIGH"]))
        with k4: st.metric("🟢 Low Risk",
                           len(log_df[log_df["risk"]=="LOW"]))
        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Risk Distribution**")
            st.bar_chart(log_df["risk"].value_counts())
        with c2:
            st.markdown("**Avg Transaction Amount by Risk**")
            st.bar_chart(log_df.groupby("risk")["amount"].mean())
        st.markdown("**📈 ML Fraud Probability Trend**")
        st.line_chart(log_df["ml_prob"])

# ══════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📋 Compliance Audit Log")
    if not st.session_state.transaction_log:
        st.info("📭 No transactions in log yet.")
    else:
        log_df = pd.DataFrame(st.session_state.transaction_log)

        def color_risk(val):
            if val == "CRITICAL": return "background-color: #fed7d7"
            elif val == "HIGH":   return "background-color: #feebc8"
            elif val == "LOW":    return "background-color: #c6f6d5"
            return ""

        st.dataframe(
            log_df.style.map(color_risk, subset=["risk"]),
            use_container_width=True
        )
        st.markdown("---")
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Compliance Report (CSV)",
            data=csv,
            file_name=f"AML_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown("---")
        st.subheader("📑 Regulatory Filing Summary")
        ctr  = log_df[log_df["amount"] >= 1000000]
        str_ = log_df[log_df["risk"].isin(["CRITICAL","HIGH"])]
        f1,f2 = st.columns(2)
        with f1: st.metric("CTR Filings Required", len(ctr))
        with f2: st.metric("STR Filings Required", len(str_))
        if len(ctr)  > 0:
            st.warning(f"⚠️ {len(ctr)} transaction(s) need CTR filing within 15 days")
        if len(str_) > 0:
            st.error(f"🚨 {len(str_)} transaction(s) need STR filing within 7 days")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:gray;font-size:12px;">
    🏦 AML Monitoring System | RBI EWS | PMLA 2002 | FEMA 1999 | FIU-IND<br>
    ⚠️ This system aids compliance officers and does not replace regulatory judgment.
</div>
""", unsafe_allow_html=True)
