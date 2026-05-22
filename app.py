
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import pandas as pd
from datetime import datetime
import time

# ── Load artifacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    model        = tf.keras.models.load_model(os.path.join(base, "aml_model.h5"))
    with open(os.path.join(base, "scaler.pkl"),       "rb") as f: scaler       = pickle.load(f)
    with open(os.path.join(base, "encoders.pkl"),     "rb") as f: encoders     = pickle.load(f)
    with open(os.path.join(base, "feature_cols.pkl"), "rb") as f: feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols

model, scaler, encoders, feature_cols = load_artifacts()

# ── RBI EWS Alert Rules ──────────────────────────────────────────────────────
def check_rbi_ews_alerts(data):
    alerts = []

    # RBI Circular RBI/2023-24/73 — Cash Transaction Report (CTR)
    if data["Transaction_Amount"] >= 1000000:
        alerts.append({
            "level":     "🔴 CRITICAL",
            "rule":      "CTR — RBI/2023-24/73",
            "message":   "Cash transaction ≥ ₹10 Lakh — Mandatory CTR filing with FIU-IND within 15 days",
            "action":    "File Cash Transaction Report (CTR) immediately"
        })

    # RBI PMLA 2002 — Suspicious Transaction Report (STR)
    if data["Transaction_Amount"] >= 500000 and data["Unusual_Timing"]:
        alerts.append({
            "level":     "🔴 CRITICAL",
            "rule":      "STR — PMLA 2002 Section 12",
            "message":   "High value transaction (≥ ₹5 Lakh) at unusual hours detected",
            "action":    "File Suspicious Transaction Report (STR) with FIU-IND within 7 days"
        })

    # RBI KYC Master Directions 2016 — Incomplete KYC
    if data["KYC_Status"] in ["Incomplete", "Pending"]:
        alerts.append({
            "level":     "🟠 HIGH",
            "rule":      "KYC — RBI Master Direction 2016",
            "message":   "Transaction by non-KYC compliant customer",
            "action":    "Freeze account and complete KYC within 30 days per RBI KYC norms"
        })

    # RBI EWS Circular — Round Amount Flag
    if data["Round_Amount_Flag"] and data["Transaction_Amount"] >= 100000:
        alerts.append({
            "level":     "🟠 HIGH",
            "rule":      "EWS — Structuring Alert",
            "message":   "Round-figure transaction ≥ ₹1 Lakh — possible structuring/smurfing",
            "action":    "Investigate for transaction structuring to avoid reporting thresholds"
        })

    # RBI FEMA 1999 — International Transfer
    if data["International_Transfer"] and data["Is_High_Risk_Country"]:
        alerts.append({
            "level":     "🔴 CRITICAL",
            "rule":      "FEMA 1999 + FATF High-Risk",
            "message":   "International transfer to/from FATF high-risk jurisdiction",
            "action":    "Enhanced Due Diligence (EDD) required. Report to RBI FEMA cell"
        })

    # RBI EWS — Dormant Account Activity
    if data["Account_Status"] == "Dormant" and data["Transaction_Amount"] >= 50000:
        alerts.append({
            "level":     "🟠 HIGH",
            "rule":      "EWS — Dormant Account Alert",
            "message":   "High value activity on dormant account detected",
            "action":    "Verify customer identity before processing. Flag for review"
        })

    # RBI EWS — High Country Risk
    if data["Country_Risk"] >= 8:
        alerts.append({
            "level":     "🟡 MEDIUM",
            "rule":      "EWS — Geographic Risk",
            "message":   f"Transaction involves country with risk score {data['Country_Risk']}/10",
            "action":    "Apply Enhanced Due Diligence (EDD) per RBI Risk-Based Approach"
        })

    # RBI EWS — High Transaction Frequency
    if data["Transaction_Count"] >= 20:
        alerts.append({
            "level":     "🟡 MEDIUM",
            "rule":      "EWS — Velocity Alert",
            "message":   f"High transaction frequency: {data['Transaction_Count']} transactions",
            "action":    "Review transaction pattern for layering/smurfing activity"
        })

    # RBI EWS — Income vs Transaction Mismatch
    if data["Transaction_Amount"] > (data["Income"] * 0.5):
        alerts.append({
            "level":     "🟠 HIGH",
            "rule":      "EWS — Income Mismatch",
            "message":   "Transaction amount exceeds 50% of annual income — unusual pattern",
            "action":    "Request source of funds documentation from customer"
        })

    # RBI EWS — Closed Account Activity
    if data["Account_Status"] == "Closed":
        alerts.append({
            "level":     "🔴 CRITICAL",
            "rule":      "EWS — Closed Account",
            "message":   "Transaction attempted on closed account",
            "action":    "Block transaction immediately. Escalate to compliance team"
        })

    return alerts

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "AML Detection Dashboard",
    page_icon  = "🏦",
    layout     = "wide"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .alert-critical {
        background: #fff5f5;
        border-left: 5px solid #e53e3e;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .alert-high {
        background: #fffaf0;
        border-left: 5px solid #dd6b20;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .alert-medium {
        background: #fffff0;
        border-left: 5px solid #d69e2e;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .rbi-badge {
        background: #2d3748;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 AML Monitoring Dashboard</h1>
    <p>Real-Time Transaction Surveillance | RBI EWS & PMLA Compliance</p>
    <p style="font-size:12px; opacity:0.7;">
        Compliant with: PMLA 2002 | RBI KYC Master Directions 2016 |
        FEMA 1999 | FIU-IND Guidelines | FATF Recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session State for Transaction Log ────────────────────────────────────────
if "transaction_log" not in st.session_state:
    st.session_state.transaction_log = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Reserve_Bank_of_India_seal.svg/200px-Reserve_Bank_of_India_seal.svg.png", width=80)
    st.markdown("### 🏦 RBI AML System")
    st.markdown("---")
    st.markdown("**Regulatory Framework:**")
    st.markdown("- 📋 PMLA 2002")
    st.markdown("- 📋 RBI KYC Directions 2016")
    st.markdown("- 📋 FEMA 1999")
    st.markdown("- 📋 FIU-IND Guidelines")
    st.markdown("- 📋 FATF 40 Recommendations")
    st.markdown("- 📋 RBI EWS Circular 2023")
    st.markdown("---")
    st.markdown("**Alert Thresholds:**")
    st.markdown("- CTR: ≥ ₹10 Lakh")
    st.markdown("- STR: Suspicious activity")
    st.markdown("- EDD: High-risk countries")
    st.markdown("---")
    total = len(st.session_state.transaction_log)
    if total > 0:
        flagged = sum(1 for t in st.session_state.transaction_log if t["risk"] == "HIGH")
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

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Transaction Screening
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Real-Time Transaction Screening")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Customer Information**")
        customer_type  = st.selectbox("Customer Type",    list(encoders["Customer_Type"].classes_))
        account_status = st.selectbox("Account Status",   list(encoders["Account_Status"].classes_))
        kyc_status     = st.selectbox("KYC Status",       list(encoders["KYC_Compliance"].classes_))
        occupation     = st.selectbox("Occupation",       list(encoders["Occupation"].classes_))
        customer_age   = st.number_input("Customer Age",  min_value=18, max_value=100, value=35)
        account_tenure = st.number_input("Account Tenure (years)", min_value=0.0, value=2.0, step=0.1)
        income         = st.number_input("Annual Income (₹)", min_value=0.0, value=500000.0, step=10000.0)

    with col2:
        st.markdown("**💳 Transaction Details**")
        transaction_type = st.selectbox("Transaction Type", list(encoders["Transaction_Type"].classes_))
        industry_risk    = st.selectbox("Industry Risk",    list(encoders["Industry_Risk"].classes_))
        txn_amount       = st.number_input("Transaction Amount (₹)", min_value=0.0, value=50000.0, step=1000.0)
        txn_count        = st.number_input("Transaction Count", min_value=1, value=10, step=1)
        country_risk     = st.slider("Country Risk Score (1-10)", min_value=1, max_value=10, value=3)
        txn_ref          = st.text_input("Transaction Reference", value=f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}")

    with col3:
        st.markdown("**🚩 Risk Indicators**")
        st.markdown("<br>", unsafe_allow_html=True)
        is_high_risk_country   = st.checkbox("🌍 High Risk Country (FATF)")
        international_transfer = st.checkbox("✈️ International Transfer")
        unusual_timing         = st.checkbox("⏰ Unusual Transaction Timing")
        round_amount_flag      = st.checkbox("🔢 Round Amount (Structuring Risk)")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**⏱️ Screening Time**")
        st.markdown(f"`{datetime.now().strftime('%d %b %Y, %H:%M:%S')}`")

    st.markdown("---")

    if st.button("🚨 Screen Transaction Now", use_container_width=True, type="primary"):

        with st.spinner("Running ML model + RBI EWS checks..."):
            time.sleep(1)

            # ── ML Prediction ────────────────────────────────────────────
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
            prob            = float(model.predict(features_scaled)[0][0])
            ml_risk         = "HIGH" if prob > 0.5 else "LOW"

            # ── RBI EWS Alerts ───────────────────────────────────────────
            ews_data = {
                "Transaction_Amount":   txn_amount,
                "Unusual_Timing":       unusual_timing,
                "KYC_Status":           kyc_status,
                "Round_Amount_Flag":    round_amount_flag,
                "International_Transfer": international_transfer,
                "Is_High_Risk_Country": is_high_risk_country,
                "Account_Status":       account_status,
                "Country_Risk":         country_risk,
                "Transaction_Count":    txn_count,
                "Income":               income,
            }
            alerts = check_rbi_ews_alerts(ews_data)

            # ── Overall Risk ─────────────────────────────────────────────
            critical_alerts = [a for a in alerts if "CRITICAL" in a["level"]]
            high_alerts     = [a for a in alerts if "HIGH"     in a["level"]]
            overall_risk    = "CRITICAL" if critical_alerts else ("HIGH" if (ml_risk == "HIGH" or high_alerts) else "LOW")

            # ── Log Transaction ──────────────────────────────────────────
            st.session_state.transaction_log.append({
                "ref":        txn_ref,
                "time":       datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "amount":     txn_amount,
                "type":       transaction_type,
                "risk":       overall_risk,
                "ml_prob":    prob,
                "alerts":     len(alerts),
                "customer":   customer_type,
            })

        # ── Display Results ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Screening Results")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            color = "🔴" if overall_risk == "CRITICAL" else ("🟠" if overall_risk == "HIGH" else "🟢")
            st.metric("Overall Risk", f"{color} {overall_risk}")
        with m2:
            st.metric("ML Fraud Probability", f"{prob:.2%}")
        with m3:
            st.metric("RBI EWS Alerts", f"{len(alerts)} triggered")
        with m4:
            st.metric("Transaction Ref", txn_ref)

        st.progress(prob)

        # ── ML Result ────────────────────────────────────────────────────
        if ml_risk == "HIGH":
            st.error(f"🤖 ML Model: HIGH RISK — Fraud probability {prob:.2%}")
        else:
            st.success(f"🤖 ML Model: LOW RISK — Fraud probability {prob:.2%}")

        # ── RBI EWS Alerts ───────────────────────────────────────────────
        if alerts:
            st.markdown("---")
            st.subheader(f"🚨 RBI EWS Alerts ({len(alerts)} triggered)")

            for alert in alerts:
                level = alert["level"]
                if "CRITICAL" in level:
                    css = "alert-critical"
                elif "HIGH" in level:
                    css = "alert-high"
                else:
                    css = "alert-medium"

                st.markdown(f"""
                <div class="{css}">
                    <b>{alert["level"]}</b> &nbsp;
                    <span class="rbi-badge">{alert["rule"]}</span><br>
                    <b>⚠️ Alert:</b> {alert["message"]}<br>
                    <b>✅ Action:</b> {alert["action"]}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No RBI EWS alerts triggered for this transaction")

        # ── Recommended Actions ───────────────────────────────────────────
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
            | 🟠 High | Document source of funds | Within 48 hours |
            """)
        elif overall_risk == "HIGH":
            st.markdown("""
            | Priority | Action | Deadline |
            |----------|--------|----------|
            | 🟠 High | Flag for manual review | Within 4 hours |
            | 🟠 High | Customer due diligence check | Within 24 hours |
            | 🟡 Medium | Request source of funds | Within 48 hours |
            | 🟡 Medium | Update risk profile | Within 7 days |
            """)
        else:
            st.markdown("""
            | Priority | Action | Deadline |
            |----------|--------|----------|
            | 🟢 Routine | Standard transaction monitoring | Ongoing |
            | 🟢 Routine | Periodic KYC review | Annually |
            """)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Transaction Analytics")

    if len(st.session_state.transaction_log) == 0:
        st.info("📭 No transactions screened yet. Use Tab 1 to screen transactions.")
    else:
        log_df = pd.DataFrame(st.session_state.transaction_log)

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        total    = len(log_df)
        critical = len(log_df[log_df["risk"] == "CRITICAL"])
        high     = len(log_df[log_df["risk"] == "HIGH"])
        low      = len(log_df[log_df["risk"] == "LOW"])

        with k1: st.metric("Total Screened",    total)
        with k2: st.metric("🔴 Critical",       critical)
        with k3: st.metric("🟠 High Risk",       high)
        with k4: st.metric("🟢 Low Risk",        low)

        st.markdown("---")

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Risk Distribution**")
            risk_counts = log_df["risk"].value_counts()
            st.bar_chart(risk_counts)

        with c2:
            st.markdown("**Transaction Amounts by Risk Level**")
            amt_by_risk = log_df.groupby("risk")["amount"].mean()
            st.bar_chart(amt_by_risk)

        st.markdown("---")
        st.markdown("**📈 ML Fraud Probability Trend**")
        st.line_chart(log_df["ml_prob"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Compliance Log
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Compliance Audit Log")

    if len(st.session_state.transaction_log) == 0:
        st.info("📭 No transactions in log yet.")
    else:
        log_df = pd.DataFrame(st.session_state.transaction_log)

        # Color code risk
        def color_risk(val):
            if val == "CRITICAL": return "background-color: #fed7d7"
            elif val == "HIGH":   return "background-color: #feebc8"
            elif val == "LOW":    return "background-color: #c6f6d5"
            return ""

        styled = log_df.style.applymap(color_risk, subset=["risk"])
        st.dataframe(styled, use_container_width=True)

        st.markdown("---")
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Compliance Report (CSV)",
            data      = csv,
            file_name = f"AML_Compliance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime      = "text/csv",
            use_container_width = True
        )

        # STR/CTR Filing Summary
        st.markdown("---")
        st.subheader("📑 Regulatory Filing Summary")
        ctr_required = log_df[log_df["amount"] >= 1000000]
        str_required = log_df[log_df["risk"].isin(["CRITICAL", "HIGH"])]

        f1, f2 = st.columns(2)
        with f1:
            st.metric("CTR Filings Required", len(ctr_required), help="Transactions ≥ ₹10 Lakh")
        with f2:
            st.metric("STR Filings Required", len(str_required), help="High/Critical risk transactions")

        if len(ctr_required) > 0:
            st.warning(f"⚠️ {len(ctr_required)} transaction(s) require CTR filing with FIU-IND within 15 days")
        if len(str_required) > 0:
            st.error(f"🚨 {len(str_required)} transaction(s) require STR filing with FIU-IND within 7 days")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray; font-size:12px;">
    🏦 AML Monitoring System | Compliant with RBI EWS Circular | PMLA 2002 | FEMA 1999 | FIU-IND Guidelines<br>
    ⚠️ This system is an aid to compliance officers and does not replace regulatory judgment.
</div>
""", unsafe_allow_html=True)
