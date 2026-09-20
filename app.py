"""
Credit Card Fraud Detection System - Streamlit Web Application
Codec Technologies Data Science Internship Project
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .fraud-alert {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .legit-alert {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and resources
@st.cache_resource
def load_model_resources():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "fraud_detection_model.pkl")
    samples_path = os.path.join(base_dir, "data", "sample_transactions.json")
    metrics_path = os.path.join(base_dir, "models", "evaluation_metrics.json")

    if not os.path.exists(model_path):
        return None, None, None

    model_bundle = joblib.load(model_path)
    
    sample_data = {}
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            sample_data = json.load(f)

    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

    return model_bundle, sample_data, metrics_data

model_bundle, sample_data, metrics_data = load_model_resources()

# Initialize session state for features if not set
feature_cols = [f"V{i}" for i in range(1, 29)]
if "transaction_input" not in st.session_state:
    st.session_state.transaction_input = {
        "Time": 45000.0,
        "Amount": 100.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }

def set_sample(sample_type, index=0):
    if sample_data and sample_type in sample_data and len(sample_data[sample_type]) > index:
        st.session_state.transaction_input = sample_data[sample_type][index].copy()

# Header
st.markdown('<div class="main-title">💳 Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Machine Learning System for Financial Transaction Security | Codec Technologies Data Science Internship</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-card-back-side.png", width=70)
    st.header("Control Panel")
    
    st.markdown("### 🧪 Quick Demo Samples")
    st.write("Test model inference instantly using real anonymized transactions from the test set:")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🟢 Legitimate Sample", use_container_width=True):
            set_sample("legitimate", index=0)
            st.rerun()
    with col_sb2:
        if st.button("🔴 Fraudulent Sample", use_container_width=True):
            set_sample("fraudulent", index=0)
            st.rerun()

    st.divider()

    st.markdown("### ℹ️ About Features V1–V28")
    st.info(
        "Due to strict financial confidentiality and privacy compliance (GDPR/PCI-DSS), "
        "features **V1 through V28** are anonymized Principal Component Analysis (PCA) transformations. "
        "The only raw attributes provided without transformation are **Time** (seconds elapsed since first transaction in dataset) and **Amount** (transaction amount in USD)."
    )

    if metrics_data and "Random Forest" in metrics_data:
        st.divider()
        st.markdown("### 🏆 Production Model Metrics")
        rf_met = metrics_data["Random Forest"]
        st.write(f"- **Algorithm:** Random Forest + SMOTE")
        st.write(f"- **Accuracy:** {rf_met['accuracy']*100:.2f}%")
        st.write(f"- **Precision:** {rf_met['precision']*100:.2f}%")
        st.write(f"- **Recall:** {rf_met['recall']*100:.2f}%")
        st.write(f"- **ROC-AUC:** {rf_met['roc_auc']:.4f}")

# Main Layout
if model_bundle is None:
    st.error("⚠️ Model file not found! Please run `python src/train_model.py` first to train and serialize the model.")
    st.stop()

# Interactive Form
with st.form("prediction_form"):
    st.subheader("📝 Enter Transaction Details")
    st.caption("Adjust the transaction parameters manually or use the Quick Demo Samples in the sidebar.")

    # Primary Attributes
    col_main1, col_main2 = st.columns(2)
    with col_main1:
        time_val = st.number_input(
            "Transaction Time (Seconds elapsed)",
            min_value=0.0,
            max_value=200000.0,
            value=float(st.session_state.transaction_input.get("Time", 50000.0)),
            step=100.0,
            help="Number of seconds elapsed between this transaction and the first transaction in the dataset."
        )
    with col_main2:
        amount_val = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=float(st.session_state.transaction_input.get("Amount", 99.50)),
            step=5.0,
            help="Monetary value of the transaction in USD."
        )

    st.markdown("#### 🔬 Anonymized PCA Features (V1 – V28)")
    with st.expander("Show / Edit PCA Components (V1 to V28)", expanded=True):
        st.write("Features V1 to V28 capture latent transaction patterns (location, merchant category, velocity, device fingerprints):")
        
        # Display features in 4 columns for compact elegance
        cols = st.columns(4)
        input_v_values = {}
        for idx, feat in enumerate(feature_cols):
            col_idx = idx % 4
            with cols[col_idx]:
                val = float(st.session_state.transaction_input.get(feat, 0.0))
                input_v_values[feat] = st.number_input(
                    f"{feat}",
                    value=val,
                    format="%.4f",
                    key=f"input_{feat}"
                )

    submit_button = st.form_submit_button("🔍 Predict Transaction", use_container_width=True)

# Prediction Logic & Display
if submit_button:
    # 1. Prepare raw dataframe with correct order
    input_dict = {"Time": time_val, "Amount": amount_val, **input_v_values}
    
    # 2. Extract scalers and model from bundle
    rf_model = model_bundle["model"]
    scaler_amount = model_bundle["scaler_amount"]
    scaler_time = model_bundle["scaler_time"]
    feature_order = model_bundle["feature_order"]

    # 3. Apply exact same scaling applied during training (pass DataFrame with column names to avoid warnings)
    scaled_amount = scaler_amount.transform(pd.DataFrame([[amount_val]], columns=['Amount']))[0][0]
    scaled_time = scaler_time.transform(pd.DataFrame([[time_val]], columns=['Time']))[0][0]

    # 4. Construct input vector adhering to model's exact feature order
    processed_row = {}
    for feat in feature_cols:
        processed_row[feat] = input_v_values[feat]
    processed_row["scaled_amount"] = scaled_amount
    processed_row["scaled_time"] = scaled_time

    X_infer = pd.DataFrame([processed_row])[feature_order]

    # 5. Predict class and probability
    prediction = int(rf_model.predict(X_infer)[0])
    probabilities = rf_model.predict_proba(X_infer)[0]
    fraud_prob = probabilities[1]
    legit_prob = probabilities[0]

    st.markdown("---")
    st.subheader("🎯 Assessment Results")

    col_res1, col_res2, col_res3 = st.columns([1.5, 1, 1])

    with col_res1:
        if prediction == 1 or fraud_prob >= 0.50:
            st.markdown(f"""
            <div class="fraud-alert">
                <h3 style="color: #991B1B; margin-top:0;">⚠️ Prediction: Potential Fraud</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <strong>Fraud Probability:</strong> <span style="font-size:1.4rem; color:#DC2626; font-weight:bold;">{fraud_prob * 100:.2f}%</span>
                </p>
                <p style="color: #4B5563; margin-bottom: 0;">
                    <em>"The model has classified this transaction as potentially fraudulent. Further verification recommended."</em>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-alert">
                <h3 style="color: #065F46; margin-top:0;">✅ Prediction: Legitimate Transaction</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <strong>Fraud Probability:</strong> <span style="font-size:1.4rem; color:#059669; font-weight:bold;">{fraud_prob * 100:.2f}%</span>
                </p>
                <p style="color: #4B5563; margin-bottom: 0;">
                    <em>"The transaction exhibits patterns consistent with regular legitimate user behavior."</em>
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_res2:
        st.markdown("**Risk Gauge:**")
        st.progress(float(fraud_prob))
        if fraud_prob < 0.20:
            st.success("Risk Level: LOW")
        elif fraud_prob < 0.60:
            st.warning("Risk Level: MEDIUM")
        else:
            st.error("Risk Level: CRITICAL")

    with col_res3:
        st.markdown("**Recommended Business Action:**")
        if fraud_prob >= 0.70:
            st.error("🛑 Immediate Block & Flag for Analyst Review")
        elif fraud_prob >= 0.40:
            st.warning("📲 Trigger Step-Up Multi-Factor Auth (SMS/OTP)")
        else:
            st.success("⚡ Approve Transaction Instantly")

    # Feature Influence Breakdown
    st.markdown("---")
    st.subheader("📊 Key Diagnostic Features for this Decision")
    
    # Analyze Top 5 most deviating features
    deviations = {feat: abs(input_v_values[feat]) for feat in feature_cols}
    top_deviations = sorted(deviations.items(), key=lambda x: x[1], reverse=True)[:5]
    
    dev_df = pd.DataFrame(top_deviations, columns=["Feature", "Deviation Magnitude (|z-score|)"])
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        st.dataframe(dev_df, use_container_width=True)
    with col_d2:
        st.info(
            "Features like **V17, V14, V12, V10, and V4** have the strongest statistical importance "
            "in detecting fraud. When their absolute values deviate significantly from 0, the model's fraud probability increases sharply."
        )

# Footer
st.markdown("---")
st.markdown(
    "<center style='color: #6B7280; font-size: 0.85rem;'>"
    "Fraud Detection System | Developed for Codec Technologies Data Science Internship | Model: Random Forest with SMOTE Resampling"
    "</center>",
    unsafe_allow_html=True
)
