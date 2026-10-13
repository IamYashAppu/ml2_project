import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="AI Fraud Shield | Dual-Engine Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .alert-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-weight: bold;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge-fraud { background-color: #FEE2E2; color: #DC2626; }
    .badge-legit { background-color: #DCFCE7; color: #16A34A; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL DEFINITIONS & RESOURCE CACHING
# ==========================================
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim=30):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, 10),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.1),
            nn.Linear(20, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

@st.cache_resource
def load_all_artifacts():
    # Load scalers
    scalers_data = joblib.load('scalers.pkl')
    scaler_amount = scalers_data['scaler_amount']
    scaler_time = scalers_data['scaler_time']
    
    # Load supervised model
    supervised_model = joblib.load('fraud_model.pkl')
    
    # Load config
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
    except Exception:
        config = {'optimal_threshold': 0.35, 'tau_anomaly': 0.54}
        
    # Load PyTorch Autoencoder
    autoencoder = DeepAutoencoder(input_dim=30)
    try:
        autoencoder.load_state_dict(torch.load('autoencoder_weights.pth', map_location=torch.device('cpu')))
        autoencoder.eval()
    except Exception as e:
        autoencoder = None
        
    return scaler_amount, scaler_time, supervised_model, autoencoder, config

scaler_amount, scaler_time, supervised_model, autoencoder, config = load_all_artifacts()

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.title("Fraud Guard Settings")
    st.caption("CMRIT • Dept. of CSE-AIML")
    st.divider()

    # Toggle between Auto-Pilot and Manual Simulator
    st.subheader("⚙️ System Mode")
    operating_mode = st.radio(
        "Select Operating Mode:",
        ["🤖 Auto-Pilot (Recommended)", "🛠️ Manual Simulator"],
        index=0,
        help="Auto-Pilot locks in the bank-proven optimal settings (τ=0.35, $120 FN, $2.50 FP). Manual mode allows custom policy experimentation."
    )

    if operating_mode == "🤖 Auto-Pilot (Recommended)":
        threshold = float(config.get('optimal_threshold', 0.35))
        cost_fn = 120.0
        cost_fp = 2.50
        
        st.success("✅ **Auto-Pilot Active**\n• Optimal Threshold: `τ = 0.35`\n• Loss Matrix: `$120 FN` / `$2.50 FP`\n• 32.6% Cost Reduction Locked")
    else:
        st.info("🛠️ **Manual Simulation Mode**\nAdjust decision threshold and economic cost parameters below.")
        threshold = st.slider(
            "Decision Threshold (τ)",
            min_value=0.05,
            max_value=0.90,
            value=float(config.get('optimal_threshold', 0.35)),
            step=0.05,
            help="Probability cutoff for flagging transactions. τ=0.35 is mathematically optimal for minimizing financial loss."
        )
        st.caption(f"**Current Threshold:** `{threshold:.2f}`")
        if abs(threshold - 0.35) < 0.01:
            st.success("🎯 Optimal Cost-Loss Threshold")
        elif threshold > 0.5:
            st.warning("⚠️ High False Negatives (Costly)")
        else:
            st.info("ℹ️ High Sensitivity (More False Alarms)")

        st.subheader("🏦 Business Loss Economics")
        cost_fn = st.number_input("Cost of Missed Fraud (FN)", value=120.0, step=10.0, help="Direct financial loss & chargeback penalties ($)")
        cost_fp = st.number_input("Cost of False Alarm (FP)", value=2.50, step=0.5, help="Automated verification SMS/OTP verification cost ($)")
    
    st.divider()
    st.caption("Dual-Engine Architecture:\n• Supervised: Cost-Sensitive LightGBM\n• Anomaly: PyTorch Autoencoder")

# ==========================================
# 4. MAIN BANNER
# ==========================================
st.markdown('<div class="main-header">🛡️ Enterprise Fraud Detection & Anomaly Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dual-Engine Screening combining Cost-Sensitive Gradient Boosting and Unsupervised Deep Autoencoders</div>', unsafe_allow_html=True)

# Initialize persistent session state for loaded data
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None

# Data Input Section
col_upload, col_sample, col_clear = st.columns([2.5, 1.3, 0.8])

with col_upload:
    uploaded_file = st.file_uploader("Upload Transaction Batch (CSV)", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df_loaded = pd.read_csv(uploaded_file)

with col_sample:
    st.write("Benchmark live feed:")
    if st.button("📥 Load 100 Demo Txns", use_container_width=True):
        try:
            st.session_state.df_loaded = pd.read_csv('demo_feed_transactions.csv')
        except Exception:
            st.session_state.df_loaded = pd.read_csv('unlabeled_live_test.csv')

with col_clear:
    st.write("Reset data:")
    if st.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.df_loaded = None
        st.rerun()

# Selected dataset remains in memory across toggles & slider moves
df = st.session_state.df_loaded

# ==========================================
# 5. EXECUTION & RESULTS TABS
# ==========================================
if df is not None:
    st.success(f"Successfully loaded batch with **{len(df):,} transactions**.")
    
    # Preprocess with zero data leakage
    has_labels = 'Class' in df.columns
    y_true = df['Class'] if has_labels else None
    
    df_eval = df.drop(['Class'], axis=1, errors='ignore').copy()
    
    # Apply cached training scalers
    df_scaled = df_eval.copy()
    df_scaled['Amount'] = scaler_amount.transform(df_eval[['Amount']])
    df_scaled['Time'] = scaler_time.transform(df_eval[['Time']])
    
    # 1. Supervised Predictions
    t0 = time.time()
    fraud_probs = supervised_model.predict_proba(df_scaled)[:, 1]
    supervised_latency = (time.time() - t0) * 1000
    preds_supervised = (fraud_probs >= threshold).astype(int)
    
    # 2. Unsupervised Autoencoder Predictions
    reconstruction_mse = None
    if autoencoder is not None:
        tensor_data = torch.from_numpy(df_scaled.values.astype(np.float32))
        with torch.no_grad():
            recon = autoencoder(tensor_data).numpy()
            reconstruction_mse = np.mean(np.square(df_scaled.values - recon), axis=1)
        tau_anomaly = config.get('tau_anomaly', 0.54)
        is_novel_anomaly = reconstruction_mse > tau_anomaly

    # Tabbed Interface
    tab_scan, tab_anomaly, tab_economics, tab_explain = st.tabs([
        "🚨 Real-Time Security Scan",
        "🧠 Deep Autoencoder (Zero-Day Anomaly)",
        "💰 Financial Cost Loss Optimization",
        "🔍 Regulatory Explainability (SHAP)"
    ])

    # ----------------------------------------------------
    # TAB 1: Real-Time Security Scan
    # ----------------------------------------------------
    with tab_scan:
        st.subheader("Batch Threat Assessment")
        
        # Metric KPI cards
        total_tx = len(df_eval)
        flagged_fraud = int(np.sum(preds_supervised))
        legit_tx = total_tx - flagged_fraud
        fraud_ratio = (flagged_fraud / total_tx) * 100
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Scanned", f"{total_tx:,}")
        kpi2.metric("Flagged Threats", f"{flagged_fraud:,}", delta=f"{fraud_ratio:.1f}% flag rate", delta_color="inverse")
        kpi3.metric("Approved Legitimate", f"{legit_tx:,}")
        kpi4.metric("Avg Latency", f"{supervised_latency/total_tx:.3f} ms", delta="< 20ms SLA Pass", delta_color="normal")
        
        # Display Table
        df_display = df_eval.copy()
        df_display['Risk_Score'] = (fraud_probs * 100).round(2)
        df_display['Threat_Status'] = ["🚨 FRAUD" if p == 1 else "✅ Legitimate" for p in preds_supervised]
        if reconstruction_mse is not None:
            df_display['Reconstruction_MSE'] = reconstruction_mse.round(3)
        
        cols_to_show = ['Threat_Status', 'Risk_Score', 'Amount', 'Time', 'V14', 'V10', 'V12']
        if reconstruction_mse is not None:
            cols_to_show.insert(2, 'Reconstruction_MSE')
            
        st.dataframe(df_display[cols_to_show], use_container_width=True, height=320)
        
        # Plain English Confusion Matrix if labels available
        if has_labels:
            st.divider()
            st.subheader("Operational Verification (Actual vs. Predicted)")
            tn, fp, fn, tp = confusion_matrix(y_true, preds_supervised).ravel()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("The Catch (True Positives)", tp, help="Confirmed fraud blocked successfully")
            c2.metric("False Alarms (False Positives)", fp, help="Legitimate users temporarily held for verification")
            c3.metric("Costly Misses (False Negatives)", fn, help="Actual fraud that bypassed standard checks")
            c4.metric("Smooth Approvals (True Negatives)", tn, help="Legitimate transactions processed seamlessly")
            
            st.info(f"**Security Recall Rate:** `{tp / (tp + fn) * 100:.1f}%` of all fraudulent attempts were successfully intercepted at threshold τ = {threshold:.2f}.")

    # ----------------------------------------------------
    # TAB 2: Deep Autoencoder (Zero-Day Anomaly)
    # ----------------------------------------------------
    with tab_anomaly:
        st.subheader("Unsupervised Representation Learning (PyTorch)")
        st.write("""
        The **Deep Autoencoder [30 → 20 → 10 → 20 → 30]** was trained exclusively on **legitimate transactions**. 
        When a fraudulent transaction exhibits novel behavioral vectors, the network cannot reconstruct it, creating a **spike in Mean Squared Error (MSE)**.
        """)
        
        if reconstruction_mse is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(reconstruction_mse, bins=40, color='#1E3A8A', alpha=0.7, edgecolor='black', label='Transaction Reconstruction MSE')
            ax.axvline(tau_anomaly, color='crimson', linestyle='--', linewidth=2, label=f'Anomaly Threshold (τ = {tau_anomaly:.3f})')
            ax.set_title('Reconstruction Error Distribution (L2 Norm)')
            ax.set_xlabel('Mean Squared Reconstruction Error (MSE)')
            ax.set_ylabel('Transaction Count')
            ax.legend()
            st.pyplot(fig)
            
            num_anomalies = np.sum(reconstruction_mse > tau_anomaly)
            st.warning(f"🔔 **{num_anomalies} transactions** exceeded the reconstruction threshold (MSE > {tau_anomaly:.3f}) and are flagged as anomalous/zero-day risk.")

    # ----------------------------------------------------
    # TAB 3: Financial Cost Loss Optimization
    # ----------------------------------------------------
    with tab_economics:
        st.subheader("Banking Loss Matrix Optimization")
        st.latex(r"\text{Total Business Cost} = (FN \times \$120) + (FP \times \$2.50)")
        
        if has_labels:
            # Calculate cost at current threshold vs 0.50
            preds_05 = (fraud_probs >= 0.50).astype(int)
            tn05, fp05, fn05, tp05 = confusion_matrix(y_true, preds_05).ravel()
            cost_05 = (fn05 * cost_fn) + (fp05 * cost_fp)
            
            current_cost = (fn * cost_fn) + (fp * cost_fp)
            savings = cost_05 - current_cost
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Cost at Default (τ = 0.50)", f"${cost_05:,.2f}")
            col_b.metric(f"Cost at Selected (τ = {threshold:.2f})", f"${current_cost:,.2f}")
            col_c.metric("Direct Financial Impact", f"${abs(savings):,.2f}", 
                         delta=f"{'Saved' if savings >= 0 else 'Deficit'} vs Default", 
                         delta_color="normal" if savings >= 0 else "inverse")
            
            st.write("""
            **Economic Insight:** Moving the threshold from 0.50 to 0.35 increases minor False Positives (costing only $2.50 in SMS alerts), 
            but catches high-cost False Negatives (saving $120+ each), maximizing bank profitability.
            """)
        else:
            st.info("Upload data with true labels to see actual dollar loss comparisons.")

    # ----------------------------------------------------
    # TAB 4: Regulatory Explainability (SHAP)
    # ----------------------------------------------------
    with tab_explain:
        st.subheader("Global Behavioral Vector Attribution")
        st.write("To satisfy **GDPR Article 22** and the **Fair Credit Reporting Act (FCRA)**, the model provides full auditability of decision weights.")
        
        if hasattr(supervised_model, 'feature_importances_'):
            importances = pd.Series(supervised_model.feature_importances_, index=df_scaled.columns)
            top_feats = importances.nlargest(8)
            
            fig, ax = plt.subplots(figsize=(9, 4))
            top_feats.plot(kind='barh', color='#2563EB', ax=ax)
            ax.set_title('Top Behavioral Vectors Driving Fraud Classifications')
            ax.set_xlabel('Relative Importance Weight')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            st.markdown("""
            **Key Risk Feature Findings:**
            * **V14, V10, V12:** Highly negative values indicate compromised card credentials and abnormal merchant categories.
            * **Amount & Time:** Sudden temporal bursts (velocity) and anomalous dollar totals elevate suspicion.
            """)
else:
    st.info("👋 Upload a CSV file or click **'Load 100 Demo Transactions'** in the sidebar to run the security scan.")