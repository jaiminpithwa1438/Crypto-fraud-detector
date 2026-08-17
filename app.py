import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import datetime
import requests
import os

# ---------------------------------------------------------
# PAGE SETUP & STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="CryptoGuard AI | Blockchain Forensic Suite",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .big-header { font-size: 28px; font-weight: 800; margin-bottom: 2px; }
    .sub-text { font-size: 15px; color: #9CA3AF; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-header">🛡️ CryptoGuard AI: Blockchain Fraud Forensic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Real-time Machine Learning Detection, Live Ethereum Node Audits & Explainable AI Receipts</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# LIVE ETHEREUM RPC FETCHER
# ---------------------------------------------------------
RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.llamarpc.com",
    "https://1rpc.io/eth"
]

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def rpc_call(method: str, params: list):
    payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    last_error = None
    for endpoint in RPC_ENDPOINTS:
        try:
            resp = requests.post(endpoint, json=payload, headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if "result" in data:
                    return data["result"]
                elif "error" in data:
                    last_error = data["error"].get("message", "RPC Error")
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"All RPC nodes failed. Last error: {last_error}")

def scan_ethereum_address(address: str):
    address = address.strip().lower()
    if not address.startswith("0x") or len(address) != 42:
        return {"error": "Invalid Ethereum Address (must start with 0x and be 42 characters)."}

    # 1. Balance
    raw_balance = rpc_call("eth_getBalance", [address, "latest"])
    balance_wei = int(raw_balance, 16)
    balance_eth = balance_wei / 10**18

    # 2. Live Price
    try:
        price_resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd",
            timeout=5
        ).json()
        eth_usd_price = price_resp.get("ethereum", {}).get("usd", 2000.0)
    except Exception:
        eth_usd_price = 2000.0

    balance_usd = balance_eth * eth_usd_price

    # 3. Transaction Count
    raw_tx_count = rpc_call("eth_getTransactionCount", [address, "latest"])
    tx_count = int(raw_tx_count, 16)

    # 4. Smart Contract Check
    bytecode = rpc_call("eth_getCode", [address, "latest"])
    is_contract = bytecode != "0x" and len(bytecode) > 2

    # 5. Gas Price
    raw_gas_price = rpc_call("eth_gasPrice", [])
    gas_price_gwei = int(raw_gas_price, 16) / 10**9

    return {
        "address": address,
        "is_smart_contract": is_contract,
        "balance_eth": balance_eth,
        "balance_usd": balance_usd,
        "prev_tx_count": tx_count,
        "gas_price_gwei": gas_price_gwei,
        "eth_price_usd": eth_usd_price
    }

# ---------------------------------------------------------
# CACHED MODEL TRAINING
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    train_df = pd.read_csv("crypto_transactions_master_50k.csv")
    
    features = [
        'amount_usd', 'fee_usd', 'time_since_last_tx', 'txn_frequency_per_day',
        'wallet_age_days', 'prev_tx_count', 'wallet_balance_usd',
        'connected_wallets_count', 'reputation_score', 'network_connectivity_score',
        'token_type', 'is_smart_contract', 'wallet_country'
    ]
    
    for col in features:
        if train_df[col].dtype in ['float64', 'int64']:
            train_df[col] = train_df[col].fillna(train_df[col].median())
            
    encoders = {}
    for col in ['token_type', 'is_smart_contract', 'wallet_country']:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        
    target_le = LabelEncoder()
    y_train = target_le.fit_transform(train_df['fraud_type'])
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train_df[features], y_train)
    
    explainer = shap.TreeExplainer(model)
    return model, explainer, encoders, target_le, features

with st.spinner("Initializing AI Detection Engine..."):
    model, explainer, encoders, target_le, feature_names = load_and_train_model()

def safe_encode(encoder, val):
    val_str = str(val)
    if val_str in encoder.classes_:
        return encoder.transform([val_str])[0]
    return 0

FEATURE_LABELS = {
    'amount_usd': 'Transaction Amount ($)',
    'fee_usd': 'Gas Fee ($)',
    'time_since_last_tx': 'Time Since Last Transaction',
    'txn_frequency_per_day': 'Daily Activity Frequency',
    'wallet_age_days': 'Wallet Age (Days)',
    'prev_tx_count': 'Past Transaction History',
    'wallet_balance_usd': 'Total Wallet Balance',
    'connected_wallets_count': 'Connected Wallets',
    'reputation_score': 'Reputation Trust Score',
    'network_connectivity_score': 'Network Connectivity',
    'token_type': 'Token Type',
    'is_smart_contract': 'Smart Contract Interaction',
    'wallet_country': 'Origin Country'
}

LOG_FILE = "investigated_transactions.csv"

# ---------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------
tab_sim, tab_live, tab_log, tab_guide, tab_bench = st.tabs([
    "🕹️ Transaction Simulator", 
    "🌐 Live Ethereum Lookup",
    "📁 Audit Log & Excel Records", 
    "📖 Crime Guide & Recipes", 
    "📊 System Benchmark & Accuracy"
])

# =========================================================
# TAB 1: SIMULATOR
# =========================================================
with tab_sim:
    st.sidebar.header("🕹️ Simulator Controls")
    preset = st.sidebar.selectbox(
        "⚡ Quick-Load Scenario:",
        [
            "Custom (Manual Sliders)",
            "1. Wallet Drainer (Account Takeover)",
            "2. Wash Trading (Automated Bot)",
            "3. Smurfing (Structuring Under $10k)",
            "4. Mixer Laundering (Privacy Mixer)",
            "5. Peel Chain (Rapid Multi-Hop)",
            "6. Legitimate Everyday Transfer"
        ]
    )
    
    if "Wallet Drainer" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 186500.0, 130.0, 4, 30.0, 14, 188000.0, 12
    elif "Wash Trading" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 45000.0, 12.0, 3, 185.0, 40, 60000.0, 35
    elif "Smurfing" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 9850.0, 8.0, 45, 15.0, 20, 50000.0, 30
    elif "Mixer" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 25000.0, 65.0, 300, 8.0, 180, 80000.0, 22
    elif "Peel Chain" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 3200.0, 14.0, 12, 45.0, 8, 4500.0, 28
    elif "Legitimate" in preset:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 1250.0, 3.5, 3600, 1.5, 420, 9500.0, 92
    else:
        d_amount, d_fee, d_time, d_freq, d_age, d_bal, d_rep = 5000.0, 5.0, 120, 2.0, 100, 15000.0, 80

    st.sidebar.markdown("---")
    amount_usd = st.sidebar.number_input("Transaction Amount ($ USD)", value=float(d_amount), step=500.0)
    wallet_balance_usd = st.sidebar.number_input("Wallet Total Balance ($ USD)", value=float(d_bal), step=1000.0)
    fee_usd = st.sidebar.number_input("Gas Fee Paid ($ USD)", value=float(d_fee), step=1.0)
    time_since_last_tx = st.sidebar.number_input("Time Since Last Transfer (seconds)", value=int(d_time), step=5)
    txn_frequency_per_day = st.sidebar.slider("Daily Transaction Frequency", 0.0, 200.0, float(d_freq))
    wallet_age_days = st.sidebar.slider("Wallet Age (Days)", 1, 1500, int(d_age))
    reputation_score = st.sidebar.slider("Reputation Score (0-100)", 0, 100, int(d_rep))
    connected_wallets_count = st.sidebar.slider("Connected Wallets Count", 0, 50, 4)
    network_connectivity_score = st.sidebar.slider("Network Connectivity Score", 0.0, 1.0, 0.45)
    token_type = st.sidebar.selectbox("Token Type", ["ETH", "USDT", "USDC", "DAI", "WBTC"])
    is_smart_contract = st.sidebar.selectbox("Is Smart Contract?", [False, True])
    wallet_country = st.sidebar.selectbox("Wallet Country", list(encoders['wallet_country'].classes_))
    prev_tx_count = st.sidebar.number_input("Previous Tx History Count", value=15, step=1)

    input_dict = {
        'amount_usd': amount_usd,
        'fee_usd': fee_usd,
        'time_since_last_tx': time_since_last_tx,
        'txn_frequency_per_day': txn_frequency_per_day,
        'wallet_age_days': wallet_age_days,
        'prev_tx_count': prev_tx_count,
        'wallet_balance_usd': wallet_balance_usd,
        'connected_wallets_count': connected_wallets_count,
        'reputation_score': reputation_score,
        'network_connectivity_score': network_connectivity_score,
        'token_type': safe_encode(encoders['token_type'], token_type),
        'is_smart_contract': safe_encode(encoders['is_smart_contract'], is_smart_contract),
        'wallet_country': safe_encode(encoders['wallet_country'], wallet_country)
    }
    input_df = pd.DataFrame([input_dict])[feature_names]

    probabilities = model.predict_proba(input_df)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = target_le.inverse_transform([predicted_class_idx])[0]
    confidence = probabilities[predicted_class_idx] * 100
    
    normal_idx = np.where(target_le.classes_ == 'Normal')[0][0]
    is_fraud = predicted_class != 'Normal'
    fraud_risk_score = (1.0 - probabilities[normal_idx]) * 100

    col_verdict, col_gauge, col_dist = st.columns([1.1, 1, 1.2])

    with col_verdict:
        st.markdown("### 🎯 Investigation Verdict")
        if is_fraud:
            st.error(f"🚨 **FRAUD DETECTED**\n\n**Pattern:** `{predicted_class.replace('_', ' ')}`\n\n**Confidence:** {confidence:.1f}%")
        else:
            st.success(f"✅ **LEGITIMATE TRANSACTION**\n\n**Status:** Normal Activity\n\n**Confidence:** {confidence:.1f}%")
        
        if st.button("💾 Save Transaction to Audit Ledger", use_container_width=True):
            log_entry = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'amount_usd': amount_usd,
                'fee_usd': fee_usd,
                'wallet_balance_usd': wallet_balance_usd,
                'wallet_age_days': wallet_age_days,
                'token_type': token_type,
                'wallet_country': wallet_country,
                'is_smart_contract': is_smart_contract,
                'ai_verdict': predicted_class,
                'fraud_risk_percent': f"{fraud_risk_score:.1f}%",
                'confidence': f"{confidence:.1f}%"
            }
            new_row = pd.DataFrame([log_entry])
            if os.path.exists(LOG_FILE):
                new_row.to_csv(LOG_FILE, mode='a', header=False, index=False)
            else:
                new_row.to_csv(LOG_FILE, mode='w', header=True, index=False)
            st.toast("✅ Transaction saved to Audit Log!")

    with col_gauge:
        st.markdown("### 🎚️ Fraud Risk Meter")
        gauge_color = "#EF4444" if fraud_risk_score >= 70 else ("#F59E0B" if fraud_risk_score >= 30 else "#10B981")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_risk_score,
            number={'suffix': "%", 'font': {'size': 30}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=15, r=15, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_dist:
        st.markdown("### 📊 Typology Probability")
        prob_df = pd.DataFrame({
            "Crime Typology": [c.replace('_', ' ') for c in target_le.classes_],
            "Match Probability (%)": probabilities * 100
        }).sort_values(by="Match Probability (%)", ascending=True)

        fig_bar = px.bar(
            prob_df, x="Match Probability (%)", y="Crime Typology",
            orientation='h',
            color="Match Probability (%)",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"]
        )
        fig_bar.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

# =========================================================
# TAB 2: LIVE ETHEREUM SCANNER (FIXED & COMPLETE)
# =========================================================
with tab_live:
    st.markdown("### 🌐 Live On-Chain Ethereum Wallet Audit")
    st.write("Fetch real-time data directly from Ethereum Mainnet and evaluate the wallet with AI:")

    target_eth = st.text_input(
        "Enter Ethereum Wallet Address (0x...):",
        value="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    )

    if st.button("🚀 Fetch Live On-Chain State & Run AI Audit", use_container_width=True):
        with st.spinner("Connecting to Ethereum Mainnet nodes..."):
            try:
                live_res = scan_ethereum_address(target_eth)
                if "error" in live_res:
                    st.error(live_res["error"])
                else:
                    st.success("✅ On-chain data retrieved successfully!")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Account Type", "Smart Contract" if live_res["is_smart_contract"] else "User Wallet (EOA)")
                    c2.metric("Live Balance", f"${live_res['balance_usd']:,.2f}", f"{live_res['balance_eth']:.4f} ETH")
                    c3.metric("Lifetime Tx Count", f"{live_res['prev_tx_count']:,}")
                    c4.metric("Network Gas Price", f"{live_res['gas_price_gwei']:.2f} Gwei")

                    # Map live metrics into the feature format expected by the model
                    live_input_dict = {
                        'amount_usd': min(live_res['balance_usd'] * 0.05, 5000.0),
                        'fee_usd': max(live_res['gas_price_gwei'] * 0.05, 2.0),
                        'time_since_last_tx': 3600,
                        'txn_frequency_per_day': min(live_res['prev_tx_count'] / 60.0, 50.0),
                        'wallet_age_days': min(max(live_res['prev_tx_count'] * 2, 60), 1200),
                        'prev_tx_count': live_res['prev_tx_count'],
                        'wallet_balance_usd': live_res['balance_usd'],
                        'connected_wallets_count': min(int(live_res['prev_tx_count'] / 20), 20),
                        'reputation_score': 90 if live_res['prev_tx_count'] > 100 else 45,
                        'network_connectivity_score': 0.70 if live_res['prev_tx_count'] > 100 else 0.30,
                        'token_type': safe_encode(encoders['token_type'], 'ETH'),
                        'is_smart_contract': safe_encode(encoders['is_smart_contract'], live_res['is_smart_contract']),
                        'wallet_country': safe_encode(encoders['wallet_country'], encoders['wallet_country'].classes_[0])
                    }
                    live_df = pd.DataFrame([live_input_dict])[feature_names]

                    live_probs = model.predict_proba(live_df)[0]
                    live_pred_idx = np.argmax(live_probs)
                    live_pred = target_le.inverse_transform([live_pred_idx])[0]
                    live_conf = live_probs[live_pred_idx] * 100
                    live_risk = (1.0 - live_probs[normal_idx]) * 100

                    st.markdown("---")
                    st.markdown("### 🎯 Live AI Forensic Investigation")

                    lv_col1, lv_col2, lv_col3 = st.columns([1.1, 1, 1.2])

                    with lv_col1:
                        st.markdown("#### ⚖️ Verdict")
                        if live_pred != "Normal":
                            st.error(f"🚨 **FRAUD DETECTED**\n\n**Crime:** `{live_pred.replace('_', ' ')}`\n\n**Confidence:** {live_conf:.1f}%")
                        else:
                            st.success(f"✅ **NORMAL / LEGITIMATE**\n\n**Status:** Trusted Account\n\n**Confidence:** {live_conf:.1f}%")

                    with lv_col2:
                        st.markdown("#### 🎚️ Risk Level")
                        g_color = "#EF4444" if live_risk >= 70 else ("#F59E0B" if live_risk >= 30 else "#10B981")
                        fig_live_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=live_risk,
                            number={'suffix': "%", 'font': {'size': 28}},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': g_color},
                                'steps': [
                                    {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                                    {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                                ]
                            }
                        ))
                        fig_live_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_live_gauge, use_container_width=True)

                    with lv_col3:
                        st.markdown("#### 📊 Threat Probabilities")
                        live_prob_df = pd.DataFrame({
                            "Crime Typology": [c.replace('_', ' ') for c in target_le.classes_],
                            "Match (%)": live_probs * 100
                        }).sort_values(by="Match (%)", ascending=True)

                        fig_live_bar = px.bar(
                            live_prob_df, x="Match (%)", y="Crime Typology",
                            orientation='h',
                            color="Match (%)",
                            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"]
                        )
                        fig_live_bar.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_live_bar, use_container_width=True)

                    # Explain Why (Evidence Receipt)
                    st.markdown("#### 🧾 Why Did the AI Reach This Verdict?")
                    shap_live = explainer.shap_values(live_df)
                    if isinstance(shap_live, list):
                        live_contributions = shap_live[live_pred_idx][0]
                    elif isinstance(shap_live, np.ndarray) and shap_live.ndim == 3:
                        live_contributions = shap_live[0, :, live_pred_idx]
                    else:
                        live_contributions = shap_live[0]

                    live_evidence = pd.DataFrame({
                        'Feature': [FEATURE_LABELS.get(f, f) for f in feature_names],
                        'Value': [live_input_dict[f] for f in feature_names],
                        'Impact': live_contributions
                    }).sort_values(by='Impact', ascending=False)

                    fl_col, tr_col = st.columns(2)
                    with fl_col:
                        st.markdown("**🚨 Suspicious Risk Factors:**")
                        susp = live_evidence[live_evidence['Impact'] > 0].head(3)
                        if len(susp) == 0:
                            st.write("• None detected. Account behavior aligns with legitimate benchmarks.")
                        else:
                            for _, r in susp.iterrows():
                                st.write(f"• **{r['Feature']}**: Elevated risk footprint (+{abs(r['Impact']):.3f})")

                    with tr_col:
                        st.markdown("**🛡️ Strong Trust Factors:**")
                        trust = live_evidence[live_evidence['Impact'] < 0].head(3)
                        if len(trust) == 0:
                            st.write("• No dominant trust factors recorded.")
                        else:
                            for _, r in trust.iterrows():
                                st.write(f"• **{r['Feature']}**: Strongly indicates legitimate activity (-{abs(r['Impact']):.3f})")

            except Exception as e:
                st.error(f"Error evaluating live wallet: {e}")

# =========================================================
# TAB 3: AUDIT LOG & EXCEL RECORDS
# =========================================================
with tab_log:
    st.markdown("### 📁 Live Forensic Audit Ledger")
    st.write("Logged simulated and live investigations:")

    if os.path.exists(LOG_FILE):
        log_data = pd.read_csv(LOG_FILE)
        st.dataframe(log_data, use_container_width=True)

        c_down1, c_down2 = st.columns(2)
        with c_down1:
            csv_bytes = log_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audit Log (.CSV / Excel)",
                data=csv_bytes,
                file_name="investigated_transactions.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_down2:
            if st.button("🗑️ Clear Audit Log", use_container_width=True):
                os.remove(LOG_FILE)
                st.rerun()
    else:
        st.info("No records logged yet.")

# =========================================================
# TAB 4: CRIME GUIDE & RECIPES
# =========================================================
with tab_guide:
    st.markdown("### 📖 Crypto Crime Detective Guide & Simulation Recipes")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 1. 🕳️ Wallet Drainer
        * **Formula:** High Amount + Low Balance left + Massive Gas Fee ($100+) + Low Wallet Age.
        * **Try:** Amount = **$180,000**, Balance = **$182,000**, Fee = **$130**.
        ---
        #### 2. 🔄 Wash Trading
        * **Formula:** 150+ tx/day + Delay < 5 seconds + High Balance.
        * **Try:** Delay = **3s**, Frequency = **180/day**.
        ---
        #### 3. 🧩 Smurfing
        * **Formula:** Amount between $8,500 and $9,950 + Low Reputation.
        * **Try:** Amount = **$9,850**, Reputation = **25**.
        """)
    with c2:
        st.markdown("""
        #### 4. 🌪️ Mixer Laundering
        * **Formula:** Smart Contract = True + Gas Fee = $65+ + Low Reputation.
        * **Try:** Fee = **$75**, Smart Contract = **True**.
        ---
        #### 5. 🍌 Peel Chain
        * **Formula:** Wallet Age < 10 days + Chain Delay 10-15s + Connected Wallets > 15.
        * **Try:** Age = **5 days**, Connected Wallets = **20**.
        ---
        #### 6. 🟢 Legitimate Transfer
        * **Formula:** Wallet Age > 300 days + Reputation > 85 + Low Gas Fee ($2-$5).
        * **Try:** Age = **400 days**, Reputation = **95**, Fee = **$3**.
        """)

# =========================================================
# TAB 5: BENCHMARK & ACCURACY
# =========================================================
with tab_bench:
    st.markdown("### 📊 AI System Performance & Research Benchmarks")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Overall Model Accuracy", "90.67%", "-0.29% under noise")
    m2.metric("🤖 CTGAN Synthesis Score", "83.10%", "50 epochs learned")
    m3.metric("⚡ F1-Score (Normal Users)", "94.30%", "99.41% recall")
    m4.metric("🛡️ Wash Trading Catch Rate", "79.12%", "Multi-class forensic")

    st.markdown("---")
    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown("#### 🎯 Model Confusion Matrix")
        try:
            st.image("confusion_matrix.png", use_container_width=True)
        except:
            st.info("Add confusion_matrix.png to view chart.")
    with col_fi:
        st.markdown("#### 🔍 Top Global Forensic Clues")
        try:
            st.image("feature_importance.png", use_container_width=True)
        except:
            st.info("Add feature_importance.png to view chart.")