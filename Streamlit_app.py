"""
streamlit_app.py  —  Customer Churn Prediction Dashboard
─────────────────────────────────────────────────────────
Run:
    streamlit run streamlit_app.py

Requires in the same folder:
    • final_pipeline.pkl
    • preprocess.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from preprocess import engineer_features

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0f1a 0%, #111428 50%, #0a0d1e 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1229 0%, #151833 100%);
    border-right: 1px solid #2a2d4a;
}
[data-testid="stSidebar"] * { color: #c8cce8 !important; }

.main-header {
    background: linear-gradient(135deg, #1a1f3e 0%, #252a52 100%);
    border: 1px solid #3a3f6e;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00e5ff, #7c4dff, #ff4081);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}
.main-header p { color: #8891c0; font-size: 1rem; margin: 0; }

.metric-card {
    background: linear-gradient(135deg, #1a1f3e 0%, #20254a 100%);
    border: 1px solid #2e3460;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card .label {
    font-size: 0.75rem;
    color: #6b75a8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 8px;
}
.metric-card .value { font-size: 1.9rem; font-weight: 800; color: #e8eaf6; }
.metric-card .sub   { font-size: 0.78rem; color: #6b75a8; margin-top: 4px; }

.result-churn {
    background: linear-gradient(135deg, #2d1b1b 0%, #3d1f1f 100%);
    border: 2px solid #ff4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,68,68,0.2);
}
.result-safe {
    background: linear-gradient(135deg, #1b2d1b 0%, #1f3d20 100%);
    border: 2px solid #00e676;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,230,118,0.2);
}
.result-label {
    font-size: 0.9rem;
    color: #8891c0;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.result-value  { font-size: 2.8rem; font-weight: 900; margin: 4px 0; }
.churn-text    { color: #ff5252; }
.safe-text     { color: #69ff47; }
.result-prob   { font-size: 1rem; color: #b0b8d8; margin-top: 8px; }

.risk-HIGH   { background:#ff1744; color:#fff; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }
.risk-MEDIUM { background:#ff9100; color:#fff; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }
.risk-LOW    { background:#00e676; color:#000; padding:5px 16px; border-radius:20px; font-weight:700; font-size:0.85rem; display:inline-block; }

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #7c8bcc;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    border-bottom: 1px solid #2a2d4a;
    padding-bottom: 8px;
    margin: 20px 0 16px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #7c4dff, #00e5ff);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(124,77,255,0.4);
}

.stTabs [data-baseweb="tab-list"] {
    background: #1a1f3e;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #6b75a8;
    font-weight: 600;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c4dff, #00e5ff) !important;
    color: white !important;
}
[data-testid="stFileUploader"] {
    background: #1a1f3e;
    border: 2px dashed #3a3f6e;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load("final_pipeline.pkl")

pipeline  = load_model()
THRESHOLD = 0.50    # ← change after threshold tuning


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def predict_single_row(data: dict) -> dict:
    df       = pd.DataFrame([data])
    feats    = engineer_features(df)
    proba    = float(pipeline.predict_proba(feats)[0][1])
    pred     = int(proba >= THRESHOLD)
    risk     = "HIGH" if proba >= 0.70 else "MEDIUM" if proba >= 0.40 else "LOW"
    return {"probability": proba, "prediction": pred, "risk": risk}


def predict_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    feats       = engineer_features(df_raw)
    probas      = pipeline.predict_proba(feats)[:, 1]
    predictions = (probas >= THRESHOLD).astype(int)
    result      = df_raw.copy()
    result["Churn_Probability"] = np.round(probas, 4)
    result["Churn_Prediction"]  = predictions
    result["Churn_Label"]       = np.where(predictions == 1, "Yes", "No")
    result["Risk_Level"]        = np.where(
        probas >= 0.70, "HIGH", np.where(probas >= 0.40, "MEDIUM", "LOW")
    )
    return result


def gauge_chart(prob: float):
    color = "#ff5252" if prob >= THRESHOLD else "#69ff47"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(prob * 100, 1),
        number = {"suffix": "%", "font": {"size": 34, "color": color}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#6b75a8",
                     "tickfont": {"color": "#6b75a8"}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1f3e",
            "bordercolor": "#2e3460",
            "steps": [
                {"range": [0,  40], "color": "#1b2d1b"},
                {"range": [40, 70], "color": "#2d2a1b"},
                {"range": [70,100], "color": "#2d1b1b"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        },
    ))
    fig.update_layout(
        height=230, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="#0d0f1a", font_color="#e8eaf6",
    )
    return fig


def donut_chart(n_churn: int, n_safe: int):
    fig = go.Figure(go.Pie(
        labels=["Will Churn", "Will Stay"],
        values=[n_churn, n_safe],
        hole=0.65,
        marker_colors=["#ff5252", "#69ff47"],
        textinfo="percent",
        textfont_color="#ffffff",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(
        height=270, paper_bgcolor="#0d0f1a", font_color="#e8eaf6",
        legend=dict(bgcolor="#1a1f3e", bordercolor="#2e3460",
                    font=dict(color="#e8eaf6")),
        margin=dict(l=10, r=10, t=20, b=10),
        annotations=[dict(text=f"<b>{n_churn+n_safe:,}</b><br>Total",
                          font_size=13, font_color="#e8eaf6", showarrow=False)],
    )
    return fig


def risk_bar_chart(df_result: pd.DataFrame):
    counts = df_result["Risk_Level"].value_counts().reindex(
        ["HIGH", "MEDIUM", "LOW"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=["#ff5252", "#ff9100", "#69ff47"],
        text=counts.values.tolist(),
        textposition="outside",
        textfont_color="#e8eaf6",
    ))
    fig.update_layout(
        height=260, paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
        font_color="#e8eaf6",
        xaxis=dict(showgrid=False, color="#6b75a8"),
        yaxis=dict(showgrid=True, gridcolor="#2a2d4a", color="#6b75a8"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def prob_histogram(df_result: pd.DataFrame):
    fig = px.histogram(df_result, x="Churn_Probability",
                       nbins=30, color_discrete_sequence=["#7c4dff"])
    fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="#ff4081",
                  annotation_text="Threshold", annotation_font_color="#ff4081")
    fig.update_layout(
        height=260, paper_bgcolor="#0d0f1a", plot_bgcolor="#0d0f1a",
        font_color="#e8eaf6",
        xaxis=dict(showgrid=False, color="#6b75a8", title="Churn Probability"),
        yaxis=dict(showgrid=True, gridcolor="#2a2d4a", color="#6b75a8", title="Count"),
        margin=dict(l=10, r=10, t=30, b=10), bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px 0;'>
        <div style='font-size:2.5rem;'>📡</div>
        <div style='font-size:1.1rem;font-weight:800;color:#00e5ff;'>ChurnSense</div>
        <div style='font-size:0.75rem;color:#555d8a;margin-top:4px;'>ML Ensemble · Production</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ Model**")
    st.markdown(f"""
    <div style='font-size:0.82rem;color:#8891c0;line-height:2.0;'>
    🤖 CatBoost + LightGBM + XGBoost<br>
    ⚖️ Soft Voting (weights 1·1·1.5)<br>
    🔁 SMOTE oversampling<br>
    🎯 Threshold: <b>{THRESHOLD}</b><br>
    📊 Optimised for ROC-AUC
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🎯 Risk Levels**")
    st.markdown("""
    <div style='font-size:0.82rem;line-height:2.4;'>
    🔴 <b style='color:#ff5252'>HIGH</b> &nbsp;&nbsp; prob ≥ 0.70<br>
    🟠 <b style='color:#ff9100'>MEDIUM</b> prob 0.40–0.70<br>
    🟢 <b style='color:#69ff47'>LOW</b> &nbsp;&nbsp; prob &lt; 0.40
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#444d70;text-align:center;'>
    Churn Prediction v1.0 · Production Ready
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>📡 Customer Churn Prediction</h1>
    <p>CatBoost · LightGBM · XGBoost Ensemble &nbsp;|&nbsp; SMOTE &nbsp;|&nbsp; Production Ready</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Single Customer Prediction", "📂  Batch CSV Prediction"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("<div class='section-title'>👤 Demographics</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: gender     = st.selectbox("Gender",        ["Male", "Female"])
    with c2: senior     = st.selectbox("Senior Citizen",["No", "Yes"])
    with c3: partner    = st.selectbox("Partner",       ["Yes", "No"])
    with c4: dependents = st.selectbox("Dependents",    ["Yes", "No"])

    st.markdown("<div class='section-title'>🧾 Account</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: tenure          = st.slider("Tenure (months)", 0, 100, 12)
    with c2: monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 65.0, 0.5)
    with c3: total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0, 10.0)

    c1, c2, c3 = st.columns(3)
    with c1: contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c2: paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c3: payment  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown("<div class='section-title'>📶 Services</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        phone_service    = st.selectbox("Phone Service",    ["Yes", "No"])
        online_security  = st.selectbox("Online Security",  ["Yes", "No", "No internet service"])
    with c2:
        multiple_lines   = st.selectbox("Multiple Lines",   ["Yes", "No", "No phone service"])
        online_backup    = st.selectbox("Online Backup",    ["Yes", "No", "No internet service"])
    with c3:
        internet_service  = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        device_protection = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
    with c4:
        tech_support      = st.selectbox("Tech Support",    ["Yes", "No", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV",    ["Yes", "No", "No internet service"])

    c1, _, _ = st.columns(3)
    with c1: streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _, _ = st.columns([1, 2, 2])
    with col_btn:
        predict_btn = st.button("🔮  Predict Churn", use_container_width=True)

    if predict_btn:
        customer = {
            "gender":           gender,
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          partner,
            "Dependents":       dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service,
            "MultipleLines":    multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security,
            "OnlineBackup":     online_backup,
            "DeviceProtection": device_protection,
            "TechSupport":      tech_support,
            "StreamingTV":      streaming_tv,
            "StreamingMovies":  streaming_movies,
            "Contract":         contract,
            "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly_charges,
            "TotalCharges":     total_charges,
        }

        with st.spinner("Analysing customer profile..."):
            result = predict_single_row(customer)

        prob = result["probability"]
        pred = result["prediction"]
        risk = result["risk"]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns([1.2, 1, 1.2])

        # ── Verdict card ──────────────────────────────────────────────────────
        with r1:
            if pred == 1:
                st.markdown(f"""
                <div class='result-churn'>
                    <div class='result-label'>Prediction</div>
                    <div class='result-value churn-text'>⚠ CHURN</div>
                    <div class='result-prob'>This customer is likely to leave</div>
                    <br><span class='risk-{risk}'>{risk} RISK</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                    <div class='result-label'>Prediction</div>
                    <div class='result-value safe-text'>✓ SAFE</div>
                    <div class='result-prob'>This customer is likely to stay</div>
                    <br><span class='risk-{risk}'>{risk} RISK</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Gauge ─────────────────────────────────────────────────────────────
        with r2:
            st.plotly_chart(gauge_chart(prob), use_container_width=True,
                            config={"displayModeBar": False})

        # ── Key factors + probability box ─────────────────────────────────────
        with r3:
            st.markdown("**📋 Key Risk Factors Detected**")
            factors = []
            if contract         == "Month-to-month":        factors.append(("🔴", "Month-to-month contract"))
            if payment          == "Electronic check":       factors.append(("🔴", "Electronic check payment"))
            if tenure            < 12:                       factors.append(("🟠", "New customer (< 1 year)"))
            if internet_service == "Fiber optic":            factors.append(("🟠", "Fiber optic internet"))
            if online_security  == "No":                     factors.append(("🟡", "No online security"))
            if tech_support     == "No":                     factors.append(("🟡", "No tech support"))
            if senior           == "Yes":                    factors.append(("🟡", "Senior citizen"))
            if partner          == "No" and dependents=="No":factors.append(("🟡", "No family ties"))
            if not factors:
                factors.append(("🟢", "No major risk factors detected"))

            for icon, text in factors[:6]:
                st.markdown(
                    f"<div style='color:#c8cce8;font-size:0.87rem;"
                    f"padding:5px 0;border-bottom:1px solid #1e2240;'>"
                    f"{icon} &nbsp;{text}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            prob_color = "#ff5252" if pred == 1 else "#69ff47"
            st.markdown(f"""
            <div style='background:#1a1f3e;border:1px solid #2e3460;
                        border-radius:12px;padding:16px 20px;'>
                <div style='color:#6b75a8;font-size:0.72rem;
                            text-transform:uppercase;letter-spacing:1px;'>
                    Churn Probability
                </div>
                <div style='color:{prob_color};font-size:2.2rem;
                            font-weight:800;margin:4px 0;'>{prob:.1%}</div>
                <div style='color:#6b75a8;font-size:0.75rem;'>
                    Decision threshold: {THRESHOLD:.0%}
                    &nbsp;|&nbsp; {'Above ⚠' if prob>=THRESHOLD else 'Below ✓'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Customer summary
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#1a1f3e;border:1px solid #2e3460;
                        border-radius:12px;padding:16px 20px;
                        font-size:0.82rem;color:#8891c0;line-height:2.0;'>
                <b style='color:#7c8bcc;'>Customer Summary</b><br>
                📅 Tenure: <b style='color:#e8eaf6;'>{tenure} months</b><br>
                💰 Monthly: <b style='color:#e8eaf6;'>${monthly_charges:.2f}</b><br>
                📄 Contract: <b style='color:#e8eaf6;'>{contract}</b><br>
                🌐 Internet: <b style='color:#e8eaf6;'>{internet_service}</b>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH CSV PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("<div class='section-title'>📂 Upload Customer CSV</div>",
                unsafe_allow_html=True)

    with st.expander("📌 Required CSV Columns (click to expand)"):
        st.markdown("""
        Upload a CSV with these **19 raw columns** (same as your train.csv — no `id`, no `Churn` needed):

        | Column | Values |
        |---|---|
        | gender | Male / Female |
        | SeniorCitizen | 0 / 1 |
        | Partner | Yes / No |
        | Dependents | Yes / No |
        | tenure | 0–100 (months) |
        | PhoneService | Yes / No |
        | MultipleLines | Yes / No / No phone service |
        | InternetService | DSL / Fiber optic / No |
        | OnlineSecurity | Yes / No / No internet service |
        | OnlineBackup | Yes / No / No internet service |
        | DeviceProtection | Yes / No / No internet service |
        | TechSupport | Yes / No / No internet service |
        | StreamingTV | Yes / No / No internet service |
        | StreamingMovies | Yes / No / No internet service |
        | Contract | Month-to-month / One year / Two year |
        | PaperlessBilling | Yes / No |
        | PaymentMethod | Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic) |
        | MonthlyCharges | float |
        | TotalCharges | float |
        """)

    uploaded_file = st.file_uploader(
        "Drop CSV here or click to browse",
        type=["csv"],
        help="Upload CSV with customer data for batch churn predictions",
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            if "Churn" in df_input.columns:
                df_input = df_input.drop(columns=["Churn"])

            st.markdown(f"""
            <div style='background:#1a1f3e;border:1px solid #2e3460;
                        border-radius:10px;padding:10px 18px;margin:8px 0;
                        color:#8891c0;font-size:0.87rem;'>
                ✅ &nbsp;<b style='color:#00e5ff;'>{uploaded_file.name}</b>
                &nbsp;|&nbsp; <b style='color:#e8eaf6;'>{len(df_input):,}</b> rows
                &nbsp;|&nbsp; {df_input.shape[1]} columns loaded
            </div>
            """, unsafe_allow_html=True)

            with st.expander("👁 Preview — first 5 rows"):
                st.dataframe(df_input.head(), use_container_width=True)

            col_b, _, _ = st.columns([1, 2, 2])
            with col_b:
                run_btn = st.button("⚡  Run Batch Prediction", use_container_width=True)

            if run_btn:
                with st.spinner(f"Predicting {len(df_input):,} customers..."):
                    df_result = predict_dataframe(df_input)

                n_total  = len(df_result)
                n_churn  = int(df_result["Churn_Prediction"].sum())
                n_safe   = n_total - n_churn
                n_high   = int((df_result["Risk_Level"] == "HIGH").sum())
                n_medium = int((df_result["Risk_Level"] == "MEDIUM").sum())
                n_low    = int((df_result["Risk_Level"] == "LOW").sum())
                avg_prob = df_result["Churn_Probability"].mean()

                # ── METRICS ───────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Batch Prediction Summary")

                m1,m2,m3,m4,m5,m6 = st.columns(6)
                cards = [
                    (m1, "Total Customers", f"{n_total:,}",   ""),
                    (m2, "Will Churn 🔴",   f"{n_churn:,}",   f"{n_churn/n_total:.1%}"),
                    (m3, "Will Stay 🟢",    f"{n_safe:,}",    f"{n_safe/n_total:.1%}"),
                    (m4, "High Risk",        f"{n_high:,}",    f"{n_high/n_total:.1%}"),
                    (m5, "Medium Risk",      f"{n_medium:,}",  f"{n_medium/n_total:.1%}"),
                    (m6, "Avg Probability",  f"{avg_prob:.1%}", ""),
                ]
                for col, lbl, val, sub in cards:
                    with col:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='label'>{lbl}</div>
                            <div class='value'>{val}</div>
                            <div class='sub'>{sub}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # ── CHARTS ────────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                ch1, ch2, ch3 = st.columns(3)
                with ch1:
                    st.markdown("**🍩 Churn vs Stay**")
                    st.plotly_chart(donut_chart(n_churn, n_safe),
                                    use_container_width=True,
                                    config={"displayModeBar": False})
                with ch2:
                    st.markdown("**📊 Risk Distribution**")
                    st.plotly_chart(risk_bar_chart(df_result),
                                    use_container_width=True,
                                    config={"displayModeBar": False})
                with ch3:
                    st.markdown("**📈 Probability Distribution**")
                    st.plotly_chart(prob_histogram(df_result),
                                    use_container_width=True,
                                    config={"displayModeBar": False})

                # ── RESULTS TABLE ─────────────────────────────────────────────
                st.markdown("### 📋 Full Results Table")
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option("styler.render.max_elements", None)


                def color_risk(val):
                    c = {"HIGH": "#ff5252", "MEDIUM": "#ff9100", "LOW": "#69ff47"}
                    return f"color: {c.get(val,'white')}"

                def color_label(val):
                    return "color: #ff5252" if val == "Yes" else "color: #69ff47"
                
               


             

                # ── DOWNLOAD ──────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                # Full CSV
                full_csv = io.BytesIO()
                df_result.to_csv(full_csv, index=False)
                full_csv_bytes = full_csv.getvalue()

                # Churners only CSV
                churners_csv = (
                    df_result[df_result["Churn_Prediction"] == 1]
                    .to_csv(index=False).encode()
                )

                # High risk only CSV
                high_risk_csv = (
                    df_result[df_result["Risk_Level"] == "HIGH"]
                    .to_csv(index=False).encode()
                )

                dl1, dl2, dl3 = st.columns(3)
                with dl1:
                    st.download_button(
                        label     = "⬇️ Full Results CSV",
                        data      = full_csv_bytes,
                        file_name = "churn_predictions_all.csv",
                        mime      = "text/csv",
                        use_container_width = True,
                        on_click="ignore"
                    )
                with dl2:
                    st.download_button(
                        label     = "⬇️ Churners Only CSV",
                        data      = churners_csv,
                        file_name = "churners_only.csv",
                        mime      = "text/csv",
                        use_container_width = True,
                        on_click="ignore"
                    )
                with dl3:
                    st.download_button(
                        label     = "⬇️ High Risk Only CSV",
                        data      = high_risk_csv,
                        file_name = "high_risk_customers.csv",
                        mime      = "text/csv",
                        use_container_width = True,
                        on_click="ignore"
                    )

                st.markdown(f"""
                <div style='background:#1a1f3e;border:1px solid #2e3460;
                            border-radius:10px;padding:14px 20px;margin-top:10px;
                            font-size:0.83rem;color:#8891c0;'>
                    📁 <b>3 download options available</b> &nbsp;|&nbsp;
                    All CSVs include: <code>Churn_Probability</code>,
                    <code>Churn_Prediction</code>, <code>Churn_Label</code>,
                    <code>Risk_Level</code> &nbsp;|&nbsp;
                    Threshold used: <b>{THRESHOLD}</b>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your CSV has all required columns listed above.")