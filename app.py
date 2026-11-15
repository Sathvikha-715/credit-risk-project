import json
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ · Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark fintech theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0f2040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(37,99,235,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #60a5fa, #93c5fd, #e2e8f0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4);
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    margin-top: 1rem;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e293b;
}

/* ── Card wrapper ── */
.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}

/* ── Inputs ── */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #0f172a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}

/* ── Alert boxes ── */
.stSuccess, .stWarning, .stError {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
}

/* ── Chart backgrounds ── */
[data-testid="stPlotlyChart"], .stPyplot > div {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL, CONFIG, EXPLAINER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("models/credit_model.pkl", "rb"))


@st.cache_resource
def load_config():
    try:
        return json.load(open("models/config.json"))
    except Exception:
        return {"threshold": 0.5, "cost_fn": 5, "cost_fp": 1, "metrics_at_tuned": {}}


@st.cache_resource
def load_explainer(_model):
    # The XGBoost part of the ensemble, used to compute TreeSHAP contributions
    return _model.named_steps["classifier"].named_estimators_["xgb"].get_booster()


model = load_model()
config = load_config()
THRESHOLD = float(config["threshold"])
REVIEW_LOW = THRESHOLD * 0.5   # between this and the threshold = manual review

explainer_error = None
try:
    explainer = load_explainer(model)
except Exception as e:
    explainer = None
    explainer_error = str(e)


def pretty(name: str) -> str:
    return name.split("__", 1)[-1].replace("_", " ")


def top_factors(input_df, n=6):
    """Top features pushing THIS applicant's risk up or down (TreeSHAP values)."""
    pre = model.named_steps["preprocessor"]
    xt = pre.transform(input_df)
    if hasattr(xt, "toarray"):
        xt = xt.toarray()
    xt = np.asarray(xt, dtype=float)
    contribs = explainer.predict(xgboost.DMatrix(xt), pred_contribs=True)[0][:-1]
    names = pre.get_feature_names_out()
    order = np.argsort(np.abs(contribs))[::-1][:n]
    return [(pretty(names[i]), float(contribs[i])) for i in order]


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-sub">Ensemble AI · RF + XGBoost</p>
    <h1 class="hero-title">CreditIQ</h1>
    <p style="color:#94a3b8; margin:0; font-size:1rem;">
        Credit risk scoring with per-applicant explanations.
    </p>
    <span class="hero-badge">⚡ Real-time inference</span>
    <span class="hero-badge" style="margin-left:0.5rem;">🎯 Cost-tuned threshold</span>
    <span class="hero-badge" style="margin-left:0.5rem;">📊 SHAP explanations</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT FORM
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Applicant Profile</div>', unsafe_allow_html=True)

with st.container():
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Financial**")
        income   = st.number_input("Annual Income ($)", min_value=0, value=45000, step=1000)
        loan     = st.number_input("Loan Amount ($)",   min_value=0, value=12000, step=500)
        interest = st.number_input("Interest Rate (%)", min_value=0.0, value=13.5, step=0.1, format="%.1f")

    with col_b:
        st.markdown("**Personal**")
        age     = st.number_input("Age", min_value=18, max_value=100, value=25, step=1)
        home    = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        emp     = st.number_input("Employment (years)", min_value=0, value=4, step=1)
        history = st.number_input("Credit History (years)", min_value=0, value=6, step=1)

    with col_c:
        st.markdown("**Loan Details**")
        grade   = st.selectbox("Loan Grade",  ["A", "B", "C", "D", "E", "F", "G"])
        intent  = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "PERSONAL",
                                               "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        default = st.selectbox("Previous Default", ["No", "Yes"])

st.markdown("")
predict_btn = st.button("🚀 Run Risk Assessment")

# ─────────────────────────────────────────────
#  PREDICTION + RESULTS
# ─────────────────────────────────────────────
if predict_btn:

    loan_to_income = loan / income if income > 0 else 0

    input_df = pd.DataFrame({
        "person_age":                   [age],
        "person_income":                [income],
        "person_home_ownership":        [home],
        "person_emp_length":            [emp],
        "loan_intent":                  [intent],
        "loan_grade":                   [grade],
        "loan_amnt":                    [loan],
        "loan_int_rate":                [interest],
        "loan_percent_income":          [loan_to_income],
        "cb_person_default_on_file":    ["Y" if default == "Yes" else "N"],
        "cb_person_cred_hist_length":   [history],
    })

    prob = model.predict_proba(input_df)[0][1]
    pct  = round(prob * 100, 2)

    if prob < REVIEW_LOW:
        risk, color_hex, risk_emoji = "Low",    "#22c55e", "🟢"
    elif prob < THRESHOLD:
        risk, color_hex, risk_emoji = "Medium", "#f59e0b", "🟡"
    else:
        risk, color_hex, risk_emoji = "High",   "#ef4444", "🔴"

    st.markdown("---")
    st.markdown('<div class="section-label">📊 Risk Summary</div>', unsafe_allow_html=True)

    # ── KPI row ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Default Risk Score", f"{pct}%")
    m2.metric("Risk Level",         f"{risk_emoji} {risk}")
    m3.metric("Loan-to-Income",     f"{round(loan_to_income * 100, 1)}%")
    m4.metric("Decision Threshold", f"{round(THRESHOLD * 100, 1)}%")

    st.markdown("")

    # ── Charts row ──
    ch1, ch2 = st.columns([1, 1])

    # — Risk meter with the decision threshold —
    with ch1:
        st.markdown("##### Risk Meter")

        fig, ax = plt.subplots(figsize=(5, 2.2))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        ax.barh([""], [1],    color="#1e293b", height=0.45, zorder=1)
        ax.barh([""], [prob], color=color_hex, height=0.45, zorder=2, linewidth=0)

        ax.axvline(THRESHOLD, color="#e2e8f0", linewidth=2, linestyle="--", zorder=3)
        ax.text(THRESHOLD, -0.42, "decision threshold", ha="center", va="top",
                color="#94a3b8", fontsize=7.5)

        ax.text(prob, 0, f" {pct}%", va="center", ha="left" if prob < 0.85 else "right",
                color="#fff", fontsize=11, fontweight="700", zorder=4)

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # — Real explanation (TreeSHAP values from the XGBoost part) —
    with ch2:
        st.markdown("##### Why this score? (SHAP)")

        if explainer is None:
            st.info(f"Explanations unavailable: {explainer_error}")
        else:
            try:
                factors = top_factors(input_df, n=6)
                labels = [f[0] for f in factors][::-1]
                values = [f[1] for f in factors][::-1]
                bar_colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

                fig2, ax2 = plt.subplots(figsize=(5, 2.6))
                fig2.patch.set_facecolor("#0f172a")
                ax2.set_facecolor("#0f172a")

                ax2.barh(labels, values, color=bar_colors, height=0.55, zorder=2)
                ax2.axvline(0, color="#475569", linewidth=1)
                ax2.xaxis.set_tick_params(labelsize=7, colors="#475569")
                ax2.yaxis.set_tick_params(labelsize=8, colors="#94a3b8")
                ax2.set_xlabel("← lowers risk        raises risk →",
                               color="#64748b", fontsize=7.5)

                for spine in ["top", "right"]:
                    ax2.spines[spine].set_visible(False)
                for spine in ["bottom", "left"]:
                    ax2.spines[spine].set_color("#1e293b")

                plt.tight_layout(pad=0.4)
                st.pyplot(fig2, use_container_width=True)
                plt.close()
                st.caption("Computed from the XGBoost part of the ensemble. "
                           "Red raises this applicant's risk, green lowers it.")
            except Exception as e:
                st.info(f"Could not compute explanation: {e}")

    # ── Decision banner ──
    st.markdown("---")
    st.markdown('<div class="section-label">🧾 Decision</div>', unsafe_allow_html=True)

    if risk == "Low":
        st.success(
            f"✅ **APPROVED** — Risk score {pct}% is well below the {round(THRESHOLD*100,1)}% threshold. "
            "Loan is suitable for straight-through processing."
        )
    elif risk == "Medium":
        st.warning(
            f"⚠️ **MANUAL REVIEW** — Risk score {pct}% is close to the {round(THRESHOLD*100,1)}% threshold. "
            "Consider requesting additional documentation or collateral."
        )
    else:
        st.error(
            f"❌ **DECLINED** — Risk score {pct}% is at or above the {round(THRESHOLD*100,1)}% threshold. "
            "Advise applicant to reduce loan amount or improve credit profile."
        )

    # ── Model notes ──
    with st.expander("🔍 Model Notes"):
        st.markdown(f"""
| Field | Value |
|---|---|
| Model | Ensemble (Random Forest + XGBoost, soft voting) |
| Risk score | `{pct}%` |
| Decision threshold | `{round(THRESHOLD*100,1)}%`, chosen to minimise cost where a missed default costs {config.get("cost_fn", 5)}x a wrongly rejected loan (assumed costs) |
| Risk bands | Low < {round(REVIEW_LOW*100,1)}% · Medium {round(REVIEW_LOW*100,1)}–{round(THRESHOLD*100,1)}% · High ≥ {round(THRESHOLD*100,1)}% |
| Loan-to-Income | `{round(loan_to_income*100,1)}%` |
| Grade | `{grade}` · Intent: `{intent}` |

> _Scores are model outputs only. Trained on a public dataset for a portfolio project. Not for real lending decisions._
        """)