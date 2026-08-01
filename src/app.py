from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# CONSTANTS
# ==========================================================

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.pkl"

PRIMARY_COLOR = "#6C8CFF"
DANGER_COLOR = "#FF6B6B"
SUCCESS_COLOR = "#3DD68C"
WARNING_COLOR = "#F5B942"
BG_COLOR = "#0E1420"
CARD_COLOR = "#161E2E"
CARD_BORDER = "#232D42"
TEXT_COLOR = "#E8EEF7"
TEXT_MUTED = "#8B96AA"

DEVELOPER = {
    "name": "Talha Siddiqui",
    "role": "Software Engineering Student · AI/ML & Data Science",
    "bio": (
        "Software Engineering student building AI/ML and automation projects. "
        "Data Science · Machine Learning · Deep Learning · Solo builder."
    ),
    "github": "https://github.com/talha-siddiqui137",
    "repo": "https://github.com/talha-siddiqui137/Customer-Churn-Prediction",
    "linkedin": "https://www.linkedin.com/in/talha-siddiqui137/",
    "email": "talha03182301690@gmail.com",
    "stack": ["Python", "scikit-learn", "Pandas", "NumPy", "Matplotlib", "Streamlit", "Plotly"],
}

FEATURE_DEFAULTS: dict[str, Any] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 32,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 2300.00,
}

# A realistic "high risk" and "low risk" sample customer, used by the
# one-click autofill buttons so reviewers can try the app instantly.
SAMPLE_HIGH_RISK: dict[str, Any] = {
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.70,
    "TotalCharges": 191.40,
}

SAMPLE_LOW_RISK: dict[str, Any] = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 65,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.20,
    "TotalCharges": 2938.00,
}


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction | Talha Siddiqui",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        .stApp {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #131A29 0%, #0A0F19 100%);
            border-right: 1px solid {CARD_BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT_COLOR} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: {CARD_BORDER};
        }}

        h1, h2, h3, h4, p, span, label, .stMarkdown {{
            color: {TEXT_COLOR};
            font-family: "Segoe UI", "Inter", sans-serif;
            letter-spacing: -0.3px;
        }}

        /* Native input widgets: force a consistent dark surface + readable text */
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {{
            background-color: #1C2536 !important;
            color: {TEXT_COLOR} !important;
            border: 1px solid {CARD_BORDER} !important;
            border-radius: 8px !important;
        }}
        ul[role="listbox"] {{
            background-color: #1C2536 !important;
        }}

        .app-card {{
            background: {CARD_COLOR};
            padding: 22px 24px;
            border-radius: 14px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
            border: 1px solid {CARD_BORDER};
            margin-bottom: 18px;
        }}
        .app-card, .app-card * {{
            color: {TEXT_COLOR} !important;
        }}

        .badge {{
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .badge-danger {{ background: rgba(255,107,107,0.16); color: {DANGER_COLOR} !important; }}
        .badge-success {{ background: rgba(61,214,140,0.16); color: {SUCCESS_COLOR} !important; }}
        .badge-warning {{ background: rgba(245,185,66,0.18); color: {WARNING_COLOR} !important; }}

        div.stButton > button {{
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: all 0.15s ease-in-out;
            background-color: #1C2536;
            color: {TEXT_COLOR};
            border: 1px solid {CARD_BORDER};
        }}
        div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 14px rgba(108,140,255,0.25);
            border-color: {PRIMARY_COLOR};
            color: {PRIMARY_COLOR};
        }}
        div.stButton > button[kind="primary"] {{
            background: {PRIMARY_COLOR};
            border: none;
            color: #0E1420;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: #8AA5FF;
            color: #0E1420;
        }}

        .step-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(108,140,255,0.14);
            color: {PRIMARY_COLOR} !important;
            font-weight: 600;
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }}

        .chip {{
            display: inline-block;
            background: #1C2536;
            border: 1px solid {CARD_BORDER};
            color: {TEXT_COLOR} !important;
            padding: 5px 12px;
            border-radius: 999px;
            font-size: 0.82rem;
            margin: 3px 4px 3px 0;
        }}

        .avatar-circle {{
            width: 64px;
            height: 64px;
            border-radius: 50%;
            background: linear-gradient(135deg, {PRIMARY_COLOR}, #3D5AFE);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0E1420 !important;
            font-weight: 700;
            font-size: 1.4rem;
            margin-bottom: 10px;
        }}

        .link-button {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            background: #1C2536;
            border: 1px solid {CARD_BORDER};
            color: {TEXT_COLOR} !important;
            text-decoration: none !important;
            font-weight: 600;
            font-size: 0.88rem;
            margin: 4px 8px 4px 0;
        }}
        .link-button:hover {{
            border-color: {PRIMARY_COLOR};
            color: {PRIMARY_COLOR} !important;
        }}

        .footer-text {{
            text-align: center;
            color: {TEXT_MUTED};
            font-size: 0.85rem;
            padding-top: 30px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# MODEL LOADING
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained Pipeline (preprocessing + classifier). Cached across reruns."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


# ==========================================================
# HELPERS
# ==========================================================

def build_input_dataframe(values: dict[str, Any]) -> pd.DataFrame:
    """Assemble the raw-column DataFrame expected by the pipeline. No manual encoding here."""
    row = {**FEATURE_DEFAULTS, **values}
    return pd.DataFrame([row])


def make_gauge(probability: float) -> go.Figure:
    """A confidence/risk gauge for churn probability (0-100)."""
    color = DANGER_COLOR if probability >= 50 else SUCCESS_COLOR
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 36, "color": TEXT_COLOR}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": CARD_COLOR,
                "bordercolor": CARD_BORDER,
                "steps": [
                    {"range": [0, 40], "color": "rgba(61,214,140,0.18)"},
                    {"range": [40, 70], "color": "rgba(245,185,66,0.18)"},
                    {"range": [70, 100], "color": "rgba(255,107,107,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR},
    )
    return fig


def make_donut(yes_pct: float, no_pct: float) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Will Churn", "Will Stay"],
            values=[yes_pct, no_pct],
            hole=0.6,
            marker=dict(colors=[DANGER_COLOR, SUCCESS_COLOR], line=dict(color=CARD_COLOR, width=2)),
            textinfo="label+percent",
            textfont={"color": "#0E1420", "size": 13},
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT_COLOR},
    )
    return fig


def get_risk_level(probability: float) -> tuple[str, str]:
    """Return (label, badge_css_class) for a churn probability."""
    if probability >= 70:
        return "High Risk", "badge-danger"
    if probability >= 40:
        return "Medium Risk", "badge-warning"
    return "Low Risk", "badge-success"


def extract_top_features(model, top_n: int = 8) -> pd.DataFrame | None:
    """
    Best-effort interpretability: if the final estimator exposes coef_
    (e.g. Logistic Regression / linear SVM) and the preprocessing step
    exposes feature names, return the top contributing features.
    Returns None quietly if unavailable — this must never crash the app.
    """
    try:
        classifier = model.named_steps.get("classifier") or list(model.named_steps.values())[-1]
        preprocessor = model.named_steps.get("preprocessor") or list(model.named_steps.values())[0]

        if not hasattr(classifier, "coef_"):
            return None

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame(
            {"feature": feature_names, "coefficient": coefficients}
        )
        importance_df["abs_coefficient"] = importance_df["coefficient"].abs()
        importance_df = importance_df.sort_values("abs_coefficient", ascending=False).head(top_n)
        importance_df["direction"] = importance_df["coefficient"].apply(
            lambda c: "Increases churn risk" if c > 0 else "Decreases churn risk"
        )
        return importance_df[["feature", "coefficient", "direction"]]
    except Exception:
        return None


def generate_report(prediction: str, confidence: float, yes_pct: float, no_pct: float, inputs: pd.DataFrame) -> str:
    lines = [
        "CUSTOMER CHURN PREDICTION REPORT",
        "=" * 40,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Prediction        : {prediction}",
        f"Confidence        : {confidence:.2f}%",
        f"Probability Churn : {yes_pct:.2f}%",
        f"Probability Stay  : {no_pct:.2f}%",
        "",
        "CUSTOMER DETAILS",
        "-" * 40,
    ]
    row = inputs.iloc[0].to_dict()
    for key, value in row.items():
        lines.append(f"{key:20s}: {value}")
    return "\n".join(lines)


@dataclass
class HistoryEntry:
    timestamp: str
    prediction: str
    churn_probability: float
    inputs: dict = field(default_factory=dict)


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "form_values" not in st.session_state:
        st.session_state.form_values = dict(FEATURE_DEFAULTS)


def apply_sample(sample: dict[str, Any]) -> None:
    st.session_state.form_values = dict(sample)


def reset_form() -> None:
    st.session_state.form_values = dict(FEATURE_DEFAULTS)


# ==========================================================
# APP BOOTSTRAP
# ==========================================================

inject_css()
init_session_state()
model = load_model()

if model is None:
    st.error(
        "**Model not found or failed to load.**\n\n"
        f"Expected a trained pipeline at `{MODEL_PATH}`. "
        "Run `tunning.py` (or `evaluate_models.py`) first to generate it."
    )
    st.stop()

fv = st.session_state.form_values


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.title("📊 Churn Predictor")
    st.success("✅ Model loaded")
    st.markdown("---")

    st.subheader("📁 Dataset")
    st.write("IBM Telco Customer Churn")

    st.subheader("🤖 Model")
    st.write("Logistic Regression (tuned via RandomizedSearchCV)")
    st.caption("Accuracy ≈ 80% · F1 ≈ 0.60 · ROC AUC ≈ 0.84")

    st.markdown("---")
    st.subheader("🎯 Purpose")
    st.write("Predicts whether a telecom customer is likely to leave the company, and why.")

    st.markdown("---")
    st.subheader("🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/talha-siddiqui137/Customer-Churn-Prediction)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/talha-siddiqui137/)")

    st.markdown("---")
    st.caption(f"Developed by Talha Siddiqui · {datetime.now().year}")


# ==========================================================
# HEADER
# ==========================================================

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown(
    "An end-to-end ML application that predicts telecom customer churn "
    "and explains the drivers behind each prediction."
)

tab_predict, tab_model, tab_about = st.tabs(["🔍 Predict", "🧠 Model & Dataset", "ℹ️ About"])


# ==========================================================
# TAB 1 — PREDICT
# ==========================================================

with tab_predict:

    # ---- Quick actions ----
    st.markdown('<span class="step-pill">STEP 1 · Fill Information</span>', unsafe_allow_html=True)
    quick_col1, quick_col2, quick_col3, _ = st.columns([1, 1, 1, 3])
    with quick_col1:
        if st.button("⚡ Sample: High Risk"):
            apply_sample(SAMPLE_HIGH_RISK)
            st.rerun()
    with quick_col2:
        if st.button("⚡ Sample: Low Risk"):
            apply_sample(SAMPLE_LOW_RISK)
            st.rerun()
    with quick_col3:
        if st.button("🔄 Reset Form"):
            reset_form()
            st.rerun()

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 👤 Customer Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(fv["gender"]))
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1], index=[0, 1].index(fv["SeniorCitizen"]),
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        partner = st.selectbox("Partner", ["Yes", "No"], index=["Yes", "No"].index(fv["Partner"]))
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=["Yes", "No"].index(fv["Dependents"]))

    with col2:
        tenure = st.number_input(
            "Tenure (Months)", min_value=0, max_value=100, value=int(fv["tenure"]), step=1,
            help="Number of months the customer has stayed with the company.",
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=float(fv["MonthlyCharges"]), step=0.01
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(fv["TotalCharges"]), step=0.01,
            help="Defaults to a sensible value if left unchanged.",
        )

    with col3:
        contract_options = ["Month-to-month", "One year", "Two year"]
        contract = st.selectbox("Contract", contract_options, index=contract_options.index(fv["Contract"]))
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(fv["PaperlessBilling"])
        )
        payment_options = [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
        ]
        payment_method = st.selectbox("Payment Method", payment_options, index=payment_options.index(fv["PaymentMethod"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 📞 Phone & Internet Services")
    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(fv["PhoneService"]))
        ml_options = ["No", "Yes", "No phone service"]
        multiple_lines = st.selectbox("Multiple Lines", ml_options, index=ml_options.index(fv["MultipleLines"]))

    with col2:
        internet_options = ["DSL", "Fiber optic", "No"]
        internet_service = st.selectbox("Internet Service", internet_options, index=internet_options.index(fv["InternetService"]))
        addon_options = ["Yes", "No", "No internet service"]
        online_security = st.selectbox("Online Security", addon_options, index=addon_options.index(fv["OnlineSecurity"]))
        online_backup = st.selectbox("Online Backup", addon_options, index=addon_options.index(fv["OnlineBackup"]))

    with col3:
        device_protection = st.selectbox("Device Protection", addon_options, index=addon_options.index(fv["DeviceProtection"]))
        tech_support = st.selectbox("Tech Support", addon_options, index=addon_options.index(fv["TechSupport"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 🎬 Streaming Services")
    col1, col2 = st.columns(2)
    with col1:
        streaming_tv = st.selectbox("Streaming TV", addon_options, index=addon_options.index(fv["StreamingTV"]))
    with col2:
        streaming_movies = st.selectbox("Streaming Movies", addon_options, index=addon_options.index(fv["StreamingMovies"]))
    st.markdown("</div>", unsafe_allow_html=True)

    # Persist current selections so sample/reset buttons stay in sync
    st.session_state.form_values = {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }

    input_df = build_input_dataframe(st.session_state.form_values)

    st.markdown('<span class="step-pill">STEP 2 · Review Summary</span>', unsafe_allow_html=True)
    with st.expander("📋 Review Customer Summary", expanded=False):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown('<span class="step-pill">STEP 3 · Predict</span>', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Customer Churn", type="primary")

    if predict_clicked:
        try:
            with st.spinner("Running prediction..."):
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                classes = list(model.classes_)
                no_probability = probability[classes.index("No")] * 100
                yes_probability = probability[classes.index("Yes")] * 100
                confidence = max(yes_probability, no_probability)
        except Exception as exc:
            st.error(
                "Something went wrong while generating the prediction. "
                "This usually means an input value doesn't match what the model was trained on."
            )
            st.caption(f"Technical detail: {exc}")
            st.stop()

        st.session_state.history.append(
            HistoryEntry(
                timestamp=datetime.now().strftime("%H:%M:%S"),
                prediction=str(prediction),
                churn_probability=round(yes_probability, 2),
                inputs=dict(st.session_state.form_values),
            )
        )

        st.divider()
        st.markdown('<span class="step-pill">STEP 4 · Analyze Result</span>', unsafe_allow_html=True)

        risk_label, risk_class = get_risk_level(yes_probability)

        res_col1, res_col2, res_col3 = st.columns([1.1, 1, 1])

        with res_col1:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            if prediction == "Yes":
                st.markdown(
                    f'<span class="badge badge-danger">⚠️ Likely to Churn</span>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<span class="badge badge-success">✅ Likely to Stay</span>', unsafe_allow_html=True
                )
            st.write("")
            st.markdown(f'<span class="badge {risk_class}">{risk_label}</span>', unsafe_allow_html=True)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.plotly_chart(make_gauge(yes_probability), use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("**Probability Breakdown**")
            st.plotly_chart(
                make_donut(yes_probability, no_probability),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col3:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("**Business Recommendation**")
            if prediction == "Yes":
                st.warning(
                    "- Offer a loyalty discount or retention plan\n"
                    "- Reach out proactively before renewal\n"
                    "- Review recent service quality issues\n"
                    "- Suggest a longer-term contract\n"
                    "- Prioritize for premium support"
                )
            else:
                st.success(
                    "- Maintain current service quality\n"
                    "- Offer relevant upgrades or add-ons\n"
                    "- Promote long-term/loyalty plans\n"
                    "- Continue standard engagement"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ---- Interpretability ----
        importance_df = extract_top_features(model)
        if importance_df is not None:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("**Top Factors Influencing This Prediction**")
            st.caption("Based on the model's learned coefficients — positive values push toward churn, negative values push toward retention.")
            st.dataframe(importance_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<span class="step-pill">STEP 5 · Download Report</span>', unsafe_allow_html=True)
        report_text = generate_report(str(prediction), confidence, yes_probability, no_probability, input_df)
        st.download_button(
            "📄 Download Prediction Report",
            data=report_text,
            file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    # ---- Prediction history ----
    if st.session_state.history:
        st.divider()
        with st.expander(f"🕓 Prediction History ({len(st.session_state.history)})", expanded=False):
            history_df = pd.DataFrame(
                [
                    {"Time": h.timestamp, "Prediction": h.prediction, "Churn Probability (%)": h.churn_probability}
                    for h in reversed(st.session_state.history)
                ]
            )
            st.dataframe(history_df, use_container_width=True, hide_index=True)


# ==========================================================
# TAB 2 — MODEL & DATASET INFO
# ==========================================================

with tab_model:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 📁 Dataset")
        st.write(
            "**IBM Telco Customer Churn Dataset** — customer demographics, "
            "account information, and subscribed services for a telecom provider, "
            "with a binary churn label."
        )
        st.markdown(
            "- ~7,000 customer records\n"
            "- 19 input features + target\n"
            "- Target: `Churn` (Yes / No)"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Model")
        st.write("**Logistic Regression**, tuned with `RandomizedSearchCV` (5-fold CV, F1-optimized).")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", "≈80%")
        m2.metric("F1 Score", "≈0.60")
        m3.metric("ROC AUC", "≈0.84")
        st.caption("Compared against Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, SVM, and KNN.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("#### 🧭 Pipeline")
    st.markdown(
        "`Raw Data` → `Cleaning` → `Feature Engineering` → `ColumnTransformer "
        "(impute + scale numeric, impute + one-hot categorical)` → `Model Training` → "
        "`Evaluation` → `Hyperparameter Tuning` → `Saved Pipeline` → `Streamlit App`"
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# TAB 3 — ABOUT
# ==========================================================

with tab_about:
    col1, col2 = st.columns([1.1, 1.4])

    with col1:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        initials = "".join([part[0] for part in DEVELOPER["name"].split()[:2]]).upper()
        st.markdown(f'<div class="avatar-circle">{initials}</div>', unsafe_allow_html=True)
        st.markdown(f"#### {DEVELOPER['name']}")
        st.caption(DEVELOPER["role"])
        st.write(DEVELOPER["bio"])
        st.markdown(
            f'<a class="link-button" href="{DEVELOPER["github"]}" target="_blank">🐙 GitHub</a>'
            f'<a class="link-button" href="{DEVELOPER["linkedin"]}" target="_blank">💼 LinkedIn</a>'
            f'<a class="link-button" href="mailto:{DEVELOPER["email"]}">✉️ Email</a>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Stack Used In This Project")
        st.markdown(
            "".join(f'<span class="chip">{tech}</span>' for tech in DEVELOPER["stack"]),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ About This Project")
        st.write(
            "This application is an end-to-end machine learning project that predicts "
            "whether a telecom customer is likely to churn, built to demonstrate production-style "
            "ML engineering practices: modular pipelines, proper preprocessing encapsulation, "
            "model comparison, hyperparameter tuning, and a deployable interface."
        )
        st.markdown(
            "**What this project demonstrates:**\n"
            "- Clean separation of concerns across preprocessing, training, evaluation, and tuning\n"
            "- A single sklearn `Pipeline` that bundles preprocessing with the classifier — no train/serve skew\n"
            "- Model comparison across 7 algorithms, ranked by F1, ROC AUC, and Accuracy\n"
            "- Hyperparameter tuning via `RandomizedSearchCV` with 5-fold cross-validation\n"
            "- A production-style Streamlit interface with interpretability and error handling"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("#### 🔗 Project Repository")
        st.markdown(
            f'<a class="link-button" href="{DEVELOPER["repo"]}" target="_blank">🐙 View on GitHub</a>',
            unsafe_allow_html=True,
        )
        st.caption("Star the repo if you find it useful — feedback and pull requests are welcome.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="footer-text">Built with Streamlit · Customer Churn Prediction © ' + str(datetime.now().year) + '</div>', unsafe_allow_html=True)