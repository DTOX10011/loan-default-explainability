"""
Streamlit dashboard for the Loan Default Risk Predictor.

Collects key borrower attributes in the sidebar, runs them through the
trained XGBoost model (`../models/xgb_model.pkl`), and explains the
resulting risk score with a SHAP waterfall plot so the prediction can be
justified feature-by-feature.

Run with:
    streamlit run app/streamlit_app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_model.pkl")
RISK_THRESHOLD = 0.5

EDUCATION_OPTIONS = [
    "Secondary___secondary_special",
    "Higher_education",
    "Incomplete_higher",
    "Lower_secondary",
    "Academic_degree",
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------
def build_feature_row(inputs, expected_columns):
    """Build a single-row DataFrame matching the model's expected features.

    NOTE: This demo reconstructs a feature vector from the handful of
    borrower inputs collected in the sidebar. The fully trained model
    expects the complete set of engineered columns produced by
    `notebooks/02_feature_engineering.ipynb` (including one-hot encoded
    categoricals). This app doesn't have access to the exact fitted
    encoders used at training time, so any expected column not covered
    by a sidebar input is left at a neutral default (0). For a
    production deployment, persist and reload the same encoders used
    during training instead of reconstructing them here.
    """
    row = pd.DataFrame(0, index=[0], columns=expected_columns, dtype=float)

    direct_fields = {
        "AMT_CREDIT": inputs["amt_credit"],
        "AMT_INCOME_TOTAL": inputs["amt_income_total"],
        "AMT_ANNUITY": inputs["amt_annuity"],
        "AMT_GOODS_PRICE": inputs["amt_goods_price"],
        "AGE_YEARS": inputs["age_years"],
        "YEARS_EMPLOYED": inputs["years_employed"],
        "EXT_SOURCE_1": inputs["ext_source_1"],
        "EXT_SOURCE_2": inputs["ext_source_2"],
        "EXT_SOURCE_3": inputs["ext_source_3"],
        "ANNUITY_INCOME_RATIO": inputs["annuity_income_ratio"],
        "CREDIT_INCOME_RATIO": inputs["credit_income_ratio"],
        "CREDIT_GOODS_RATIO": inputs["credit_goods_ratio"],
        "INCOME_PER_PERSON": inputs["income_per_person"],
        "LOAN_TERM_MONTHS": inputs["loan_term_months"],
        "EXT_SOURCE_MEAN": inputs["ext_source_mean"],
        "EXT_SOURCE_MIN": inputs["ext_source_min"],
        "EXT_SOURCE_STD": inputs["ext_source_std"],
    }
    for col, val in direct_fields.items():
        if col in row.columns:
            row.at[0, col] = val

    # One-hot encoded gender — model keeps CODE_GENDER_M (and CODE_GENDER_XNA,
    # left at 0 here since the sidebar never offers that option); "Female" is
    # the dropped baseline category, so it's correctly represented by leaving
    # CODE_GENDER_M at 0.
    if "CODE_GENDER_M" in row.columns:
        row.at[0, "CODE_GENDER_M"] = 1 if inputs["code_gender"] == "Male" else 0
    if "FLAG_OWN_CAR" in row.columns:
        row.at[0, "FLAG_OWN_CAR"] = 1 if inputs["flag_own_car"] == "Yes" else 0

    # One-hot encoded education level — column names use underscores in place
    # of spaces/slashes (e.g. NAME_EDUCATION_TYPE_Secondary___secondary_special).
    # "Academic_degree" is the dropped baseline category, so it has no matching
    # column and is correctly represented by leaving all dummies at 0.
    education_col = f"NAME_EDUCATION_TYPE_{inputs['name_education_type']}"
    if education_col in row.columns:
        row.at[0, education_col] = 1

    return row[expected_columns]


def compute_derived_features(amt_credit, amt_income_total, amt_annuity,
                              amt_goods_price, ext_source_1, ext_source_2,
                              ext_source_3, cnt_fam_members=1):
    annuity_income_ratio = amt_annuity / (amt_income_total + 1)
    credit_income_ratio = amt_credit / (amt_income_total + 1)
    credit_goods_ratio = amt_credit / (amt_goods_price + 1)
    income_per_person = amt_income_total / (cnt_fam_members + 1)
    loan_term_months = amt_credit / (amt_annuity + 1)

    ext_values = np.array([ext_source_1, ext_source_2, ext_source_3])
    ext_source_mean = ext_values.mean()
    ext_source_min = ext_values.min()
    ext_source_std = ext_values.std()

    return {
        "annuity_income_ratio": annuity_income_ratio,
        "credit_income_ratio": credit_income_ratio,
        "credit_goods_ratio": credit_goods_ratio,
        "income_per_person": income_per_person,
        "loan_term_months": loan_term_months,
        "ext_source_mean": ext_source_mean,
        "ext_source_min": ext_source_min,
        "ext_source_std": ext_source_std,
    }


# ---------------------------------------------------------------------------
# Sidebar — borrower inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Borrower Information")

amt_credit = st.sidebar.number_input(
    "Credit Amount ($)", min_value=0.0, value=500000.0, step=10000.0
)
amt_income_total = st.sidebar.number_input(
    "Total Income ($)", min_value=0.0, value=150000.0, step=5000.0
)
amt_annuity = st.sidebar.number_input(
    "Loan Annuity ($)", min_value=0.0, value=25000.0, step=1000.0
)
amt_goods_price = st.sidebar.number_input(
    "Goods Price ($)", min_value=0.0, value=450000.0, step=10000.0
)

age_years = st.sidebar.slider("Age (years)", min_value=18, max_value=70, value=35)
years_employed = st.sidebar.slider(
    "Years Employed", min_value=0, max_value=40, value=5
)

ext_source_1 = st.sidebar.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5)
ext_source_2 = st.sidebar.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5)
ext_source_3 = st.sidebar.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5)

code_gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
name_education_type = st.sidebar.selectbox("Education Level", EDUCATION_OPTIONS)
flag_own_car = st.sidebar.selectbox("Owns a Car?", ["Yes", "No"])

predict_clicked = st.sidebar.button("Predict", type="primary")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🏦 Loan Default Risk Predictor")
st.caption(
    "Enter borrower details in the sidebar and click **Predict** to score "
    "default risk and see a SHAP-based explanation of the result."
)

tab_predict, tab_performance = st.tabs(["Risk Prediction", "Model Performance"])

with tab_predict:
    if predict_clicked:
        try:
            model = load_model()
        except FileNotFoundError:
            st.error(
                f"Could not find a trained model at `{MODEL_PATH}`. "
                "Run `notebooks/03_modeling.ipynb` first to train and save it."
            )
            st.stop()

        derived = compute_derived_features(
            amt_credit, amt_income_total, amt_annuity, amt_goods_price,
            ext_source_1, ext_source_2, ext_source_3,
        )

        inputs = {
            "amt_credit": amt_credit,
            "amt_income_total": amt_income_total,
            "amt_annuity": amt_annuity,
            "amt_goods_price": amt_goods_price,
            "age_years": age_years,
            "years_employed": years_employed,
            "ext_source_1": ext_source_1,
            "ext_source_2": ext_source_2,
            "ext_source_3": ext_source_3,
            "code_gender": code_gender,
            "name_education_type": name_education_type,
            "flag_own_car": flag_own_car,
            **derived,
        }

        booster = model.get_booster()
        expected_columns = booster.feature_names
        feature_row = build_feature_row(inputs, expected_columns)

        risk_score = model.predict_proba(feature_row)[0, 1]
        is_high_risk = risk_score >= RISK_THRESHOLD

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Default Risk", f"{risk_score * 100:.1f}%")

            if is_high_risk:
                st.error(
                    f"⚠️ **High Risk** — predicted default probability "
                    f"({risk_score * 100:.1f}%) is at or above the "
                    f"{RISK_THRESHOLD * 100:.0f}% decision threshold."
                )
            else:
                st.success(
                    f"✅ **Low Risk** — predicted default probability "
                    f"({risk_score * 100:.1f}%) is below the "
                    f"{RISK_THRESHOLD * 100:.0f}% decision threshold."
                )

        with col2:
            st.subheader("Why this score? (SHAP Explanation)")
            explainer = load_explainer(model)
            shap_values = explainer(feature_row)

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Fill in the borrower details in the sidebar, then click **Predict**.")

with tab_performance:
    st.subheader("Model Performance")
    st.markdown(
        "The following metrics summarize the two candidate models trained in "
        "`notebooks/03_modeling.ipynb` on a held-out 20% test split "
        "(SMOTE applied to the training data only). Update these figures once "
        "the pipeline has been run end-to-end on the full dataset."
    )

    perf_df = pd.DataFrame(
        {
            "Model": ["XGBoost", "LightGBM"],
            "AUC-ROC": ["0.7317", "0.7311"],
            "Precision (Default)": ["TBD", "TBD"],
            "Recall (Default)": ["TBD", "TBD"],
            "F1 (Default)": ["TBD", "TBD"],
        }
    )
    st.table(perf_df)

    st.markdown(
        """
        **Notes**
        - Class imbalance was handled with SMOTE applied only to the training split.
        - `EXT_SOURCE_1/2/3` (external credit bureau scores) are consistently the
          strongest predictors of default risk — see `notebooks/04_explainability.ipynb`.
        - The deployed model here is XGBoost; see the modeling notebook for the
          full XGBoost vs. LightGBM comparison (ROC curves, PR curves, and
          feature importance).
        """
    )
