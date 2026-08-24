# 🏦 Loan Default Prediction with Explainability Dashboard

An end-to-end machine learning pipeline that predicts loan default risk using the Home Credit dataset, with SHAP-based explainability for every prediction — addressing real-world regulatory interpretability requirements.

## 📋 Overview

This project trains a gradient-boosted classifier to predict the probability that a borrower will default on a loan, then wraps every prediction in a SHAP-based explanation layer so that individual decisions can be inspected and justified. The result is served through an interactive Streamlit dashboard, giving both a global view of what drives default risk across the portfolio and a local, per-applicant breakdown of "why this score."

## 🎯 Problem Statement

Banks and fintechs need to predict which borrowers will default — but regulators increasingly require *why* a decision was made, not just *what* the decision is. Black-box models that perform well but can't be explained create compliance risk (e.g. fair lending / adverse action requirements) and erode trust with loan officers and applicants alike. This project builds a production-style ML pipeline that combines high predictive performance with full explainability, so every prediction can be traced back to the features that drove it.

## 🗂️ Project Structure

```
loan-default-explainability/
├── app/                  # Streamlit explainability dashboard
├── data/
│   ├── raw/              # Original, immutable Home Credit data (gitignored)
│   └── processed/        # Cleaned / feature-engineered datasets (gitignored)
├── models/                # Trained model artifacts (gitignored)
├── notebooks/             # Exploratory analysis & experimentation
├── src/                   # Reusable pipeline code (data prep, training, explainability)
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

## 🛠️ Tech Stack

| Category         | Tools                                  |
|-------------------|-----------------------------------------|
| Data handling     | pandas, numpy                          |
| Modeling          | scikit-learn, XGBoost, LightGBM        |
| Class imbalance   | imbalanced-learn                       |
| Explainability    | SHAP, LIME                             |
| Visualization     | matplotlib, seaborn, plotly            |
| Dashboard         | Streamlit                              |
| Utilities         | joblib, scipy                          |

## 📊 Dataset

This project uses the **[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)** dataset, which contains anonymized loan application data including applicant demographics, previous credit history, and repayment behavior. The target variable indicates whether an applicant had payment difficulties (defaulted) on their loan.

Raw and processed data files are not committed to this repository (see `.gitignore`) — download the dataset separately and place it under `data/raw/` before running the pipeline.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/DTOX10011/loan-default-explainability.git
cd loan-default-explainability

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place the Home Credit dataset under data/raw/, then run the pipeline
# (see notebooks/ and src/ for data prep, training, and evaluation steps)

# Launch the explainability dashboard
streamlit run app/app.py
```

## 📈 Results

*TBD — model performance metrics (ROC-AUC, precision/recall, F1) and comparison across candidate models will be added here once training is complete.*

## 🔍 Explainability Approach

- **Global explainability:** SHAP summary and feature importance plots reveal which features drive default risk across the entire portfolio.
- **Local explainability:** For each individual prediction, SHAP force/waterfall plots break down exactly how each feature pushed the model's output toward or away from "default," giving a human-readable justification for the score.
- **LIME** is used as a complementary, model-agnostic sanity check for local explanations.
- All explanations are surfaced in the Streamlit dashboard so loan officers and reviewers can inspect the reasoning behind any prediction without needing to read model internals.

## ✍️ Author

**DTOX10011**
GitHub: [@DTOX10011](https://github.com/DTOX10011)
