# Statistical Learning Approach for Employee Attrition

A machine learning project that predicts employee attrition using the IBM HR Analytics dataset. Three classifiers are trained and evaluated — Logistic Regression, Random Forest, and XGBoost — with SHAP-based explainability and cross-validation for model stability assessment.

## Dataset

**Source:** IBM HR Analytics Employee Attrition & Performance  
**File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`  
**Size:** 1,470 employees × 35 features  
**Target:** `Attrition` (Yes / No) — binary classification with class imbalance (~84% No, ~16% Yes)

Key features include Age, Department, JobRole, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, and more.

## Project Structure

```
├── Employee_Attrition_Prediction.ipynb   # Main analysis notebook
├── WA_Fn-UseC_-HR-Employee-Attrition.csv # Dataset
├── results/
│   └── figures/                          # Saved plots
│       ├── logistic_confusion_matrix.png
│       ├── logistic_roc_curve.png
│       ├── RF_confusion_matrix.png
│       ├── RF_roc_curve.png
│       ├── XGBoost_confusion_matrix.png
│       ├── XGBoost_roc_curve.png
│       ├── xgboost_feature_importance.png
│       ├── shap_summary_plot.png
│       ├── shap_bar_plot.png
│       └── shap_waterfall.png
└── README.md
```

## Methodology

### 1. Data Preprocessing
- Dropped constant/redundant columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`)
- Encoded the target variable (`Attrition`: Yes → 1, No → 0)
- Applied one-hot encoding to categorical features
- Standardized features using `StandardScaler`

### 2. Train/Test Split
- 80% training / 20% test split with stratification to preserve class balance

### 3. Models Trained

| Model | Class Imbalance Handling |
|-------|--------------------------|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"`, 100 estimators |
| XGBoost | `scale_pos_weight` set to majority/minority ratio |

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score

### 5. Feature Importance & Explainability
- Top-10 feature importances from XGBoost
- SHAP values: summary plot, bar plot, and individual waterfall explanation

### 6. Model Stability (Cross-Validation)
- 5-fold Stratified K-Fold cross-validation
- Metrics: Accuracy and ROC-AUC (mean ± std) for Logistic Regression and XGBoost

## Requirements

Install dependencies with pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
```

For the Streamlit app, also install:

```bash
pip install streamlit joblib
```

## Usage

1. Clone the repository and navigate to the project folder.
2. Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the root directory.
3. Open and run the notebook:

```bash
jupyter notebook Employee_Attrition_Prediction.ipynb
```

All figures are automatically saved to `results/figures/`.

## Streamlit Inference App

This project includes a Streamlit web app (`app.py`) for inference on uploaded employee CSV files.

### App Capabilities

- Upload employee CSV files and run inference with the saved Logistic Regression artifacts
- Display per-row outputs:
	- `Attrition_Probability`
	- `Predicted_Attrition`
	- `Risk_Level`
- Interactive decision threshold slider that updates prediction labels and evaluation metrics
- Inference visualizations:
	- Risk level distribution
	- Probability distribution (with threshold marker)
	- Predicted 0/1 count
- SHAP inference explanations (summary + bar) for threshold-selected high-risk rows
- Evaluation metrics/plots when uploaded `Attrition` labels are present:
	- Accuracy, Precision, Recall, F1, ROC-AUC
	- Confusion Matrix
	- ROC Curve (with current threshold operating point)

### One-Time Artifact Generation

Before running the app, generate model artifacts from the training script:

```bash
python3 Employee_Attrition_Prediction.py
```

This creates:

- `results/artifacts/model.joblib`
- `results/artifacts/scaler.joblib`
- `results/artifacts/feature_columns.json`

### Run the App

```bash
python3 -m streamlit run app.py
```

### Streamlit Community Cloud Deployment

For Streamlit Community Cloud, use `requirements.txt` for dependencies and set the Python version directly in the Streamlit deployment UI.

Recommended deployment flow:

1. Push the latest `main` branch to GitHub.
2. In Streamlit Community Cloud, create a **new app** from this repository.
3. Before deploying, open **Advanced settings**.
4. Set **Python version** to **3.11**.
5. Set the main file path to `app.py`.
6. Deploy and verify the app with a sample upload.

If a previous deployment was created with a different interpreter version, delete that app and redeploy from scratch with Python 3.11 selected in Advanced settings.

### Test Dataset

Use `synthetic_employee_attrition_test.csv` for demo/testing.

### Demo Script (Presentation-Ready)

1. Start app and upload `synthetic_employee_attrition_test.csv`.
2. Show the top metrics row (predicted count, average probability, threshold, accuracy/precision/recall/F1).
3. Move the threshold slider and highlight how labels/metrics/confusion matrix update live.
4. Show SHAP plots to explain drivers for currently high-risk predictions.
5. Export results with `Download Predictions CSV`.

### Recommended Starting Threshold

On the current balanced synthetic test set, a practical starting threshold is around **0.36** (best F1 in a coarse threshold sweep):

- F1: 0.7933
- Accuracy: 0.7715
- Precision: 0.7242
- Recall: 0.8770

Use this as an initial operating point, then tune based on business cost of false positives vs false negatives.

## Next Steps / Upgrade Roadmap

The current system is end-to-end for training and inference. The following upgrades can make it more production-ready and user-friendly.

### 1) Model and Data Operations

- Add dataset and model versioning so each deployed model is traceable to a specific training dataset and code revision.
- Track experiment metadata (thresholds, metrics, feature schema, run date) for reproducibility.
- Save model cards that summarize intended use, known limitations, and evaluation context.

### 2) Reliability and Validation

- Add automated tests for preprocessing, schema alignment, and artifact loading.
- Add input schema validation in the app (required columns, allowed value formats, null checks).
- Add CI checks to run linting/tests before merges.

### 3) Deployment and Monitoring

- Containerize the app with Docker for consistent runtime behavior.
- Add deployment automation (CI/CD) for repeatable releases.
- Add monitoring for data drift, prediction drift, and runtime errors.
- For Streamlit Community Cloud specifically, treat the Python version as a platform setting and pin it to 3.11 during app creation.

### 4) Agentic Explanation Layer (Recommended)

Add an agentic component that interprets model outputs for end users in plain language.

Proposed capabilities:

- Explain key metrics on the current upload (Accuracy, Precision, Recall, F1, ROC-AUC) in business terms.
- Summarize figure insights (confusion matrix trade-offs, ROC interpretation, threshold impact).
- Explain top SHAP drivers for high-risk predictions with concise, user-friendly narratives.
- Generate role-based summaries (executive summary vs analyst detail view).
- Provide action-oriented suggestions (for example, retention interventions for high-risk groups).

Implementation note:

- Keep this explanation layer separate from model prediction logic.
- Use guardrails so explanations are descriptive and evidence-based (no unsupported causal claims).

### 5) Governance and UX Improvements

- Add audit logging for uploads, thresholds used, and generated outputs.
- Add an optional report export (PDF/HTML) combining metrics, figures, and agentic narrative.
- Add role-based access if deployed in an enterprise environment.

## Productionization Baseline (Implemented)

This repository now includes a first productionization baseline across versioning, validation, CI/CD checks, and monitoring.

### 1) Model Versioning Metadata

Training now writes model metadata to:

- `results/artifacts/model_metadata.json`

Metadata includes:

- model name
- model version (timestamped)
- training timestamp (UTC)
- feature count
- source dataset
- git commit hash (when available)

### 2) Input Validation

The Streamlit app validates uploaded schema before inference using `src/inference_utils.py`.

- Missing required columns are reported explicitly.
- Inference is halted when schema is invalid.

### 3) CI/CD Quality Gate (GitHub Actions)

Workflow file:

- `.github/workflows/ci.yml`

Current checks on push/PR to `main`:

- dependency install from `requirements.txt`
- Python compile checks (`app.py`, `src/*.py`)
- unit test execution (`python -m unittest discover -s tests -v`)

### 4) Monitoring Logs

The app now supports monitoring event logging via a UI button.

- Log destination: `results/monitoring/inference_events.jsonl`
- Captures: app/model version, rows scored, threshold, predicted leavers, average probability
- Includes test metrics when labels are available

### 5) Test Coverage Added

Unit tests were added for core production utilities:

- `tests/test_inference_utils.py`
- `tests/test_ops_utils.py`

Run locally:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
