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

## Usage

1. Clone the repository and navigate to the project folder.
2. Ensure `WA_Fn-UseC_-HR-Employee-Attrition.csv` is in the root directory.
3. Open and run the notebook:

```bash
jupyter notebook Employee_Attrition_Prediction.ipynb
```

All figures are automatically saved to `results/figures/`.
