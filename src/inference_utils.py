import pandas as pd

DROP_COLUMNS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]

# Required raw columns for app inference input (excluding optional dropped columns and target).
REQUIRED_INPUT_COLUMNS = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


def preprocess_for_inference(df, feature_columns):
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    if "Attrition" in df.columns:
        df = df.drop(columns=["Attrition"])

    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded.reindex(columns=feature_columns, fill_value=0)


def parse_true_attrition_labels(df):
    if "Attrition" not in df.columns:
        return None

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }
    mapped = df["Attrition"].astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        return None
    return mapped.astype(int)


def assign_risk_levels(probabilities, threshold):
    low_cutoff = max(0.0, threshold / 2.0)
    bins = [-0.01, low_cutoff, threshold, 1.0]
    return pd.cut(probabilities, bins=bins, labels=["Low", "Medium", "High"], include_lowest=True)


def validate_input_schema(df):
    missing_columns = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    return missing_columns
