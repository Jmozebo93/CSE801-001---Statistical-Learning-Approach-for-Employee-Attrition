import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Create a basic page layout
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Intelligence Dashboard")
st.write("Upload an employee CSV file to score attrition risk, review model explanations, and inspect evaluation metrics.")

# Load the model's artifacts once
@st.cache_resource
def load_artifacts():
    model = joblib.load("results/artifacts/model.joblib")
    scaler = joblib.load("results/artifacts/scaler.joblib")
    with open("results/artifacts/feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns

def preprocess_for_inference(df, scaler, feature_columns):
    drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # If target exists in uploaded file, remove it before prediction
    if "Attrition" in df.columns:  
        df = df.drop(columns=["Attrition"])

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Align incoming data to training schema  
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_encoded


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


# Load the model and feature columns
model, scaler, feature_columns = load_artifacts()

# File uploader
uploaded_file = st.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start inference.")
else:
    # Read the uploaded file
    raw_df = pd.read_csv(uploaded_file)
    display_df = raw_df.drop(columns=["Attrition"], errors="ignore")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )
    st.caption("Rows with probability greater than or equal to this threshold are labeled as predicted attrition.")

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(display_df.head())

    # preprocess for inference
    X_aligned = preprocess_for_inference(raw_df.copy(), scaler, feature_columns)
    X_input = scaler.transform(X_aligned)
    proba = model.predict_proba(X_input)[:, 1]
    pred = (proba >= threshold).astype(int)
    true_labels = parse_true_attrition_labels(raw_df)

    # show the prediction results
    results_df = display_df.copy()
    results_df["Attrition_Probability"] = proba
    results_df["Predicted_Attrition"] = pred
    results_df["Risk_Level"] = assign_risk_levels(results_df["Attrition_Probability"], threshold)

    st.subheader("Scored Employee Predictions")
    st.dataframe(results_df)

    # Key metrics
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Predicted Leavers", int(results_df["Predicted_Attrition"].sum()))
    c2.metric("Avg Attrition Probability", round(float(results_df["Attrition_Probability"].mean()), 4))
    c3.metric("Threshold", f"{threshold:.2f}")
    if true_labels is not None:
        c4.metric("Test Accuracy", f"{accuracy_score(true_labels, pred):.3f}")
        c5.metric("Test Precision", f"{precision_score(true_labels, pred, zero_division=0):.3f}")
        c6.metric("Test Recall", f"{recall_score(true_labels, pred, zero_division=0):.3f}")
        c7.metric("Test F1", f"{f1_score(true_labels, pred, zero_division=0):.3f}")
    else:
        c4.metric("Test Accuracy", "N/A")
        c5.metric("Test Precision", "N/A")
        c6.metric("Test Recall", "N/A")
        c7.metric("Test F1", "N/A")

    # Probability distribution chart
    st.subheader("Inference Analytics")

    fig, ax = plt.subplots(figsize=(8, 4))
    risk_counts = results_df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax, palette=["#2E8B57", "#DAA520", "#CD5C5C"])
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Employees")
    ax.set_title("Risk Level Distribution")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(results_df["Attrition_Probability"], bins=20, kde=True, ax=ax)
    ax.axvline(threshold, color="#B22222", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Predicted Attrition Probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    pred_counts = results_df["Predicted_Attrition"].value_counts().sort_index()
    sns.barplot(x=pred_counts.index.astype(str), y=pred_counts.values, ax=ax, palette=["#4C78A8", "#F58518"])
    ax.set_xlabel("Predicted Attrition")
    ax.set_ylabel("Employees")
    ax.set_title("Predicted Attrition Count (0/1)")
    st.pyplot(fig)

    st.subheader("Model Explanations (SHAP)")
    high_risk_mask = proba >= threshold
    X_shap_pool = X_input[high_risk_mask]
    if len(X_shap_pool) == 0:
        st.info("No records meet the current threshold for high-risk SHAP explanations.")
    else:
        sample_size = min(300, len(X_shap_pool))
        background_size = min(200, len(X_input))
        X_sample = X_shap_pool[:sample_size]
        X_sample_df = pd.DataFrame(X_sample, columns=feature_columns)

        explainer = shap.LinearExplainer(model, X_input[:background_size])
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        else:
            shap_matrix = shap_values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, X_sample_df, show=False, max_display=15)
        plt.title("SHAP Summary (High-Risk Rows)")
        st.pyplot(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_matrix, X_sample_df, plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Feature Importance (High-Risk Rows)")
        st.pyplot(fig)
        plt.close(fig)

    # Download predictions
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        data=csv_bytes,
        file_name="attrition_predictions.csv",
        mime="text/csv",
    )

    # If true labels are present, compute test/evaluation metrics for this uploaded file.
    if true_labels is not None:
        st.subheader("Ground-Truth Evaluation (Uploaded Labels)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{accuracy_score(true_labels, pred):.3f}")
        c2.metric("Precision", f"{precision_score(true_labels, pred, zero_division=0):.3f}")
        c3.metric("Recall", f"{recall_score(true_labels, pred, zero_division=0):.3f}")
        c4.metric("F1", f"{f1_score(true_labels, pred, zero_division=0):.3f}")
        try:
            uploaded_auc = roc_auc_score(true_labels, proba)
            c5.metric("ROC-AUC", f"{uploaded_auc:.3f}")
        except ValueError:
            uploaded_auc = None
            c5.metric("ROC-AUC", "N/A")

        cm = confusion_matrix(true_labels, pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Uploaded File)")
        st.pyplot(fig)

        if uploaded_auc is not None:
            fpr, tpr, _ = roc_curve(true_labels, proba)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, label=f"AUC = {uploaded_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")

            tn, fp, fn, tp = confusion_matrix(true_labels, pred).ravel()
            tpr_point = tp / (tp + fn) if (tp + fn) else 0.0
            fpr_point = fp / (fp + tn) if (fp + tn) else 0.0
            ax.scatter([fpr_point], [tpr_point], color="#B22222", s=60, label=f"Threshold {threshold:.2f}")

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve (Uploaded File)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("ROC-AUC requires both positive and negative labels in uploaded Attrition data.")
    else:
        st.info("No usable Attrition labels found in upload. Showing inference-only analytics.")