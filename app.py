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

from src.inference_utils import (
    assign_risk_levels,
    parse_true_attrition_labels,
    preprocess_for_inference,
    validate_input_schema,
)
from src.ops_utils import load_model_metadata, write_monitoring_event

# Create a basic page layout
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.markdown(
    """
    <style>
        .hero-card {
            padding: 1.1rem 1.25rem;
            border-radius: 1rem;
            border: 1px solid rgba(49, 51, 63, 0.12);
            background: linear-gradient(135deg, rgba(48, 91, 185, 0.08), rgba(194, 78, 140, 0.06));
            margin-bottom: 1rem;
        }
        .section-card {
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(49, 51, 63, 0.10);
            background: white;
        }
        .small-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5f6b7a;
            margin-bottom: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="small-label">Employee Attrition Intelligence Dashboard</div>
        <h2 style="margin-bottom:0.35rem;">Score attrition risk, inspect explanations, and review model quality in one place.</h2>
        <p style="margin-bottom:0;">Upload a CSV, choose a decision threshold, and review predictions, confidence, and evaluation metrics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the model's artifacts once
@st.cache_resource
def load_artifacts():
    model = joblib.load("results/artifacts/model.joblib")
    scaler = joblib.load("results/artifacts/scaler.joblib")
    with open("results/artifacts/feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns


# Load the model and feature columns
model, scaler, feature_columns = load_artifacts()
model_metadata = load_model_metadata()

with st.sidebar:
    st.header("Model Metadata")
    st.write(f"Model: {model_metadata['model_name']}")
    st.write(f"Version: {model_metadata['model_version']}")
    st.write(f"Trained At (UTC): {model_metadata['trained_at_utc']}")
    st.write(f"Feature Count: {model_metadata['feature_count']}")
    st.divider()
    st.subheader("How to use")
    st.write("1. Upload a CSV file with the employee fields.")
    st.write("2. Adjust the threshold if you want stricter or looser risk labeling.")
    st.write("3. Review the table, charts, and SHAP explanations.")
    st.divider()
    st.caption("Monitoring events can be logged from the main page after scoring.")

# File uploader
uploaded_file = st.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start inference.")
else:
    # Read the uploaded file
    raw_df = pd.read_csv(uploaded_file)
    missing_columns = validate_input_schema(raw_df)
    if missing_columns:
        st.error(
            "Input schema validation failed. Missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

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
    X_aligned = preprocess_for_inference(raw_df.copy(), feature_columns)
    X_input = scaler.transform(X_aligned)
    proba = model.predict_proba(X_input)[:, 1]
    pred = (proba >= threshold).astype(int)
    true_labels = parse_true_attrition_labels(raw_df)

    # show the prediction results
    results_df = display_df.copy()
    results_df["Attrition_Probability"] = proba
    results_df["Predicted_Attrition"] = pred
    results_df["Risk_Level"] = assign_risk_levels(results_df["Attrition_Probability"], threshold)

    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    summary_col_1.metric("Rows Scored", len(results_df))
    summary_col_2.metric("Predicted Leavers", int(results_df["Predicted_Attrition"].sum()))
    summary_col_3.metric("Avg Attrition Probability", round(float(results_df["Attrition_Probability"].mean()), 4))
    summary_col_4.metric("Threshold", f"{threshold:.2f}")

    overview_tab, analytics_tab, shap_tab, evaluation_tab = st.tabs(
        ["Overview", "Analytics", "Explainability", "Evaluation"]
    )

    with overview_tab:
        left_col, right_col = st.columns([1.1, 1.6])
        with left_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### What this means")
            st.write(
                "The model scores each employee with an attrition probability, then labels rows as low, medium, or high risk using the selected threshold."
            )
            st.write(
                "Use a lower threshold to catch more potential leavers, or a higher threshold to focus on the strongest signals."
            )
            st.markdown("**Quick summary**")
            st.write(f"- Rows scored: {len(results_df)}")
            st.write(f"- Predicted leavers: {int(results_df['Predicted_Attrition'].sum())}")
            st.write(f"- Current threshold: {threshold:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Uploaded Dataset Preview")
            st.dataframe(display_df.head(), use_container_width=True)
            st.markdown("### Scored Employee Predictions")
            st.dataframe(results_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with analytics_tab:
        fig, ax = plt.subplots(figsize=(8, 4))
        risk_counts = results_df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax, palette=["#2E8B57", "#DAA520", "#CD5C5C"])
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Employees")
        ax.set_title("Risk Level Distribution")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
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

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="attrition_predictions.csv",
            mime="text/csv",
        )

    with shap_tab:
        st.markdown("### Model Explanations (SHAP)")
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

    with evaluation_tab:
        if true_labels is not None:
            st.markdown("### Ground-Truth Evaluation (Uploaded Labels)")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Accuracy", f"{accuracy_score(true_labels, pred):.3f}")
            e2.metric("Precision", f"{precision_score(true_labels, pred, zero_division=0):.3f}")
            e3.metric("Recall", f"{recall_score(true_labels, pred, zero_division=0):.3f}")
            e4.metric("F1", f"{f1_score(true_labels, pred, zero_division=0):.3f}")
            try:
                uploaded_auc = roc_auc_score(true_labels, proba)
                e5.metric("ROC-AUC", f"{uploaded_auc:.3f}")
            except ValueError:
                uploaded_auc = None
                e5.metric("ROC-AUC", "N/A")

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
            st.info("No usable Attrition labels found in upload. Show evaluation metrics by uploading a labeled test file.")

    monitoring_event = {
        "app_version": "1.0.0",
        "model_version": model_metadata["model_version"],
        "rows_scored": int(len(results_df)),
        "threshold": float(threshold),
        "predicted_leavers": int(results_df["Predicted_Attrition"].sum()),
        "avg_probability": float(results_df["Attrition_Probability"].mean()),
        "labels_present": bool(true_labels is not None),
    }
    if true_labels is not None:
        monitoring_event.update(
            {
                "accuracy": float(accuracy_score(true_labels, pred)),
                "precision": float(precision_score(true_labels, pred, zero_division=0)),
                "recall": float(recall_score(true_labels, pred, zero_division=0)),
                "f1": float(f1_score(true_labels, pred, zero_division=0)),
            }
        )

    if st.button("Log Monitoring Event"):
        write_monitoring_event(monitoring_event)
        st.success("Monitoring event saved to results/monitoring/inference_events.jsonl")
    st.caption("Tip: upload a labeled file to unlock evaluation metrics and ROC/CM charts.")