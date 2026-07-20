"""
agent_utils.py

Phase 1 agentic component: static explanation builders.
All functions return plain English strings derived from model outputs.
No LLM calls are made in Phase 1 — explanations are template-driven.
"""


def build_upload_summary(results_df, threshold, true_labels=None, metrics=None):
    """Return a plain English summary of the uploaded inference results."""
    total = len(results_df)
    predicted_leavers = int(results_df["Predicted_Attrition"].sum())
    stay = total - predicted_leavers
    avg_prob = float(results_df["Attrition_Probability"].mean())

    risk_counts = results_df["Risk_Level"].value_counts().to_dict()
    high = risk_counts.get("High", 0)
    medium = risk_counts.get("Medium", 0)
    low = risk_counts.get("Low", 0)

    lines = [
        f"**Upload Summary**",
        f"",
        f"This upload contains **{total} employees**. "
        f"At the current threshold of **{threshold:.2f}**, the model predicts "
        f"**{predicted_leavers} employees ({predicted_leavers / total * 100:.1f}%) are likely to leave** "
        f"and **{stay} ({stay / total * 100:.1f}%) are expected to stay**.",
        f"",
        f"The average predicted attrition probability across all employees is **{avg_prob:.1%}**.",
        f"",
        f"Risk breakdown:",
        f"- High risk: {high} employees",
        f"- Medium risk: {medium} employees",
        f"- Low risk: {low} employees",
    ]

    if true_labels is not None and metrics:
        acc = metrics.get("accuracy")
        f1 = metrics.get("f1")
        recall = metrics.get("recall")
        if acc is not None and f1 is not None:
            lines += [
                f"",
                f"Ground-truth labels were included. The model achieved "
                f"**{acc:.1%} accuracy** and **{f1:.2f} F1 score** on this file.",
            ]
            if recall is not None:
                lines.append(
                    f"Recall (sensitivity to actual leavers) was **{recall:.1%}**, "
                    f"meaning the model identified that share of true attrition cases."
                )

    return "\n".join(lines)


def build_metrics_explanation(metrics):
    """Return a plain English explanation of classification metrics."""
    if not metrics:
        return (
            "No ground-truth labels were found in this upload. "
            "Upload a file with an `Attrition` column to enable evaluation metrics."
        )

    acc = metrics.get("accuracy")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    auc = metrics.get("roc_auc")

    lines = [
        "**What the metrics mean for this upload**",
        "",
    ]

    if acc is not None:
        lines.append(
            f"- **Accuracy ({acc:.1%}):** Of all employees scored, the model correctly "
            f"classified this proportion. Accuracy alone can be misleading when attrition is rare."
        )
    if precision is not None:
        lines.append(
            f"- **Precision ({precision:.1%}):** Of the employees the model flagged as likely to leave, "
            f"this share actually did leave. Higher precision means fewer false alarms."
        )
    if recall is not None:
        lines.append(
            f"- **Recall ({recall:.1%}):** Of all employees who actually left, "
            f"the model caught this share. Higher recall means fewer missed cases."
        )
    if f1 is not None:
        lines.append(
            f"- **F1 Score ({f1:.2f}):** A balance between precision and recall. "
            f"Useful when both false positives and false negatives carry a cost."
        )
    if auc is not None:
        lines.append(
            f"- **ROC-AUC ({auc:.2f}):** Measures the model's ability to rank "
            f"likely leavers above likely stayers, regardless of threshold. "
            f"Values above 0.7 are generally considered useful."
        )

    lines += [
        "",
        "**Threshold note:** Moving the threshold slider changes precision and recall "
        "in opposite directions. A lower threshold catches more leavers (higher recall) "
        "but also produces more false alarms (lower precision).",
    ]

    return "\n".join(lines)


def build_shap_explanation(top_features):
    """
    Return a plain English explanation of SHAP-derived top features.

    Parameters
    ----------
    top_features : list of str
        Ordered list of top feature names by mean |SHAP value|, most important first.
    """
    if not top_features:
        return (
            "SHAP explanations are not available for this upload. "
            "This may be because no employees meet the current threshold."
        )

    top_n = top_features[:5]
    feature_list = "\n".join(f"  {i + 1}. {f}" for i, f in enumerate(top_n))

    lines = [
        "**What is driving high-risk predictions**",
        "",
        "SHAP (SHapley Additive exPlanations) measures how much each input feature "
        "contributes to a prediction. Features with higher SHAP values push the model "
        "toward predicting attrition.",
        "",
        f"The top factors influencing high-risk predictions in this upload are:",
        "",
        feature_list,
        "",
        "These are patterns observed across the high-risk group in this file. "
        "They reflect statistical associations in the training data, not confirmed causes.",
        "",
        "**Guardrail:** These findings describe patterns; they should not be used "
        "to make individual employment decisions without further review.",
    ]

    return "\n".join(lines)
