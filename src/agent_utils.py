"""
agent_utils.py

Agentic component helpers.
Phase 1: static template-driven explanations (no API required).
Phase 2: LLM-enhanced summaries via Google Gemini (requires GEMINI_API_KEY secret).

LLM functions fall back to static templates gracefully if the key is missing or
the API call fails — the rest of the app is never affected.
"""

import os
import time
from src.cache import get_cached_response, cache_response


def _get_gemini_model():
    """
    Return a configured Gemini GenerativeModel, or None if unavailable.
    Reads the key from the environment variable GEMINI_API_KEY.
    Uses gemini-2.0-flash, the current supported model (gemini-1.5-flash is deprecated).
    """
    try:
        import google.generativeai as genai  # noqa: PLC0415

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        return None


_last_gemini_error = None  # Store last error for debugging


def _get_gemini_error():
    """Return the last Gemini API error, if any."""
    return _last_gemini_error


class GeminiFallbackError(Exception):
    """Raised when Gemini API fails and we fall back to static templates."""
    pass


def _call_gemini(prompt, fallback_fn=None):
    """
    Call Gemini with the given prompt. Return the response text.
    Uses caching to avoid redundant API calls and save quota.
    Raises GeminiFallbackError if API fails so app.py can handle fallback.
    
    Note: fallback_fn parameter is deprecated and ignored. All failures raise GeminiFallbackError.
    """
    global _last_gemini_error
    
    # Check cache first
    cached = get_cached_response(prompt)
    if cached is not None:
        _last_gemini_error = None
        return cached
    
    model = _get_gemini_model()
    if model is None:
        _last_gemini_error = "Model unavailable (no API key)"
        raise GeminiFallbackError(_last_gemini_error)
    
    # Try up to 2 times (initial + 1 retry on quota exceeded)
    for attempt in range(2):
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Cache successful response
            cache_response(prompt, response_text)
            _last_gemini_error = None
            return response_text
        except Exception as e:
            error_str = str(e)
            _last_gemini_error = f"{type(e).__name__}: {error_str[:100]}"
            
            # If quota exceeded and this is first attempt, wait and retry
            if "ResourceExhausted" in error_str and "429" in error_str and attempt == 0:
                retry_delay = 1
                if "retry in" in error_str.lower():
                    try:
                        import re
                        match = re.search(r"retry in (\d+(?:\.\d+)?)", error_str)
                        if match:
                            retry_delay = float(match.group(1))
                    except Exception:
                        pass
                
                retry_delay = min(retry_delay, 5)
                time.sleep(retry_delay)
                continue
            
            # All other errors: raise so caller knows it failed
            raise GeminiFallbackError(_last_gemini_error)
    
    # If we got here, both attempts failed
    raise GeminiFallbackError("API call failed after retry")


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


# ---------------------------------------------------------------------------
# Phase 2: LLM-enhanced summaries (fall back to static if key is unavailable)
# ---------------------------------------------------------------------------

def llm_upload_summary(results_df, threshold, true_labels=None, metrics=None):
    """Return an LLM-enhanced upload summary, falling back to static template."""
    static = build_upload_summary(results_df, threshold, true_labels, metrics)
    total = len(results_df)
    leavers = int(results_df["Predicted_Attrition"].sum())
    avg_prob = float(results_df["Attrition_Probability"].mean())
    risk_counts = results_df["Risk_Level"].value_counts().to_dict()

    metrics_line = ""
    if metrics:
        metrics_line = (
            f"Accuracy: {metrics.get('accuracy', 'N/A'):.1%}, "
            f"F1: {metrics.get('f1', 'N/A'):.2f}, "
            f"Recall: {metrics.get('recall', 'N/A'):.1%}. "
        )

    prompt = f"""You are an HR analytics assistant. Summarize these attrition model results
in 3-4 clear sentences for an HR manager. Be factual and concise. Do not make causal claims.
Do not advise on individual employment decisions.

Facts:
- Employees scored: {total}
- Predicted leavers: {leavers} ({leavers / total * 100:.1f}%)
- Average attrition probability: {avg_prob:.1%}
- Decision threshold: {threshold:.2f}
- High risk: {risk_counts.get('High', 0)}, Medium: {risk_counts.get('Medium', 0)}, Low: {risk_counts.get('Low', 0)}
- {metrics_line if metrics_line else 'No ground-truth labels provided.'}
"""
    return _call_gemini(prompt, None)


def llm_metrics_explanation(metrics):
    """Return an LLM-enhanced metrics explanation, falling back to static template."""
    if not metrics:
        return build_metrics_explanation(metrics)

    prompt = f"""You are an HR analytics assistant. Explain these model evaluation metrics
in 3-5 plain English sentences for a non-technical HR manager. Do not use jargon.
Highlight what the numbers mean practically. Do not advise on hiring or firing decisions.

Metrics:
- Accuracy: {metrics.get('accuracy', 'N/A')}
- Precision: {metrics.get('precision', 'N/A')}
- Recall: {metrics.get('recall', 'N/A')}
- F1 Score: {metrics.get('f1', 'N/A')}
- ROC-AUC: {metrics.get('roc_auc', 'N/A')}
"""
    return _call_gemini(prompt, None)


def llm_shap_explanation(top_features):
    """Return an LLM-enhanced SHAP explanation, falling back to static template."""
    if not top_features:
        return build_shap_explanation(top_features)

    features_str = ", ".join(top_features[:5])
    prompt = f"""You are an HR analytics assistant. Explain in 3-4 plain English sentences
why these employee features are associated with attrition risk according to a SHAP analysis.
Do not make causal claims. Do not advise on individual employees.
End with a brief guardrail note that these are statistical patterns, not certainties.

Top SHAP features: {features_str}
"""
    return _call_gemini(prompt, None)
