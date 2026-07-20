import unittest

import pandas as pd

from src.agent_utils import (
    build_metrics_explanation,
    build_shap_explanation,
    build_upload_summary,
)


class TestBuildUploadSummary(unittest.TestCase):
    def _make_df(self, n=10, leavers=3):
        probs = [0.8] * leavers + [0.2] * (n - leavers)
        preds = [1] * leavers + [0] * (n - leavers)
        risk = ["High"] * leavers + ["Low"] * (n - leavers)
        return pd.DataFrame(
            {
                "Attrition_Probability": probs,
                "Predicted_Attrition": preds,
                "Risk_Level": risk,
            }
        )

    def test_contains_total_count(self):
        df = self._make_df(n=10, leavers=3)
        result = build_upload_summary(df, threshold=0.5)
        self.assertIn("10 employees", result)

    def test_contains_predicted_leavers(self):
        df = self._make_df(n=10, leavers=4)
        result = build_upload_summary(df, threshold=0.5)
        self.assertIn("4 employees", result)

    def test_with_metrics(self):
        df = self._make_df(n=10, leavers=3)
        metrics = {"accuracy": 0.85, "f1": 0.72, "recall": 0.80}
        result = build_upload_summary(
            df, threshold=0.5, true_labels=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0], metrics=metrics
        )
        self.assertIn("accuracy", result.lower())
        self.assertIn("F1", result)


class TestBuildMetricsExplanation(unittest.TestCase):
    def test_no_metrics_returns_fallback(self):
        result = build_metrics_explanation(None)
        self.assertIn("No ground-truth", result)

    def test_accuracy_in_output(self):
        result = build_metrics_explanation({"accuracy": 0.82, "f1": 0.65})
        self.assertIn("Accuracy", result)
        self.assertIn("F1", result)

    def test_threshold_note_included(self):
        result = build_metrics_explanation({"accuracy": 0.82})
        self.assertIn("Threshold note", result)


class TestBuildShapExplanation(unittest.TestCase):
    def test_empty_features_returns_fallback(self):
        result = build_shap_explanation([])
        self.assertIn("not available", result)

    def test_top_features_listed(self):
        features = ["OverTime_Yes", "MonthlyIncome", "Age", "YearsAtCompany", "JobLevel"]
        result = build_shap_explanation(features)
        self.assertIn("OverTime_Yes", result)
        self.assertIn("MonthlyIncome", result)

    def test_guardrail_present(self):
        features = ["OverTime_Yes", "MonthlyIncome"]
        result = build_shap_explanation(features)
        self.assertIn("Guardrail", result)


if __name__ == "__main__":
    unittest.main()
