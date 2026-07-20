import unittest

import pandas as pd

from src.inference_utils import (
    assign_risk_levels,
    parse_true_attrition_labels,
    preprocess_for_inference,
    validate_input_schema,
)


class TestInferenceUtils(unittest.TestCase):
    def test_parse_true_attrition_labels_yes_no(self):
        df = pd.DataFrame({"Attrition": ["Yes", "No", "yes", "no"]})
        labels = parse_true_attrition_labels(df)
        self.assertEqual(labels.tolist(), [1, 0, 1, 0])

    def test_parse_true_attrition_labels_invalid_returns_none(self):
        df = pd.DataFrame({"Attrition": ["Yes", "Maybe"]})
        self.assertIsNone(parse_true_attrition_labels(df))

    def test_assign_risk_levels(self):
        probs = pd.Series([0.1, 0.3, 0.8])
        levels = assign_risk_levels(probs, threshold=0.5)
        self.assertEqual(levels.astype(str).tolist(), ["Low", "Medium", "High"])

    def test_preprocess_alignment(self):
        df = pd.DataFrame(
            {
                "Age": [30],
                "BusinessTravel": ["Travel_Rarely"],
                "Department": ["Sales"],
                "OverTime": ["No"],
            }
        )
        feature_cols = ["Age", "BusinessTravel_Travel_Rarely", "Department_Sales", "OverTime_Yes"]
        aligned = preprocess_for_inference(df, feature_cols)
        self.assertEqual(list(aligned.columns), feature_cols)
        self.assertEqual(int(aligned.loc[0, "Age"]), 30)

    def test_validate_input_schema_missing_columns(self):
        df = pd.DataFrame({"Age": [30], "BusinessTravel": ["Travel_Rarely"]})
        missing = validate_input_schema(df)
        self.assertIn("Department", missing)


if __name__ == "__main__":
    unittest.main()
