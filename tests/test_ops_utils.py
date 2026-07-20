import json
import os
import tempfile
import unittest

from src.ops_utils import load_model_metadata, write_monitoring_event


class TestOpsUtils(unittest.TestCase):
    def test_load_model_metadata_default(self):
        data = load_model_metadata("/tmp/nonexistent_model_metadata_file.json")
        self.assertEqual(data["model_name"], "LogisticRegression")

    def test_write_monitoring_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "inference_events.jsonl")
            write_monitoring_event({"rows_scored": 10}, log_path=log_path)
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r") as f:
                line = f.readline().strip()
            payload = json.loads(line)
            self.assertEqual(payload["rows_scored"], 10)
            self.assertIn("timestamp_utc", payload)


if __name__ == "__main__":
    unittest.main()
