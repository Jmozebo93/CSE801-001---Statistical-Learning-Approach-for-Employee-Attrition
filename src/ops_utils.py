import json
import os
from datetime import datetime, timezone


DEFAULT_METADATA = {
    "model_name": "LogisticRegression",
    "model_version": "unknown",
    "trained_at_utc": "unknown",
    "feature_count": "unknown",
    "source_dataset": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
}


def load_model_metadata(metadata_path="results/artifacts/model_metadata.json"):
    if not os.path.exists(metadata_path):
        return DEFAULT_METADATA.copy()

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    merged = DEFAULT_METADATA.copy()
    merged.update(metadata)
    return merged


def write_monitoring_event(event, log_path="results/monitoring/inference_events.jsonl"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    event_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(event_payload) + "\n")
