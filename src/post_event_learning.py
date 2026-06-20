"""
post_event_learning.py — Post-Event Feedback & Drift Monitoring
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_logger, DATA_PROC, REPORTS_DIR, print_section

log = get_logger("post_event_learning")
FEEDBACK_CSV = DATA_PROC / "post_event_feedback.csv"


FEEDBACK_SCHEMA = [
    "timestamp", "event_cause", "event_type", "priority",
    "requires_road_closure", "latitude", "longitude",
    # Predictions
    "pred_severity", "pred_duration_min", "pred_officers",
    "pred_barricade_class", "pred_diversion",
    # Actuals (filled in after event)
    "actual_severity", "actual_duration_min", "actual_officers",
    "actual_barricade_class", "actual_diversion",
]


def log_prediction(prediction_result: dict, event_info: dict):
    """Store a prediction record to the feedback CSV."""
    row = {
        "timestamp":           datetime.utcnow().isoformat(),
        "event_cause":         event_info.get("event_cause"),
        "event_type":          event_info.get("event_type"),
        "priority":            event_info.get("priority"),
        "requires_road_closure": event_info.get("requires_road_closure"),
        "latitude":            event_info.get("latitude"),
        "longitude":           event_info.get("longitude"),
        "pred_severity":       prediction_result.get("congestion_severity", {}).get("class"),
        "pred_duration_min":   prediction_result.get("predicted_duration_min"),
        "pred_officers":       prediction_result.get("manpower", {}).get("officers_required"),
        "pred_barricade_class":prediction_result.get("barricades", {}).get("class"),
        "pred_diversion":      prediction_result.get("diversion", {}).get("class"),
        "actual_severity":     None,
        "actual_duration_min": None,
        "actual_officers":     None,
        "actual_barricade_class": None,
        "actual_diversion":    None,
    }
    df_new = pd.DataFrame([row])
    if FEEDBACK_CSV.exists():
        df_existing = pd.read_csv(FEEDBACK_CSV)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(FEEDBACK_CSV, index=False)
    log.info(f"Prediction logged. Total records: {len(df)}")


def record_actual(record_index: int, actual_severity: int = None,
                  actual_duration_min: float = None, actual_officers: int = None,
                  actual_barricade_class: int = None, actual_diversion: int = None):
    """Update an existing record with actual values post-event."""
    if not FEEDBACK_CSV.exists():
        log.error("Feedback CSV not found. No predictions have been logged yet.")
        return
    df = pd.read_csv(FEEDBACK_CSV)
    if record_index >= len(df):
        log.error(f"Record index {record_index} out of range (max {len(df)-1})")
        return
    updates = {
        "actual_severity": actual_severity,
        "actual_duration_min": actual_duration_min,
        "actual_officers": actual_officers,
        "actual_barricade_class": actual_barricade_class,
        "actual_diversion": actual_diversion,
    }
    for k, v in updates.items():
        if v is not None:
            df.loc[record_index, k] = v
    df.to_csv(FEEDBACK_CSV, index=False)
    log.info(f"Record {record_index} updated with actuals.")


def compute_drift_report() -> dict:
    """Compare predictions vs actuals and check for drift."""
    print_section("POST-EVENT DRIFT ANALYSIS")
    if not FEEDBACK_CSV.exists():
        log.warning("No feedback data found.")
        return {}

    df = pd.read_csv(FEEDBACK_CSV)
    df_eval = df.dropna(subset=["actual_severity", "pred_severity"])

    if len(df_eval) == 0:
        log.info("No complete records yet (no actuals logged).")
        return {}

    from sklearn.metrics import mean_absolute_error, accuracy_score

    report = {
        "total_predictions": len(df),
        "evaluated_records": len(df_eval),
        "severity": {},
        "duration": {},
        "recommendation": ""
    }

    # Severity accuracy
    sev_acc = accuracy_score(df_eval["actual_severity"].astype(int),
                              df_eval["pred_severity"].astype(int))
    report["severity"]["accuracy"] = round(sev_acc, 4)
    report["severity"]["mean_error"] = round(
        float(df_eval["actual_severity"].astype(float).sub(
              df_eval["pred_severity"].astype(float)).abs().mean()), 3)

    # Duration MAE
    dur_eval = df.dropna(subset=["actual_duration_min", "pred_duration_min"])
    if len(dur_eval) > 0:
        report["duration"]["MAE"] = round(
            mean_absolute_error(dur_eval["actual_duration_min"],
                                 dur_eval["pred_duration_min"]), 2)

    # Drift flag
    if sev_acc < 0.70:
        report["recommendation"] = "⚠ Significant drift detected — RETRAIN RECOMMENDED"
        log.warning("DRIFT DETECTED — Accuracy below 70%. Consider retraining.")
    else:
        report["recommendation"] = "✓ Model performance acceptable. No retraining needed."
        log.info("Model performance acceptable.")

    print(json.dumps(report, indent=2))
    # Save
    out = REPORTS_DIR / "drift_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


def get_feedback_df() -> pd.DataFrame:
    """Load the full feedback DataFrame."""
    if FEEDBACK_CSV.exists():
        return pd.read_csv(FEEDBACK_CSV)
    return pd.DataFrame(columns=FEEDBACK_SCHEMA)


if __name__ == "__main__":
    compute_drift_report()
