"""
predict.py — Real-Time Event Prediction Pipeline
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (get_logger, load_model, load_json, MODELS_DIR, REPORTS_DIR,
                   SEVERITY_LABELS, MANPOWER_MAP, BARRICADE_MAP,
                   BARRICADE_LABELS, DIVERSION_LABELS, CAUSE_SEVERITY,
                   DIVERSION_CAUSES, print_section)

log = get_logger("predict")

CITY_LAT = 12.9716
CITY_LON  = 77.5946

# ──────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2=CITY_LAT, lon2=CITY_LON) -> float:
    R = 6371.0
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat1 - rlat2; dlon = rlon1 - rlon2
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


# ──────────────────────────────────────────────────────────────────────────────
def _scalar_to_label(model, X_row: pd.DataFrame) -> dict:
    """Run model prediction and return label + proba."""
    pred  = int(model.predict(X_row)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_row)[0]
        confidence = float(proba.max())
    else:
        confidence = 1.0
    return {"class": pred, "confidence": confidence, "proba": proba}


# ──────────────────────────────────────────────────────────────────────────────
def build_feature_row(
    event_cause: str,
    event_type: str,
    priority: str,
    requires_road_closure: bool,
    latitude: float,
    longitude: float,
    zone_enc: int = 0,
    police_station_enc: int = 0,
    corridor_enc: int = 0,
    veh_type_enc: int = 0,
    authenticated: bool = True,
    timestamp: datetime = None,
    zone_events_1d: float = 5.0,
    zone_events_7d: float = 20.0,
    zone_events_30d: float = 60.0,
    prev_severity_zone: float = 1.0,
    zone_mean_severity: float = 1.0,
    cause_zone_avg_dur: float = 180.0,
    station_avg_dur: float = 180.0,
    spatial_cluster: int = 0,
    cluster_density: int = 500,
    has_junction: int = 0,
    **kwargs
) -> pd.DataFrame:

    if timestamp is None:
        timestamp = datetime.utcnow()

    # --- event encodings ---
    cause_map = {
        "vehicle_breakdown":0, "others":1, "pot_holes":2, "construction":3,
        "water_logging":4, "accident":5, "tree_fall":6, "road_conditions":7,
        "congestion":8, "public_event":9, "procession":10, "vip_movement":11,
        "protest":12, "debris":13, "test_demo":14, "fog / low visibility":15
    }
    event_cause_enc = cause_map.get(event_cause.lower().strip(), 1)
    event_type_enc  = 1 if event_type.lower() == "unplanned" else 0
    priority_enc    = 1 if priority.lower() == "high" else 0
    road_closure_enc= 1 if requires_road_closure else 0

    hour   = timestamp.hour
    minute = timestamp.minute
    wd     = timestamp.weekday()
    day    = timestamp.day
    month  = timestamp.month
    woy    = timestamp.isocalendar()[1]

    peak_hours_m = set(range(7, 11))
    peak_hours_e = set(range(17, 22))
    night_hours  = set(range(22, 24)) | set(range(0, 6))

    is_peak = int(hour in (peak_hours_m | peak_hours_e))
    is_night= int(hour in night_hours)
    dist    = haversine(latitude, longitude)

    # target enc approximation
    cause_target_enc = CAUSE_SEVERITY.get(event_cause.lower(), 0) * 1.0
    corridor_freq    = 0.01

    row = {
        "hour": hour, "minute": minute, "day": day, "weekday": wd,
        "month": month, "week_of_year": woy,
        "is_weekend": int(wd >= 5), "is_peak": is_peak,
        "is_night": is_night,
        "is_morning_peak": int(hour in peak_hours_m),
        "is_evening_peak": int(hour in peak_hours_e),
        "time_since_midnight": hour*3600 + minute*60,
        "hour_sin": np.sin(2*np.pi*hour/24),
        "hour_cos": np.cos(2*np.pi*hour/24),
        "weekday_sin": np.sin(2*np.pi*wd/7),
        "weekday_cos": np.cos(2*np.pi*wd/7),
        "month_sin": np.sin(2*np.pi*month/12),
        "month_cos": np.cos(2*np.pi*month/12),
        "latitude": latitude, "longitude": longitude,
        "dist_from_center_km": dist,
        "spatial_cluster": spatial_cluster,
        "cluster_density": cluster_density,
        "has_junction": has_junction,
        "zone_enc": zone_enc,
        "police_station_enc": police_station_enc,
        "corridor_enc": corridor_enc,
        "event_cause_enc": event_cause_enc,
        "veh_type_enc": veh_type_enc,
        "event_type_enc": event_type_enc,
        "priority_enc": priority_enc,
        "road_closure_enc": road_closure_enc,
        "has_veh_no": 0,
        "authenticated_enc": int(authenticated),
        "corridor_freq": corridor_freq,
        "cause_target_enc": cause_target_enc,
        "zone_events_1d": zone_events_1d,
        "zone_events_7d": zone_events_7d,
        "zone_events_30d": zone_events_30d,
        "prev_severity_zone": prev_severity_zone,
        "zone_mean_severity": zone_mean_severity,
        "cause_zone_avg_dur": cause_zone_avg_dur,
        "station_avg_dur": station_avg_dur,
        "cause_x_peak": event_cause_enc * is_peak,
        "cause_x_weekend": event_cause_enc * int(wd >= 5),
        "zone_x_priority": zone_enc * priority_enc,
        "dist_x_priority": dist * priority_enc,
        "cluster_x_cause": spatial_cluster * event_cause_enc,
        "hour_x_event_type": hour * event_type_enc,
        "closure_x_priority": road_closure_enc * priority_enc,
        "sev_x_cluster": cluster_density * cause_target_enc,
    }
    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────────────
def predict_event(
    event_cause: str = "accident",
    event_type: str = "unplanned",
    priority: str = "High",
    requires_road_closure: bool = False,
    latitude: float = 12.9716,
    longitude: float = 77.5946,
    timestamp: datetime = None,
    **kwargs
) -> dict:
    """
    Main prediction function. Returns full resource recommendations.
    """
    print_section("REAL-TIME PREDICTION")
    X_row = build_feature_row(
        event_cause=event_cause,
        event_type=event_type,
        priority=priority,
        requires_road_closure=requires_road_closure,
        latitude=latitude,
        longitude=longitude,
        timestamp=timestamp or datetime.utcnow(),
        **kwargs
    )

    results = {}

    # ── Severity ──────────────────────────────────────────────────────────
    try:
        sev_model = load_model("clf_congestion_severity")
        sev_out   = _scalar_to_label(sev_model, X_row)
        sev_class = sev_out["class"]
        results["congestion_severity"] = {
            "class": sev_class,
            "label": SEVERITY_LABELS.get(sev_class, "Unknown"),
            "confidence": round(sev_out["confidence"] * 100, 1),
        }
    except Exception as e:
        log.warning(f"Severity model error: {e}")
        sev_class = CAUSE_SEVERITY.get(event_cause.lower(), 1)
        results["congestion_severity"] = {
            "class": sev_class,
            "label": SEVERITY_LABELS.get(sev_class, "Unknown"),
            "confidence": 60.0,
        }

    # ── Duration ──────────────────────────────────────────────────────────
    try:
        dur_model = load_model("reg_duration")
        dur_pred  = float(np.clip(dur_model.predict(X_row)[0], 0, None))
        results["predicted_duration_min"] = round(dur_pred, 1)
        results["predicted_duration_hr"]  = round(dur_pred / 60, 2)
    except Exception as e:
        log.warning(f"Duration model error: {e}")
        results["predicted_duration_min"] = 180.0

    # ── Manpower ──────────────────────────────────────────────────────────
    try:
        man_model = load_model("clf_manpower_req")
        man_out   = _scalar_to_label(man_model, X_row)
        man_class = man_out["class"]
    except Exception:
        man_class = sev_class
    officers = MANPOWER_MAP.get(man_class, 8)
    results["manpower"] = {
        "class": man_class,
        "officers_required": officers,
        "confidence": round(man_out.get("confidence", 0.7) * 100, 1)
            if "man_out" in dir() else 70.0,
    }

    # ── Barricades ────────────────────────────────────────────────────────
    try:
        bar_model = load_model("clf_barricade_req")
        bar_out   = _scalar_to_label(bar_model, X_row)
        bar_class = bar_out["class"]
    except Exception:
        bar_class = min(sev_class, 3)
    results["barricades"] = {
        "class": bar_class,
        "label": BARRICADE_LABELS.get(bar_class, "Unknown"),
        "count": BARRICADE_MAP.get(bar_class, 5),
    }

    # ── Diversion ─────────────────────────────────────────────────────────
    try:
        div_model = load_model("clf_diversion_req")
        div_out   = _scalar_to_label(div_model, X_row)
        div_class = div_out["class"]
    except Exception:
        div_class = 1 if (requires_road_closure or
                          event_cause.lower() in DIVERSION_CAUSES) else 0
    results["diversion"] = {
        "class": div_class,
        "label": DIVERSION_LABELS.get(div_class, "Unknown"),
    }

    # ── Summary ───────────────────────────────────────────────────────────
    results["input_summary"] = {
        "event_cause": event_cause,
        "event_type": event_type,
        "priority": priority,
        "road_closure": requires_road_closure,
        "location": f"{latitude:.4f}, {longitude:.4f}",
        "timestamp": str(timestamp or datetime.utcnow()),
    }

    # Print pretty summary
    print("\n" + "-"*55)
    print(f"  EVENT          : {event_cause} ({event_type})")
    print(f"  PRIORITY       : {priority}  | Road Closure: {requires_road_closure}")
    print(f"  LOCATION       : {latitude:.4f}, {longitude:.4f}")
    print("-"*55)
    sev_r = results["congestion_severity"]
    print(f"  SEVERITY       : {sev_r['label']} [{sev_r['class']}]  "
          f"(conf: {sev_r['confidence']}%)")
    print(f"  DURATION       : {results['predicted_duration_min']} min  "
          f"≈ {results['predicted_duration_hr']} hr")
    print(f"  MANPOWER       : {results['manpower']['officers_required']} officers")
    print(f"  BARRICADES     : {results['barricades']['count']} "
          f"({results['barricades']['label']})")
    print(f"  DIVERSION      : {results['diversion']['label']}")
    print("-"*55)

    return results


if __name__ == "__main__":
    result = predict_event(
        event_cause="accident",
        event_type="unplanned",
        priority="High",
        requires_road_closure=True,
        latitude=12.9716,
        longitude=77.5946,
    )
