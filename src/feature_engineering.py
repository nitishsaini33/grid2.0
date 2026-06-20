"""
feature_engineering.py — Full Feature Construction Pipeline
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import get_logger, DATA_PROC, print_section

log = get_logger("feature_engineering")

# Bangalore city centre
CITY_LAT = 12.9716
CITY_LON = 77.5946

# Peak hour windows (hour of day)
MORNING_PEAK = set(range(7, 11))    # 7–10
EVENING_PEAK = set(range(17, 22))   # 5–9 pm
NIGHT_HOURS  = set(range(22, 24)) | set(range(0, 6))

# ──────────────────────────────────────────────────────────────────────────────
#  TIME FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("TIME FEATURES")
    dt = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["hour"]        = dt.dt.hour.fillna(0).astype(int)
    df["minute"]      = dt.dt.minute.fillna(0).astype(int)
    df["day"]         = dt.dt.day.fillna(1).astype(int)
    df["weekday"]     = dt.dt.weekday.fillna(0).astype(int)
    df["month"]       = dt.dt.month.fillna(1).astype(int)
    df["week_of_year"]= dt.dt.isocalendar().week.fillna(1).astype(int)
    df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
    df["is_peak"]     = df["hour"].isin(MORNING_PEAK | EVENING_PEAK).astype(int)
    df["is_night"]    = df["hour"].isin(NIGHT_HOURS).astype(int)
    df["is_morning_peak"] = df["hour"].isin(MORNING_PEAK).astype(int)
    df["is_evening_peak"] = df["hour"].isin(EVENING_PEAK).astype(int)
    df["time_since_midnight"] = df["hour"] * 3600 + df["minute"] * 60

    # Cyclical encoding
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    log.info(f"Added {17} time features")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  SPATIAL FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def add_spatial_features(df: pd.DataFrame, n_clusters: int = 15) -> pd.DataFrame:
    print_section("SPATIAL FEATURES")

    # Distance from city centre (Haversine approx)
    lat_r = np.radians(df["latitude"])
    lon_r = np.radians(df["longitude"])
    clat  = np.radians(CITY_LAT)
    clon  = np.radians(CITY_LON)
    dlat  = lat_r - clat
    dlon  = lon_r - clon
    a = np.sin(dlat/2)**2 + np.cos(lat_r) * np.cos(clat) * np.sin(dlon/2)**2
    df["dist_from_center_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

    # KMeans spatial clusters
    coords = df[["latitude", "longitude"]].fillna(df[["latitude","longitude"]].median())
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["spatial_cluster"] = km.fit_predict(coords)

    # Cluster density (events per cluster)
    cluster_counts = df["spatial_cluster"].value_counts().to_dict()
    df["cluster_density"] = df["spatial_cluster"].map(cluster_counts)

    # Has junction
    df["has_junction"] = (df["junction"].fillna("None") != "None").astype(int)

    log.info("Added spatial features: dist_from_center, spatial_cluster, cluster_density, has_junction")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  CATEGORICAL ENCODING
# ──────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    print_section("CATEGORICAL ENCODING")

    le = LabelEncoder()

    # Label encode moderate cardinality cols
    for col in ["zone", "police_station", "corridor", "event_cause", "veh_type"]:
        df[f"{col}_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))

    # Binary
    df["event_type_enc"]      = (df["event_type"] == "unplanned").astype(int)
    df["priority_enc"]        = (df["priority"].fillna("Low") == "High").astype(int)
    df["road_closure_enc"]    = df["requires_road_closure"].astype(int)
    df["has_veh_no"]          = (df["veh_no"].fillna("Unknown") != "Unknown").astype(int)
    df["authenticated_enc"]   = (df["authenticated"].fillna("false").str.lower() == "true").astype(int)

    # Frequency encoding for corridor (high cardinality)
    freq = df["corridor"].value_counts(normalize=True).to_dict()
    df["corridor_freq"] = df["corridor"].map(freq).fillna(0)

    # Target encoding for event_cause vs congestion severity
    if "congestion_severity" in df.columns:
        te = df.groupby("event_cause")["congestion_severity"].mean()
        df["cause_target_enc"] = df["event_cause"].map(te).fillna(te.mean())

    log.info("Categorical encoding complete")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  HISTORICAL / ROLLING FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("HISTORICAL FEATURES")

    df = df.sort_values("start_datetime").reset_index(drop=True)
    dt = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["_ts"] = dt

    # ── Events per zone in rolling windows ────────────────────────────────
    # Rolling count of events per zone over time windows
    for days in [1, 7, 30]:
        col = f"zone_events_{days}d"
        df[col] = 0.0
        for zone, grp_idx in df.groupby("zone").groups.items():
            grp = df.loc[grp_idx].copy()
            # Drop rows with NaT timestamps for rolling
            valid_mask = grp["start_datetime"].notna()
            grp_valid = grp[valid_mask].sort_values("start_datetime")
            if len(grp_valid) < 2:
                df.loc[grp_idx, col] = len(grp_valid)
                continue
            try:
                rolling = (grp_valid
                           .set_index("start_datetime")["zone_enc"]
                           .rolling(f"{days}D")
                           .count())
                # Map back by original index
                result_series = pd.Series(rolling.values, index=grp_valid.index)
                df.loc[result_series.index, col] = result_series.values
                # Fill NaT rows with zone mean
                nat_mask_global = df.index.isin(grp_idx) & df["start_datetime"].isna()
                if nat_mask_global.any():
                    df.loc[nat_mask_global, col] = result_series.mean()
            except Exception:
                df.loc[grp_idx, col] = len(grp_valid) / max(days, 1)

    # ── Previous event severity at same zone ──────────────────────────────
    if "congestion_severity" in df.columns:
        df["prev_severity_zone"] = (
            df.groupby("zone")["congestion_severity"]
              .shift(1)
              .fillna(df["congestion_severity"].median())
        )

        # Mean severity per zone
        zone_mean_sev = df.groupby("zone")["congestion_severity"].transform("mean")
        df["zone_mean_severity"] = zone_mean_sev

    # ── Average duration per zone/event_cause ─────────────────────────────
    if "duration_min" in df.columns:
        df["cause_zone_avg_dur"] = df.groupby(
            ["event_cause", "zone"])["duration_min"].transform("mean")
        df["station_avg_dur"] = df.groupby(
            "police_station")["duration_min"].transform("mean")

    df = df.drop(columns=["_ts", "_date"], errors="ignore")
    log.info("Historical features added")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  INTERACTION FEATURES
# ──────────────────────────────────────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    print_section("INTERACTION FEATURES")

    df["cause_x_peak"]         = df["event_cause_enc"] * df["is_peak"]
    df["cause_x_weekend"]      = df["event_cause_enc"] * df["is_weekend"]
    df["zone_x_priority"]      = df["zone_enc"] * df["priority_enc"]
    df["dist_x_priority"]      = df["dist_from_center_km"] * df["priority_enc"]
    df["cluster_x_cause"]      = df["spatial_cluster"] * df["event_cause_enc"]
    df["hour_x_event_type"]    = df["hour"] * df["event_type_enc"]
    df["closure_x_priority"]   = df["road_closure_enc"] * df["priority_enc"]

    if "congestion_severity" in df.columns:
        df["sev_x_cluster"]    = df["congestion_severity"] * df["cluster_density"]

    log.info("Interaction features added")
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  SELECT FINAL FEATURE SET
# ──────────────────────────────────────────────────────────────────────────────
TARGET_COLS = [
    "congestion_severity", "duration_min",
    "manpower_req", "barricade_req", "diversion_req"
]

FEATURE_COLS = [
    # Time
    "hour", "minute", "day", "weekday", "month", "week_of_year",
    "is_weekend", "is_peak", "is_night", "is_morning_peak", "is_evening_peak",
    "time_since_midnight", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos",
    # Spatial
    "latitude", "longitude",
    "dist_from_center_km", "spatial_cluster", "cluster_density", "has_junction",
    # Categorical encoded
    "zone_enc", "police_station_enc", "corridor_enc",
    "event_cause_enc", "veh_type_enc",
    "event_type_enc", "priority_enc", "road_closure_enc",
    "has_veh_no", "authenticated_enc",
    "corridor_freq", "cause_target_enc",
    # Historical
    "zone_events_1d", "zone_events_7d", "zone_events_30d",
    "prev_severity_zone", "zone_mean_severity",
    "cause_zone_avg_dur", "station_avg_dur",
    # Interactions
    "cause_x_peak", "cause_x_weekend", "zone_x_priority",
    "dist_x_priority", "cluster_x_cause", "hour_x_event_type",
    "closure_x_priority", "sev_x_cluster",
]


def run_feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (X, targets) DataFrames."""
    df = add_time_features(df)
    df = add_spatial_features(df)
    df = encode_categoricals(df)
    df = add_historical_features(df)
    df = add_interaction_features(df)

    # Only keep cols that exist
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    tgt_cols  = [c for c in TARGET_COLS  if c in df.columns]

    X = df[feat_cols].copy()
    Y = df[tgt_cols].copy()

    # Final NaN fill
    X = X.fillna(X.median(numeric_only=True))

    log.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    log.info(f"Targets available: {tgt_cols}")

    # Save
    X.to_parquet(DATA_PROC / "features.parquet",   index=False)
    Y.to_parquet(DATA_PROC / "targets.parquet",     index=False)
    df.to_parquet(DATA_PROC / "full_featured.parquet", index=False)

    return X, Y


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    df = run_preprocessing()
    X, Y = run_feature_engineering(df)
    print(f"\nX shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
