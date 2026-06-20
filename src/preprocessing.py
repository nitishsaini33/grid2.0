"""
preprocessing.py — Data Cleaning, Target Engineering & Imputation
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    get_logger, RAW_CSV, DATA_PROC,
    CAUSE_SEVERITY, DIVERSION_CAUSES,
    SEVERITY_LABELS, print_section
)

log = get_logger("preprocessing")


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════════════════════
def load_raw() -> pd.DataFrame:
    print_section("LOADING RAW DATA")
    df = pd.read_csv(RAW_CSV, encoding="utf-8", on_bad_lines="skip")
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATETIME PARSING
# ══════════════════════════════════════════════════════════════════════════════
DT_COLS = ["start_datetime", "end_datetime", "closed_datetime",
           "resolved_datetime", "modified_datetime", "created_date"]

def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    print_section("PARSING DATETIMES")
    for col in DT_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df[col] = df[col].dt.tz_localize(None)   # strip tz → naive UTC
    log.info("Datetime columns parsed")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DURATION DERIVATION
# ══════════════════════════════════════════════════════════════════════════════
def derive_duration(df: pd.DataFrame) -> pd.DataFrame:
    print_section("DERIVING DURATION")

    def _diff_min(a, b):
        return (b - a).dt.total_seconds() / 60

    # Priority: closed > resolved > end
    df["duration_min"] = np.nan

    mask_closed   = df["start_datetime"].notna() & df["closed_datetime"].notna()
    mask_resolved = df["start_datetime"].notna() & df["resolved_datetime"].notna()
    mask_end      = df["start_datetime"].notna() & df["end_datetime"].notna()

    df.loc[mask_closed,   "duration_min"] = _diff_min(df["start_datetime"], df["closed_datetime"])
    df.loc[mask_resolved & ~mask_closed, "duration_min"] = \
        _diff_min(df["start_datetime"], df["resolved_datetime"])
    df.loc[mask_end & ~mask_closed & ~mask_resolved, "duration_min"] = \
        _diff_min(df["start_datetime"], df["end_datetime"])

    # Clip negative to 0, clip max at 99th percentile
    df["duration_min"] = df["duration_min"].clip(lower=0)
    cap = df["duration_min"].quantile(0.99)
    df["duration_min"] = df["duration_min"].clip(upper=cap)

    n = df["duration_min"].notna().sum()
    log.info(f"Duration derived for {n:,}/{len(df):,} rows  (cap={cap:.1f} min)")
    log.info(f"Duration stats (min):\n{df['duration_min'].describe().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    print_section("BUILDING TARGETS")

    cause = df["event_cause"].fillna("others").str.lower().str.strip()
    # normalise debris variants
    cause = cause.replace({"debris": "debris_", "Debris": "debris_"})

    # ── Target 1: congestion_severity ─────────────────────────────────────
    sev_score = cause.map(CAUSE_SEVERITY).fillna(0).astype(int)

    # priority modifier
    prio_mod = (df["priority"].fillna("Low").str.strip() == "High").astype(int)

    # road closure → auto bump to 3
    closure = df["requires_road_closure"].fillna(False).astype(bool)

    # unplanned event → +1 (up to 3)
    unplanned = (df["event_type"].fillna("unplanned") == "unplanned").astype(int)

    # duration modifier: long events → +1
    dur_mod = (df["duration_min"].fillna(0) > 120).astype(int)

    raw_score = sev_score + prio_mod + unplanned * 0.5 + dur_mod * 0.5
    raw_score = raw_score.clip(0, 4)

    sev = pd.cut(raw_score,
                 bins=[-0.1, 0.9, 1.9, 2.9, 10],
                 labels=[0, 1, 2, 3]).astype(int)

    df["congestion_severity"] = np.where(closure, 3, sev)

    print("  congestion_severity dist:\n",
          df["congestion_severity"].value_counts().sort_index().to_frame().to_string())

    # ── Target 2: duration_min (already built) ────────────────────────────
    # Fill remaining NaNs with median per (event_cause, priority)
    med = df.groupby(["event_cause", "priority"])["duration_min"].transform("median")
    df["duration_min"] = df["duration_min"].fillna(med)
    df["duration_min"] = df["duration_min"].fillna(df["duration_min"].median())
    log.info("duration_min NaNs filled with group medians")

    # ── Target 3: manpower_req ────────────────────────────────────────────
    # Maps severity (0-3) → manpower class (0-4)
    manpower_rule = {0: 0, 1: 1, 2: 2, 3: 3}
    df["manpower_req"] = df["congestion_severity"].map(manpower_rule)
    # Bump to class 4 (20 officers) when road closure + Severe
    df.loc[closure & (df["congestion_severity"] == 3), "manpower_req"] = 4
    print("  manpower_req dist:\n",
          df["manpower_req"].value_counts().sort_index().to_frame().to_string())

    # ── Target 4: barricade_req ───────────────────────────────────────────
    bar = df["congestion_severity"].copy()
    bar = np.where(closure & bar >= 2, 3, bar)               # Heavy if closure
    bar = np.where(~closure & (df["congestion_severity"] == 3), 2, bar)  # Medium
    df["barricade_req"] = bar.astype(int).clip(0, 3)
    print("  barricade_req dist:\n",
          df["barricade_req"].value_counts().sort_index().to_frame().to_string())

    # ── Target 5: diversion_req ───────────────────────────────────────────
    div_cause = df["event_cause"].isin(DIVERSION_CAUSES)
    df["diversion_req"] = ((closure | (div_cause & prio_mod.astype(bool)))).astype(int)
    print("  diversion_req dist:\n",
          df["diversion_req"].value_counts().to_frame().to_string())

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DROP USELESS COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
DROP_COLS = [
    "map_file", "comment", "meta_data", "direction",
    "route_path", "citizen_accident_id", "assigned_to_police_id",
    "resolved_by_id", "closed_by_id", "last_modified_by_id",
    "created_by_id", "id", "description",
    "resolved_at_address", "gba_identifier",
    "cargo_material", "age_of_truck",
]

def drop_useless(df: pd.DataFrame) -> pd.DataFrame:
    print_section("DROPPING USELESS COLUMNS")
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info(f"Dropped {len(to_drop)} columns → {df.shape[1]} remaining")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLEAN & IMPUTE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    print_section("CLEANING & IMPUTING FEATURES")
    rows_before = len(df)

    # ── Fill categoricals ─────────────────────────────────────────────────
    df["corridor"]         = df["corridor"].fillna("Unknown")
    df["veh_type"]         = df["veh_type"].fillna("Unknown")
    df["veh_no"]           = df["veh_no"].fillna("Unknown")
    df["reason_breakdown"] = df["reason_breakdown"].fillna("None")
    df["kgid"]             = df["kgid"].fillna("Unknown")
    df["junction"]         = df["junction"].fillna("None")

    # ── Impute zone from spatial clusters ─────────────────────────────────
    zone_mask = df["zone"].isna()
    if zone_mask.any():
        from sklearn.cluster import KMeans
        coords = df[["latitude", "longitude"]].copy()
        known_zones = df.loc[~zone_mask, ["latitude", "longitude", "zone"]].copy()
        km = KMeans(n_clusters=10, random_state=42, n_init=10)
        km.fit(known_zones[["latitude", "longitude"]])
        cluster_to_zone = {}
        known_zones["_clust"] = km.predict(known_zones[["latitude", "longitude"]])
        cluster_to_zone = known_zones.groupby("_clust")["zone"].agg(
            lambda x: x.mode()[0]).to_dict()
        df.loc[zone_mask, "zone"] = km.predict(
            df.loc[zone_mask, ["latitude", "longitude"]]).tolist()
        df.loc[zone_mask, "zone"] = df.loc[zone_mask, "zone"].map(cluster_to_zone)
    df["zone"] = df["zone"].fillna("Unknown")

    # ── Impute police_station ─────────────────────────────────────────────
    df["police_station"] = df["police_station"].fillna("Unknown")

    # ── Impute numeric coords via simple mean ─────────────────────────────
    for col in ["endlatitude", "endlongitude",
                "resolved_at_latitude", "resolved_at_longitude"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── IQR outlier clipping for spatial coords ───────────────────────────
    for col in ["latitude", "longitude"]:
        Q1, Q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(Q1, Q3)

    rows_after = len(df)
    pct = rows_after / rows_before * 100
    log.info(f"Rows before cleaning: {rows_before:,}")
    log.info(f"Rows after  cleaning: {rows_after:,}  ({pct:.1f}% retained)")
    assert pct >= 90, f"Retention too low: {pct:.1f}%"
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PREPROCESSING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing() -> pd.DataFrame:
    df = load_raw()
    df = parse_datetimes(df)
    df = derive_duration(df)
    df = build_targets(df)
    df = drop_useless(df)
    df = clean_and_impute(df)

    out_path = DATA_PROC / "cleaned.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved cleaned data → {out_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    run_preprocessing()
