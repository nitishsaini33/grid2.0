"""
utils.py -- Shared utilities for Traffic Impact Prediction System
"""
import sys, io
# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
REPORTS_DIR= BASE_DIR / "models" / "reports"

for p in [DATA_RAW, DATA_PROC, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

RAW_CSV = DATA_RAW / "Astram event data.csv"

# ─── Logging ──────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
                                datefmt="%H:%M:%S")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(BASE_DIR / "pipeline.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

log = get_logger("utils")

# ─── Model I/O ────────────────────────────────────────────────────────────────
def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    log.info(f"Saved model → {path}")

_model_cache = {}

def load_model(name: str):
    if name in _model_cache:
        return _model_cache[name]
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    _model_cache[name] = model
    return model

def save_json(data: dict, name: str):
    path = REPORTS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log.info(f"Saved report → {path}")

def load_json(name: str) -> dict:
    path = REPORTS_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# ─── Severity Mapping ─────────────────────────────────────────────────────────
SEVERITY_LABELS = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
MANPOWER_MAP    = {0: 2, 1: 4, 2: 8, 3: 12, 4: 20}
BARRICADE_MAP   = {0: 0, 1: 2, 2: 5, 3: 10}
BARRICADE_LABELS= {0: "None", 1: "Light", 2: "Medium", 3: "Heavy"}
DIVERSION_LABELS= {0: "No Diversion", 1: "Diversion Required"}

# Event cause → base severity score
CAUSE_SEVERITY = {
    "vehicle_breakdown": 1,
    "pot_holes":         1,
    "others":            0,
    "road_conditions":   0,
    "Debris":            1,
    "debris":            1,
    "Fog / Low Visibility": 1,
    "tree_fall":         1,
    "water_logging":     1,
    "construction":      2,
    "accident":          2,
    "public_event":      2,
    "vip_movement":      2,
    "congestion":        2,
    "procession":        2,
    "protest":           2,
    "test_demo":         0,
}

# Event causes that typically need diversion
DIVERSION_CAUSES = {
    "congestion", "construction", "vip_movement",
    "procession", "protest", "public_event", "accident"
}

# ─── Print helpers ────────────────────────────────────────────────────────────
def print_section(title: str):
    width = 70
    sep = "=" * width
    print("\n" + sep)
    print(f"  {title}")
    print(sep)

def print_metrics(metrics: dict, title: str = "Metrics"):
    print(f"\n  [{title}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:35s}: {v:.4f}")
        else:
            print(f"    {k:35s}: {v}")
