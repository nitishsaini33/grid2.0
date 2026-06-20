import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import engine, Base, SessionLocal
from models import TrafficEvent, ModelData
import json
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PROC = BASE_DIR / "data" / "processed"

def migrate():
    print("Creating database tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    parquet_path = DATA_PROC / "full_featured.parquet"
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}")
        return
    
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Fill na values to avoid DB insert errors
    df = df.replace({float('nan'): None})
    if 'start_datetime' in df.columns:
        df['start_datetime'] = df['start_datetime'].apply(lambda x: None if pd.isna(x) else x)
    
    print("Inserting records into database...")
    db: Session = SessionLocal()
    
    count = 0
    for _, row in df.iterrows():
        try:
            start_dt = row.get("start_datetime")
            if pd.isna(start_dt):
                start_dt = None
                
            event = TrafficEvent(
                start_datetime=start_dt,
                event_cause=row.get("event_cause"),
                event_type=row.get("event_type"),
                priority=row.get("priority"),
                zone=row.get("zone"),
                police_station=row.get("police_station"),
                requires_road_closure=row.get("requires_road_closure"),
                latitude=row.get("latitude"),
                longitude=row.get("longitude"),
                congestion_severity=row.get("congestion_severity"),
                duration_min=row.get("duration_min"),
                manpower_req=row.get("manpower_req"),
                barricade_req=row.get("barricade_req"),
                diversion_req=row.get("diversion_req"),
                status=row.get("status", "closed")
            )
            db.add(event)
            count += 1
            if count % 1000 == 0:
                db.commit()
                print(f"Inserted {count} records...")
        except Exception as e:
            print(f"Error on row: {e}")
            db.rollback()
            continue
            
    db.commit()

    print("Migrating model data to database...")
    REPORTS_DIR = BASE_DIR / "models" / "reports"
    
    # Classification results
    clf_path = REPORTS_DIR / "classification_results.json"
    if clf_path.exists():
        with open(clf_path) as f:
            clf_data = json.load(f)
            db.add(ModelData(key="classification_results", data=clf_data))
    
    # Regression results
    reg_path = REPORTS_DIR / "regression_results.json"
    if reg_path.exists():
        with open(reg_path) as f:
            reg_data = json.load(f)
            db.add(ModelData(key="regression_results", data=reg_data))
            
    # Feature importance
    try:
        model = joblib.load(BASE_DIR / "models" / "saved" / "clf_congestion_severity.joblib")
        imp = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "estimators_"):
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    imp = est.feature_importances_
                    break
        if imp is not None:
            feat_names = [f"f{i}" for i in range(len(imp))]
            idx = np.argsort(imp)[-20:]
            res = [{"feature": feat_names[i], "importance": float(imp[i])} for i in idx]
            db.add(ModelData(key="feature_importance", data=res))
    except Exception as e:
        print(f"Failed to load feature importance: {e}")

    db.commit()
    db.close()
    print(f"Migration completed. Total records inserted: {count}")

if __name__ == "__main__":
    migrate()
