import json
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from database import engine, Base, get_db
from models import TrafficEvent, ModelData
from pydantic import BaseModel

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ClearWay API", description="Traffic Impact Predictor Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "models" / "reports"

app.mount("/api/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

@app.get("/")
def read_root():
    return {"status": "ok", "app": "ClearWay API"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return Response(
        content='{"status":"healthy"}',
        media_type="application/json",
        status_code=200
    )

@app.get("/favicon.ico")
def get_favicon():
    return Response(content=b"", media_type="image/x-icon")

class PredictRequest(BaseModel):
    event_cause: str
    event_type: str
    priority: str
    road_closure: bool
    lat: float
    lon: float
    hour: int

@app.get("/api/kpis")
def get_kpis(db: Session = Depends(get_db)):
    total = db.query(TrafficEvent).count()
    if total == 0:
        return {"total_events": 0}
        
    severe_n = db.query(TrafficEvent).filter(TrafficEvent.congestion_severity == 3).count()
    closure_n = db.query(TrafficEvent).filter(TrafficEvent.requires_road_closure == True).count()
    closed_n = db.query(TrafficEvent).filter(TrafficEvent.status == "closed").count()
    
    avg_sev = db.query(func.avg(TrafficEvent.congestion_severity)).scalar() or 0
    from sqlalchemy.sql import text
    median_dur = db.execute(text("SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_min) FROM traffic_events")).scalar() or 0
    
    return {
        "total_events": total,
        "avg_severity": round(avg_sev, 2),
        "median_duration": round(median_dur, 0),
        "severe_events": severe_n,
        "road_closures": closure_n,
        "closed_pct": round((closed_n / total) * 100, 1) if total > 0 else 0
    }

@app.get("/api/charts/causes")
def get_causes(db: Session = Depends(get_db)):
    res = db.query(TrafficEvent.event_cause, func.count(TrafficEvent.id)).group_by(TrafficEvent.event_cause).order_by(func.count(TrafficEvent.id).desc()).limit(14).all()
    return [{"cause": r[0], "count": r[1]} for r in res]

@app.get("/api/charts/zones")
def get_zones(db: Session = Depends(get_db)):
    res = db.query(TrafficEvent.zone, func.avg(TrafficEvent.congestion_severity), func.count(TrafficEvent.id)).group_by(TrafficEvent.zone).all()
    data = [{"zone": r[0] if r[0] else "Unknown", "mean_sev": round(r[1] or 0, 2), "count": r[2]} for r in res]
    return sorted(data, key=lambda x: x["mean_sev"], reverse=True)

@app.get("/api/charts/durations")
def get_durations(db: Session = Depends(get_db)):
    events = db.query(TrafficEvent.duration_min).filter(TrafficEvent.duration_min != None).all()
    buckets = {}
    for (dur,) in events:
        b = min(int(dur // 30) * 30, 1440)
        buckets[b] = buckets.get(b, 0) + 1
    
    res = [{"duration": k, "count": v} for k, v in buckets.items()]
    return sorted(res, key=lambda x: x["duration"])

@app.get("/api/map")
def get_map(db: Session = Depends(get_db)):
    events = db.query(TrafficEvent.latitude, TrafficEvent.longitude, TrafficEvent.congestion_severity, TrafficEvent.event_cause, TrafficEvent.zone).filter(TrafficEvent.latitude != None).order_by(TrafficEvent.start_datetime.desc()).limit(1000).all()
    return [{"lat": e[0], "lon": e[1], "severity": e[2], "cause": e[3], "zone": e[4]} for e in events]

@app.get("/api/models/metrics")
def get_model_metrics(db: Session = Depends(get_db)):
    clf_record = db.query(ModelData).filter(ModelData.key == "classification_results").first()
    reg_record = db.query(ModelData).filter(ModelData.key == "regression_results").first()
    
    clf_results = clf_record.data if clf_record else {}
    reg_results = reg_record.data if reg_record else {}
    return {"classification": clf_results, "regression": reg_results}

@app.get("/api/features/importance")
def get_feature_importance(db: Session = Depends(get_db)):
    feat_record = db.query(ModelData).filter(ModelData.key == "feature_importance").first()
    if feat_record:
        return feat_record.data
    return []

@app.get("/api/history")
def get_history(skip: int = 0, limit: int = 500, db: Session = Depends(get_db)):
    events = db.query(TrafficEvent).order_by(TrafficEvent.start_datetime.desc()).offset(skip).limit(limit).all()
    return events

@app.post("/api/predict")
def predict_endpoint(req: PredictRequest, db: Session = Depends(get_db)):
    # Import the ML prediction logic
    import sys
    src_dir = str(BASE_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    from predict import predict_event
    
    ts = datetime.utcnow().replace(hour=req.hour)
    
    # Get intelligent prediction
    ml_result = predict_event(
        event_cause=req.event_cause,
        event_type=req.event_type,
        priority=req.priority,
        requires_road_closure=req.road_closure,
        latitude=req.lat,
        longitude=req.lon,
        timestamp=ts
    )
    
    # Extract values for database
    base_sev = ml_result["congestion_severity"]["class"]
    dur = int(ml_result["predicted_duration_min"])
    officers = ml_result["manpower"]["officers_required"]

    event = TrafficEvent(
        start_datetime=datetime.utcnow(),
        event_cause=req.event_cause,
        event_type=req.event_type,
        priority=req.priority,
        requires_road_closure=req.road_closure,
        latitude=req.lat,
        longitude=req.lon,
        congestion_severity=base_sev,
        duration_min=dur,
        manpower_req=officers,
        status="open"
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    # Attach event_id for the frontend
    ml_result["event_id"] = event.id

    return ml_result
