from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON
from database import Base

class TrafficEvent(Base):
    __tablename__ = "traffic_events"

    id = Column(Integer, primary_key=True, index=True)
    start_datetime = Column(DateTime, index=True)
    event_cause = Column(String, index=True)
    event_type = Column(String)
    priority = Column(String)
    zone = Column(String, index=True)
    police_station = Column(String)
    requires_road_closure = Column(Boolean)
    latitude = Column(Float)
    longitude = Column(Float)
    congestion_severity = Column(Integer)
    duration_min = Column(Integer)
    manpower_req = Column(Integer)
    barricade_req = Column(Integer)
    diversion_req = Column(Integer)
    status = Column(String, default="open")

class ModelData(Base):
    __tablename__ = "model_data"

    key = Column(String, primary_key=True, index=True)
    data = Column(JSON)
