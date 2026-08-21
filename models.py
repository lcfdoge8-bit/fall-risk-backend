from __future__ import annotations
from datetime import UTC, datetime
from sqlalchemy import Column, DateTime, Float, Integer, String
from database import Base

# Database Model tailored for deep learning extraction
class PatientRecord(Base):
    __tablename__ = "patient_predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    # The 10 required deep learning features
    age = Column(Integer, nullable=False)
    night_bed_exits = Column(Integer, nullable=False)
    night_activity_duration_min = Column(Integer, nullable=False)
    past_falls = Column(Integer, nullable=False)
    mobility_score = Column(Integer, nullable=False)
    high_risk_medication = Column(Integer, nullable=False)
    cognitive_impairment = Column(Integer, nullable=False)
    polypharmacy_count = Column(Integer, nullable=False)
    orthostatic_hypotension = Column(Integer, nullable=False)
    tug_seconds = Column(Float, nullable=False)
    # Outcome tracking
    fall_risk_level = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)