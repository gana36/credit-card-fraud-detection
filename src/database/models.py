"""
Database models for storing predictions
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class Prediction(Base):
    """Model for storing prediction logs"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    features = Column(JSON)  # Input features
    fraud_probability = Column(Float)
    prediction = Column(Integer)  # 0 or 1
    model_version = Column(String, index=True)
    model_name = Column(String)
    latency_ms = Column(Float)

    def __repr__(self):
        return f"<Prediction(id={self.id}, timestamp={self.timestamp}, fraud_prob={self.fraud_probability})>"


# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://mlops:mlops_password@postgres:5432/predictions"
)

# Create engine
engine = create_engine(DATABASE_URL)

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
