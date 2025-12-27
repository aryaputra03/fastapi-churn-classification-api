"""
SQLAlchemy Database Models
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from src.api.database import Base

class PredictionLog(Base):
    """
    Table to store prediction logs
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    input_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<PredictionLog(id={self.id}, customer_id={self.customer_id}, prediction={self.prediction})>"
    
class ModelMetrics(Base):
    """
    Table to store model performance metrics
    """
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    confusion_matrix = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<ModelMetrics(version={self.model_version}, accuracy={self.accuracy})>"
    
class Customer(Base):
    """
    Table to store customer information
    """
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, unique=True, index=True, nullable=False)
    gender = Column(String)
    tenure = Column(Integer)
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    contract = Column(String)
    internet_service = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Customer(id={self.customer_id}, tenure={self.tenure})>"