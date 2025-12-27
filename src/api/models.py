"""
Pydantic Models for API Request/Response
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    customer_id: str = Field(..., examples="C12345")
    gender: str = Field(..., examples="Male")
    tenure: int = Field(..., ge=0, le=72, examples=24)
    monthly_charges: float = Field(..., ge=0, examples=70.5)
    total_charges: float = Field(..., ge=0, examples=1500.75)
    contract: str = Field(..., examples="One year")
    payment_method: str = Field(..., examples="Bank transfer")
    internet_service: str = Field(..., examples="Fiber optic")

    @field_validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female']
        if v not in allowed:
            raise ValueError(f"Gender must be one of {allowed}")
        return v
    
    @field_validator('contract')
    def validate_contract(cls, v):
        allowed = ['Month-to-month', 'One year', 'Two year']
        if v not in allowed:
            raise ValueError(f"Contract must be one of {allowed}")
        return v
    
    @field_validator('internet_service')
    def validate_internet_service(cls, v):
        allowed = ['DSL', 'Fiber optic', 'No']
        if v not in allowed:
            raise ValueError(f"Internet service must be one of {allowed}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "C12345",
                "gender": "Male",
                "tenure": 24,
                "monthly_charges": 75.5,
                "total_charges": 1810.0,
                "contract": "One year",
                "payment_method": "Bank transfer",
                "internet_service": "Fiber optic"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    customer_id: str
    prediction: int = Field(..., description="0: No churn, 1: Churn")
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    no_churn_probability: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

    class Config:
        json_schema_extra ={
            "example":{
                "customer_id": "C12345",
                "prediction": 1,
                "churn_probability": 0.75,
                "no_churn_probability": 0.25,
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    customers: List[PredictionRequest]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str 
    model_loaded: bool
    timestamp: datetime
    error: Optional[str] = None

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    model_version: str
    features: List[str]
    trained_at: Optional[str]
    accuracy: Optional[float]

class PredictionHistoryResponse(BaseModel):
    """Prediction history response"""
    id: int
    customer_id: str
    prediction: int
    probability: float
    created_at: datetime

    class Config:
        from_attributes = True 
