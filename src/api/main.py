"""
FastAPI Main Application

REST API for customer churn prediction with database logging.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import io

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionHistroyResponse,
)

from src.api.database import get_db, engine
from src.api import crud, schemas
from src.api.ml_service import MLService
from src.utils import logger
from sqlalchemy.orm import Session

from src.api.database import Base
# Base.metadata.create_all(bind=engine)

# app = FastAPI(
#     title="Churn Prediciton API",
#     description="ML API for customer churn prediction with tracking",
#     version="1.0.0",
#     docs_url="/docs",
#     redocs_url="/redocs",        
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credetionals=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ml_service = MLService()

# @app.get("/", response_model=dict)
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Churn Prediction API",
#         "version": "1.0.0",
#         "docs": "/docs",
#         "health": "/health"
#     }

# @app.get("/health", )