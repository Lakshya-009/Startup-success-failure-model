
"""
SFIP - Startup ML Service
File: ml_service.py

Provides REST APIs for:
1. ML model health
2. Startup success/failure scoring
3. Historical startup similarity

The trained model is loaded once during application startup.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from startup_model import StartupRiskModel


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Application lifecycle
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading SFIP startup prediction model...")

    try:
        app.state.model = StartupRiskModel.load()
        logger.info("SFIP ML model loaded successfully.")

    except Exception:
        logger.exception("Failed to load SFIP ML model.")
        app.state.model = None

    yield

    app.state.model = None
    logger.info("SFIP ML service stopped.")


app = FastAPI(
    title="SFIP Startup Intelligence ML API",
    description="Startup success/failure scoring and historical startup similarity.",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------------------------------------------------
# Request schemas
# -------------------------------------------------------------------

class StartupPredictionRequest(BaseModel):
    funding_usd_millions: Any = None
    country: str | None = None
    industry: str | None = None


class SimilarStartupsRequest(BaseModel):
    text: str = Field(min_length=1)
    industry: str | None = None
    k_each: int = Field(default=4, ge=1, le=10)


# -------------------------------------------------------------------
# Model access
# -------------------------------------------------------------------

def get_model(request: Request) -> StartupRiskModel:

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is currently unavailable."
        )

    return model


# -------------------------------------------------------------------
# Root endpoint
# -------------------------------------------------------------------

@app.get("/")
def home():

    return {
        "service": "SFIP Startup Intelligence ML API",
        "message": "ML service is running.",
        "documentation": "/docs"
    }


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

@app.get("/health")
def health(request: Request):

    model = getattr(request.app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded."
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model.card.get("model_version")
    }


# -------------------------------------------------------------------
# Startup prediction endpoint
# -------------------------------------------------------------------

@app.post("/predict")
def predict_startup(
    payload: StartupPredictionRequest,
    request: Request
):

    model = get_model(request)

    try:
        result = model.predict(payload.model_dump())

        return {
            "success": True,
            "analysis_source": "ML",
            "result": result
        }

    except Exception:
        logger.exception("Startup prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Startup prediction could not be completed."
        )


# -------------------------------------------------------------------
# Similar startups endpoint
# -------------------------------------------------------------------

@app.post("/similar-startups")
def find_similar_startups(
    payload: SimilarStartupsRequest,
    request: Request
):

    model = get_model(request)

    try:
        result = model.similar_startups(
            text=payload.text,
            industry=payload.industry,
            k_each=payload.k_each
        )

        return {
            "success": True,
            "analysis_source": "ML",
            "result": result
        }

    except Exception:
        logger.exception("Similar startup search failed.")

        raise HTTPException(
            status_code=500,
            detail="Similar startup search could not be completed."
        )
