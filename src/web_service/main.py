"""FastAPI service predicting Abalone ring counts from physical measurements."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import AbaloneFeatures, RingsPrediction
from src.web_service.utils import DEFAULT_MODEL_PATH, load_object


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; MODEL_PATH overrides the training output path."""
    app.state.model = load_object(Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)))
    yield


app = FastAPI(
    title="Abalone ring-count prediction",
    description="Predict the number of shell rings of an abalone from its measurements.",
    lifespan=lifespan,
)


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


@app.post("/predict", response_model=RingsPrediction, status_code=201)
def predict(payload: AbaloneFeatures, request: Request) -> RingsPrediction:
    return run_inference(payload, request.app.state.model)
