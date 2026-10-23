"""Turn a validated API request into a model prediction."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.modelling.predicting import predict
from src.web_service.lib.models import AbaloneFeatures, RingsPrediction


def run_inference(features: AbaloneFeatures, model: Pipeline) -> RingsPrediction:
    """Predict the ring count of one abalone with the training-time column names."""
    row = pd.DataFrame([features.dict(by_alias=True)])
    return RingsPrediction(rings=float(predict(model, row)[0]))
