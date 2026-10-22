"""Metrics for continuous ring-count predictions."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def scores(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }
