"""Persistence for the fitted preprocessing and model pipeline."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Load only model files produced by this project; pickle files can execute code."""
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Run the training command first.")
    return joblib.load(path)
