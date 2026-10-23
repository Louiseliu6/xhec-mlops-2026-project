"""Loading of the trained objects served by the API."""

from pathlib import Path

from sklearn.pipeline import Pipeline

from src.modelling.utils import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "src/web_service/local_objects/abalone_model.pkl"


def load_object(path: Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load the fitted preprocessing and model pipeline saved by training."""
    return load_model(path)
