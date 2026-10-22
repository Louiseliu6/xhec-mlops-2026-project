"""Load a trained pipeline and predict ring counts for new CSV rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.modelling.preprocessing import validate_features
from src.modelling.utils import load_model


def predict(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(validate_features(features))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Abalone ring counts from a CSV.")
    parser.add_argument("input_path", type=Path, help="CSV with the eight feature columns")
    parser.add_argument(
        "--model-path", type=Path, default=Path("src/web_service/local_objects/abalone_model.pkl")
    )
    args = parser.parse_args()
    features = pd.read_csv(args.input_path)
    model = load_model(args.model_path)
    print(json.dumps(predict(model, features).tolist()))


if __name__ == "__main__":
    main()
