"""Input validation and train-fitted preprocessing shared by training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_COLUMNS = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]
FEATURE_COLUMNS = ["Sex", *NUMERIC_COLUMNS]
TARGET_COLUMN = "Rings"


def validate_features(features: pd.DataFrame) -> pd.DataFrame:
    """Check the shared training/inference input contract before preprocessing."""
    missing = set(FEATURE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    validated = features[FEATURE_COLUMNS].copy()
    invalid_sex = validated["Sex"].notna() & ~validated["Sex"].isin({"M", "F", "I"})
    if invalid_sex.any():
        raise ValueError("Sex must be M, F or I")

    for column in NUMERIC_COLUMNS:
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must be numeric") from error
        values = validated[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"{column} must be finite and non-negative")
    return validated


def read_training_data(path: Path) -> pd.DataFrame:
    """Load the course CSV and reject an unexpected schema or invalid targets."""
    if not path.is_file():
        raise FileNotFoundError(f"Abalone CSV not found: {path}")
    data = pd.read_csv(path)
    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if list(data.columns) != expected:
        raise ValueError(f"Expected CSV columns in this order: {expected}")
    if data.empty:
        raise ValueError("The Abalone CSV is empty")
    if not data[TARGET_COLUMN].notna().all() or not data[TARGET_COLUMN].gt(0).all():
        raise ValueError("Rings must be present and positive")
    data[FEATURE_COLUMNS] = validate_features(data)
    return data


def clean_height(features: pd.DataFrame) -> pd.DataFrame:
    """Treat non-positive heights as missing before the fitted median imputer."""
    cleaned = features.copy()
    cleaned.loc[cleaned["Height"] <= 0, "Height"] = np.nan
    return cleaned


def make_pipeline(regressor: object) -> Pipeline:
    """Put every learned transformation inside the saved model pipeline."""
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS), ("categorical", categorical, ["Sex"])]
    )
    return Pipeline(
        [
            (
                "clean_height",
                FunctionTransformer(clean_height, validate=False, feature_names_out="one-to-one"),
            ),
            ("preprocess", preprocess),
            ("regressor", regressor),
        ]
    )
