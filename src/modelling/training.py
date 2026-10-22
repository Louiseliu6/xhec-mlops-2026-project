"""Select on validation data, refit once, evaluate once on held-out test data."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modelling.evaluation import scores
from src.modelling.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_pipeline,
    read_training_data,
)
from src.modelling.utils import save_model

SEED = 42


class DataSplits(NamedTuple):
    """Named partitions, retaining the original split_data tuple interface."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_trainval: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_trainval: pd.Series
    y_test: pd.Series


@dataclass
class TrainingData:
    """Validated partitions and provenance shared by the training stages."""

    splits: DataSplits
    data_sha256: str
    provenance: dict[str, str]


def git_metadata() -> dict[str, str]:
    """Record source provenance without requiring a Git checkout to train."""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "worktree_dirty": "unknown"}
    return {"git_commit": commit, "worktree_dirty": str(bool(status))}


def split_data(data: pd.DataFrame) -> DataSplits:
    """Use the same 60/20/20 split as the modelling notebook."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features, target, test_size=0.20, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=SEED
    )
    return DataSplits(x_train, x_val, x_trainval, x_test, y_train, y_val, y_trainval, y_test)


def candidate_models() -> dict[str, BaseEstimator]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=SEED, n_jobs=2
        ),
    }


def prepare_training_data(csv_path: Path) -> TrainingData:
    """Validate the CSV and prepare deterministic partitions and provenance."""
    data = read_training_data(csv_path)
    return TrainingData(
        splits=split_data(data),
        data_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        provenance=git_metadata(),
    )


def configure_tracking(tracking_dir: Path, experiment_name: str) -> None:
    """Set tracking explicitly in each stage that writes MLflow runs."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)


def train_candidate(
    name: str,
    estimator: BaseEstimator,
    data: TrainingData,
    tracking_dir: Path,
    experiment_name: str = "abalone_step2",
) -> dict[str, float]:
    """Fit one candidate on training data and log its validation metrics only."""
    configure_tracking(tracking_dir, experiment_name)
    model = make_pipeline(clone(estimator))
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"phase": "validation", **data.provenance})
        mlflow.log_params(
            {
                "candidate": name,
                "split_seed": SEED,
                "data_sha256": data.data_sha256,
                **estimator.get_params(),
            }
        )
        model.fit(data.splits.x_train, data.splits.y_train)
        metrics = scores(data.splits.y_val, model.predict(data.splits.x_val))
        mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
    return metrics


def select_best_model(validation: dict[str, dict[str, float]]) -> str:
    """Select exclusively by validation MAE; held-out test metrics are not inputs."""
    return min(validation, key=lambda name: validation[name]["mae"])


def refit_and_evaluate(
    best_name: str,
    estimator: BaseEstimator,
    data: TrainingData,
    validation: dict[str, dict[str, float]],
    tracking_dir: Path,
    experiment_name: str = "abalone_step2",
) -> tuple[Pipeline, dict]:
    """Refit the selected configuration and evaluate once on the held-out test set."""
    configure_tracking(tracking_dir, experiment_name)
    splits = data.splits
    # Refit the chosen configuration on train + validation. Test stays untouched until now.
    final_model = make_pipeline(clone(estimator))
    final_model.fit(splits.x_trainval, splits.y_trainval)
    test_metrics = scores(splits.y_test, final_model.predict(splits.x_test))
    baseline = DummyRegressor(strategy="mean").fit(splits.x_trainval, splits.y_trainval)
    baseline_metrics = scores(splits.y_test, baseline.predict(splits.x_test))

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.set_tags({"phase": "final_test", **data.provenance})
        mlflow.log_params(
            {
                "selected_model": best_name,
                "split_seed": SEED,
                "data_sha256": data.data_sha256,
                **estimator.get_params(),
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics(
            {f"baseline_test_{key}": value for key, value in baseline_metrics.items()}
        )
        mlflow.sklearn.log_model(final_model, "model")

    result = {
        "selected_model": best_name,
        "selected_validation_mae": validation[best_name]["mae"],
        "data_sha256": data.data_sha256,
        "split_seed": SEED,
        "rows": {
            "train": len(splits.x_train),
            "validation": len(splits.x_val),
            "test": len(splits.x_test),
        },
        "validation": validation,
        "test": test_metrics,
        "test_baseline": baseline_metrics,
    }
    return final_model, result


def save_training_results(
    model: Pipeline, result: dict, model_path: Path, results_path: Path
) -> dict:
    """Persist the full fitted pipeline and its metrics for downstream inference."""
    save_model(model, model_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_and_save(
    csv_path: Path,
    model_path: Path,
    tracking_dir: Path = Path("mlruns"),
    results_path: Path = Path("artifacts/step2_results.json"),
) -> dict:
    """Train candidates and save the chosen, fully fitted preprocessing/model pipeline."""
    data = prepare_training_data(csv_path)
    candidates = candidate_models()
    validation = {
        name: train_candidate(name, estimator, data, tracking_dir)
        for name, estimator in candidates.items()
    }
    best_name = select_best_model(validation)
    model, result = refit_and_evaluate(
        best_name, candidates[best_name], data, validation, tracking_dir
    )
    return save_training_results(model, result, model_path, results_path)
