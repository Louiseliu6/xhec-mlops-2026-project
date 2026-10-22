"""Select on validation data, refit once, evaluate once on held-out test data."""

import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.modelling.evaluation import scores
from src.modelling.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    make_pipeline,
    read_training_data,
)
from src.modelling.utils import save_model

SEED = 42


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


def split_data(data: pd.DataFrame) -> tuple:
    """Use the same 60/20/20 split as the modelling notebook."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        features, target, test_size=0.20, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=SEED
    )
    return x_train, x_val, x_trainval, x_test, y_train, y_val, y_trainval, y_test


def candidate_models() -> dict:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=SEED, n_jobs=2
        ),
    }


def train_and_save(
    csv_path: Path,
    model_path: Path,
    tracking_dir: Path = Path("mlruns"),
    results_path: Path = Path("artifacts/step2_results.json"),
) -> dict:
    """Train candidates and save the chosen, fully fitted preprocessing/model pipeline."""
    data = read_training_data(csv_path)
    data_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    x_train, x_val, x_trainval, x_test, y_train, y_val, y_trainval, y_test = split_data(data)

    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("abalone_step2")

    provenance = git_metadata()
    candidates = candidate_models()
    validation = {}
    for name, estimator in candidates.items():
        model = make_pipeline(clone(estimator))
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"phase": "validation", **provenance})
            mlflow.log_params(
                {
                    "candidate": name,
                    "split_seed": SEED,
                    "data_sha256": data_sha256,
                    **estimator.get_params(),
                }
            )
            model.fit(x_train, y_train)
            metrics = scores(y_val, model.predict(x_val))
            mlflow.log_metrics({f"val_{key}": value for key, value in metrics.items()})
            validation[name] = metrics

    best_name = min(validation, key=lambda name: validation[name]["mae"])
    # Refit the chosen configuration on train + validation. Test stays untouched until now.
    final_model = make_pipeline(clone(candidates[best_name]))
    final_model.fit(x_trainval, y_trainval)
    test_metrics = scores(y_test, final_model.predict(x_test))
    baseline = DummyRegressor(strategy="mean").fit(x_trainval, y_trainval)
    baseline_metrics = scores(y_test, baseline.predict(x_test))

    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.set_tags({"phase": "final_test", **provenance})
        mlflow.log_params(
            {
                "selected_model": best_name,
                "split_seed": SEED,
                "data_sha256": data_sha256,
                **candidates[best_name].get_params(),
            }
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metrics(
            {f"baseline_test_{key}": value for key, value in baseline_metrics.items()}
        )
        mlflow.sklearn.log_model(final_model, "model")

    save_model(final_model, model_path)
    result = {
        "selected_model": best_name,
        "selected_validation_mae": validation[best_name]["mae"],
        "data_sha256": data_sha256,
        "split_seed": SEED,
        "rows": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "validation": validation,
        "test": test_metrics,
        "test_baseline": baseline_metrics,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
