"""Prefect orchestration for the reusable Abalone training stages."""

import argparse
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner

from src.modelling.training import (
    candidate_models,
    prepare_training_data,
    refit_and_evaluate,
    save_training_results,
    select_best_model,
    train_candidate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = Path("data/abalone.csv")
DEFAULT_MODEL_PATH = Path("src/web_service/local_objects/abalone_model.pkl")
DEFAULT_TRACKING_DIR = Path("mlruns")
DEFAULT_RESULTS_PATH = Path("artifacts/step3_results.json")

# Keep the training functions usable without Prefect. Do not cache stages that log
# MLflow runs or write artifacts: every retraining run must execute them again.
prepare_data_task = task(name="prepare-training-data", persist_result=False)(prepare_training_data)
train_candidate_task = task(
    name="train-candidate", task_run_name="train-{name}", persist_result=False
)(train_candidate)
select_model_task = task(name="select-best-model", persist_result=False)(select_best_model)
refit_task = task(name="refit-and-evaluate", persist_result=False)(refit_and_evaluate)
save_results_task = task(name="save-training-results", persist_result=False)(save_training_results)


def resolve_path(path: Path) -> Path:
    """Interpret relative paths against the checkout, not the process working directory."""
    return (PROJECT_ROOT / path).resolve()


@flow(name="abalone-training", task_runner=SequentialTaskRunner(), persist_result=False)
def training_flow(
    csv_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    results_path: Path = DEFAULT_RESULTS_PATH,
    experiment_name: str = "abalone_step3",
) -> dict:
    """Train and save the model; relative input/output paths use the project root."""
    csv_path = resolve_path(csv_path)
    model_path = resolve_path(model_path)
    tracking_dir = resolve_path(tracking_dir)
    results_path = resolve_path(results_path)

    data = prepare_data_task(csv_path)
    candidates = candidate_models()
    # MLflow uses process-global tracking state, so candidate tasks run sequentially.
    validation = {
        name: train_candidate_task(name, estimator, data, tracking_dir, experiment_name)
        for name, estimator in candidates.items()
    }
    best_name = select_model_task(validation)
    model, result = refit_task(
        best_name, candidates[best_name], data, validation, tracking_dir, experiment_name
    )
    result = save_results_task(model, result, model_path, results_path)
    get_run_logger().info(
        "Selected %s; held-out test MAE %.3f rings; saved pipeline to %s",
        best_name,
        result["test"]["mae"],
        model_path,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Prefect training. Relative paths use the project root."
    )
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--experiment-name", default="abalone_step3")
    args = parser.parse_args()
    training_flow(
        args.csv_path, args.model_path, args.tracking_dir, args.results_path, args.experiment_name
    )


if __name__ == "__main__":
    main()
