"""Serve a local Prefect deployment for daily Abalone retraining."""

import argparse
from pathlib import Path

from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments.runner import EntrypointType
from prefect.settings import PREFECT_API_URL

from src.modelling.flows import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_TRACKING_DIR,
    resolve_path,
    training_flow,
)

DEPLOYMENT_NAME = "daily-retraining"
DEFAULT_CRON = "0 2 * * *"


def serve_training(
    cron: str = DEFAULT_CRON,
    csv_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    results_path: Path = DEFAULT_RESULTS_PATH,
    experiment_name: str = "abalone_step3",
) -> None:
    """Register a UTC schedule and serve one run at a time on this process runner.

    Keep the server and this process running. Use only one serving process for
    these output paths; the runner limit is not a cross-process artifact lock.
    Graceful shutdown pauses the deployment's schedule.
    """
    if not PREFECT_API_URL.value():
        raise ValueError(
            "Set PREFECT_API_URL to your running Prefect server, "
            "for example http://127.0.0.1:4200/api"
        )
    schedule = CronSchedule(cron=cron, timezone="UTC")
    training_flow.serve(
        name=DEPLOYMENT_NAME,
        schedules=[{"schedule": schedule, "active": True}],
        paused=False,
        parameters={
            "csv_path": str(resolve_path(csv_path)),
            "model_path": str(resolve_path(model_path)),
            "tracking_dir": str(resolve_path(tracking_dir)),
            "results_path": str(resolve_path(results_path)),
            "experiment_name": experiment_name,
        },
        enforce_parameter_schema=True,
        entrypoint_type=EntrypointType.MODULE_PATH,
        limit=1,
        pause_on_shutdown=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve scheduled retraining. Requires a running Prefect server."
    )
    parser.add_argument(
        "--cron", default=DEFAULT_CRON, help="Cron schedule in UTC (default: 02:00 daily)"
    )
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--experiment-name", default="abalone_step3")
    args = parser.parse_args()
    serve_training(
        args.cron,
        args.csv_path,
        args.model_path,
        args.tracking_dir,
        args.results_path,
        args.experiment_name,
    )


if __name__ == "__main__":
    main()
