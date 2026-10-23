"""Check deployment scheduling, parameters and local runner safety settings."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prefect.deployments.runner import EntrypointType
from prefect.settings import PREFECT_API_URL, temporary_settings

from src.modelling.deploy import serve_training
from src.modelling.flows import training_flow


class DeploymentTest(unittest.TestCase):
    def test_serves_daily_utc_with_single_run_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with (
                temporary_settings({PREFECT_API_URL: "http://127.0.0.1:4200/api"}),
                patch("src.modelling.flows.PROJECT_ROOT", root),
                patch.object(training_flow, "serve") as serve,
            ):
                serve_training()
            serve.assert_called_once()
            options = serve.call_args.kwargs
            self.assertEqual(options["name"], "daily-retraining")
            self.assertEqual(options["limit"], 1)
            self.assertTrue(options["pause_on_shutdown"])
            self.assertTrue(options["enforce_parameter_schema"])
            self.assertFalse(options["paused"])
            self.assertEqual(options["entrypoint_type"], EntrypointType.MODULE_PATH)
            self.assertEqual(len(options["schedules"]), 1)
            schedule = options["schedules"][0]
            self.assertTrue(schedule["active"])
            self.assertEqual(schedule["schedule"].cron, "0 2 * * *")
            self.assertEqual(schedule["schedule"].timezone, "UTC")
            parameters = options["parameters"]
            self.assertEqual(json.loads(json.dumps(parameters)), parameters)
            self.assertEqual(parameters["csv_path"], str((root / "data/abalone.csv").resolve()))
            self.assertEqual(parameters["experiment_name"], "abalone_step3")
            for key in ("model_path", "tracking_dir", "results_path"):
                self.assertTrue(Path(parameters[key]).is_absolute())

    def test_custom_schedule_and_parameters(self) -> None:
        # Resolve so the expected absolute paths also carry a drive letter on Windows.
        custom = Path("/custom").resolve()
        with (
            temporary_settings({PREFECT_API_URL: "http://127.0.0.1:4200/api"}),
            patch.object(training_flow, "serve") as serve,
        ):
            serve_training(
                cron="0 6 * * 1",
                csv_path=custom / "data.csv",
                model_path=custom / "model.pkl",
                tracking_dir=custom / "mlruns",
                results_path=custom / "results.json",
                experiment_name="custom-experiment",
            )
        options = serve.call_args.kwargs
        self.assertEqual(options["schedules"][0]["schedule"].cron, "0 6 * * 1")
        self.assertEqual(options["schedules"][0]["schedule"].timezone, "UTC")
        self.assertEqual(
            options["parameters"],
            {
                "csv_path": str(custom / "data.csv"),
                "model_path": str(custom / "model.pkl"),
                "tracking_dir": str(custom / "mlruns"),
                "results_path": str(custom / "results.json"),
                "experiment_name": "custom-experiment",
            },
        )

    def test_requires_explicit_server(self) -> None:
        with (
            temporary_settings({PREFECT_API_URL: None}),
            patch.object(training_flow, "serve") as serve,
        ):
            with self.assertRaisesRegex(ValueError, "Set PREFECT_API_URL"):
                serve_training()
            serve.assert_not_called()

    def test_rejects_invalid_schedule_before_serving(self) -> None:
        with (
            temporary_settings({PREFECT_API_URL: "http://127.0.0.1:4200/api"}),
            patch.object(training_flow, "serve") as serve,
        ):
            with self.assertRaises(ValueError):
                serve_training(cron="invalid")
            serve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
