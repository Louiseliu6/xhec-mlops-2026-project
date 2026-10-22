"""Exercise real Prefect task states against an isolated local test database."""

import asyncio
import json
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import TaskRunFilter, TaskRunFilterFlowRunId
from prefect.client.schemas.objects import TaskRun
from prefect.settings import (
    PREFECT_HOME,
    PREFECT_LOGGING_TO_API_ENABLED,
    PREFECT_SERVER_ANALYTICS_ENABLED,
    temporary_settings,
)
from prefect.states import State
from prefect.testing.utilities import prefect_test_harness

from src.modelling.flows import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_PATH,
    training_flow,
)
from src.modelling.predicting import predict
from src.modelling.preprocessing import NUMERIC_COLUMNS
from src.modelling.training import train_and_save
from src.modelling.utils import load_model


class TrainingFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        home = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.enterClassContext(
            temporary_settings(
                {
                    PREFECT_HOME: Path(home),
                    PREFECT_SERVER_ANALYTICS_ENABLED: False,
                    PREFECT_LOGGING_TO_API_ENABLED: False,
                }
            )
        )
        cls.enterClassContext(prefect_test_harness())

    def task_runs(self, state: State) -> list[TaskRun]:
        async def read_runs() -> list[TaskRun]:
            async with get_client() as client:
                return await client.read_task_runs(
                    task_run_filter=TaskRunFilter(
                        flow_run_id=TaskRunFilterFlowRunId(any_=[state.state_details.flow_run_id])
                    )
                )

        return asyncio.run(read_runs())

    def test_flow_matches_plain_training_outside_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rng = np.random.default_rng(42)
            features = pd.DataFrame(rng.uniform(0.1, 1, (90, 7)), columns=NUMERIC_COLUMNS)
            features.insert(0, "Sex", np.resize(["M", "F", "I"], len(features)))
            features.loc[0, "Height"] = 0
            features["Rings"] = rng.integers(5, 20, len(features))
            (root / "data").mkdir()
            csv_path = root / "data/abalone.csv"
            features.to_csv(csv_path, index=False)
            outside = root / "outside"
            outside.mkdir()

            with patch("src.modelling.flows.PROJECT_ROOT", root), chdir(outside):
                state = training_flow(return_state=True)
            self.assertTrue(state.is_completed())
            result = state.result()
            self.assertEqual(json.loads((root / DEFAULT_RESULTS_PATH).read_text()), result)
            self.assertEqual(list(outside.iterdir()), [])

            runs = self.task_runs(state)
            self.assertEqual(len(runs), 7)
            self.assertTrue(all(run.state.is_completed() for run in runs))
            self.assertEqual(
                {run.name for run in runs if run.name.startswith("train-")},
                {"train-mean_baseline", "train-linear_regression", "train-random_forest"},
            )
            client = MlflowClient(tracking_uri=(root / "mlruns").as_uri())
            experiment = client.get_experiment_by_name("abalone_step3")
            self.assertEqual(len(client.search_runs([experiment.experiment_id])), 4)

            expected = train_and_save(
                csv_path, root / "plain.pkl", root / "mlruns", root / "plain.json"
            )
            for key in ("selected_model", "data_sha256", "split_seed", "rows"):
                self.assertEqual(result[key], expected[key])
            self.assertAlmostEqual(
                result["selected_validation_mae"], expected["selected_validation_mae"], places=12
            )
            for name, metrics in result["validation"].items():
                for metric, value in metrics.items():
                    self.assertAlmostEqual(value, expected["validation"][name][metric], places=12)
            for phase in ("test", "test_baseline"):
                for metric, value in result[phase].items():
                    self.assertAlmostEqual(value, expected[phase][metric], places=12)
            np.testing.assert_allclose(
                predict(load_model(root / DEFAULT_MODEL_PATH), features),
                predict(load_model(root / "plain.pkl"), features),
            )

    def test_invalid_data_fails_before_training_or_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            invalid_csv = root / "invalid.csv"
            pd.DataFrame({"unexpected": [1]}).to_csv(invalid_csv, index=False)
            for csv_path, error in (
                (root / "missing.csv", FileNotFoundError),
                (invalid_csv, ValueError),
            ):
                with self.subTest(csv_path=csv_path):
                    state = training_flow(
                        csv_path=csv_path,
                        model_path=root / "model.pkl",
                        tracking_dir=root / "mlruns",
                        results_path=root / "results.json",
                        return_state=True,
                    )
                    self.assertTrue(state.is_failed())
                    with self.assertRaises(error):
                        state.result()
                    runs = self.task_runs(state)
                    self.assertEqual(len(runs), 1)
                    self.assertTrue(runs[0].state.is_failed())
                    for path in ("model.pkl", "results.json", "mlruns"):
                        self.assertFalse((root / path).exists())


if __name__ == "__main__":
    unittest.main()
