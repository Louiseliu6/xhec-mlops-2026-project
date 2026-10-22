"""End-to-end check of the training and inference interface."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.modelling.predicting import predict
from src.modelling.preprocessing import FEATURE_COLUMNS, read_training_data
from src.modelling.training import candidate_models, git_metadata, split_data, train_and_save
from src.modelling.utils import load_model, save_model


class ModellingTest(unittest.TestCase):
    def test_train_save_reload_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            count = 90
            rng = np.random.default_rng(42)
            data = pd.DataFrame(
                {
                    "Sex": np.resize(["M", "F", "I"], count),
                    "Length": rng.uniform(0.2, 0.8, count),
                    "Diameter": rng.uniform(0.1, 0.6, count),
                    "Height": rng.uniform(0.05, 0.25, count),
                    "Whole weight": rng.uniform(0.1, 2.0, count),
                    "Shucked weight": rng.uniform(0.05, 0.9, count),
                    "Viscera weight": rng.uniform(0.02, 0.5, count),
                    "Shell weight": rng.uniform(0.03, 0.6, count),
                    "Rings": rng.integers(5, 20, count),
                }
            )
            data.loc[0, "Height"] = 0.0
            csv_path = root / "abalone.csv"
            model_path = root / "abalone_model.pkl"
            data.to_csv(csv_path, index=False)

            with patch("src.modelling.training.save_model", wraps=save_model) as saved:
                result = train_and_save(
                    csv_path,
                    model_path,
                    tracking_dir=root / "mlruns",
                    results_path=root / "results.json",
                )
            model = load_model(model_path)
            loaded_data = read_training_data(csv_path)
            features = loaded_data.loc[[0, 1], FEATURE_COLUMNS]
            predictions = predict(model, features)
            fitted_model = saved.call_args.args[0]
            np.testing.assert_allclose(predictions, predict(fitted_model, features))

            x_train, x_val, x_trainval, x_test, *targets = split_data(loaded_data)
            partitions = [set(part.index) for part in (x_train, x_val, x_test)]
            for i, left in enumerate(partitions):
                for right in partitions[i + 1 :]:
                    self.assertTrue(left.isdisjoint(right))
            self.assertEqual(set.union(*partitions), set(loaded_data.index))
            self.assertEqual(set(x_trainval.index), partitions[0] | partitions[1])
            self.assertEqual([len(part) for part in (x_train, x_val, x_test)], [54, 18, 18])
            for features_part, target in zip((x_train, x_val, x_trainval, x_test), targets):
                self.assertTrue(features_part.index.equals(target.index))

            self.assertEqual(len(predictions), 2)
            self.assertTrue(np.isfinite(predictions).all())
            self.assertEqual(sum(result["rows"].values()), count)
            best_name = min(
                result["validation"], key=lambda name: result["validation"][name]["mae"]
            )
            self.assertEqual(result["selected_model"], best_name)
            self.assertEqual(
                result["selected_validation_mae"], result["validation"][best_name]["mae"]
            )
            self.assertTrue((root / "results.json").is_file())

            client = MlflowClient(tracking_uri=(root / "mlruns").as_uri())
            experiment = client.get_experiment_by_name("abalone_step2")
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 4)
            for run in runs:
                self.assertEqual(run.data.tags["git_commit"], git_metadata()["git_commit"])
                self.assertEqual(run.data.tags["worktree_dirty"], git_metadata()["worktree_dirty"])
                phase = run.data.tags["phase"]
                self.assertIn(phase, {"validation", "final_test"})
                name = run.data.params.get("candidate", run.data.params.get("selected_model"))
                for key, value in candidate_models()[name].get_params().items():
                    self.assertEqual(run.data.params[key], str(value))
            with self.assertRaisesRegex(ValueError, "Missing input columns"):
                predict(model, data[["Sex"]].head(1))

    def test_git_metadata_without_git(self) -> None:
        for error in (FileNotFoundError(), subprocess.CalledProcessError(128, "git")):
            with self.subTest(error=type(error).__name__):
                with patch("src.modelling.training.subprocess.check_output", side_effect=error):
                    self.assertEqual(
                        git_metadata(), {"git_commit": "unknown", "worktree_dirty": "unknown"}
                    )


if __name__ == "__main__":
    unittest.main()
