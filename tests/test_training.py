"""Regression checks for the independently callable training stages."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.modelling.predicting import predict
from src.modelling.preprocessing import NUMERIC_COLUMNS
from src.modelling.training import (
    candidate_models,
    prepare_training_data,
    refit_and_evaluate,
    save_training_results,
    select_best_model,
    train_and_save,
    train_candidate,
)
from src.modelling.utils import load_model


class TrainingStagesTest(unittest.TestCase):
    def test_stages_match_training_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rng = np.random.default_rng(42)
            features = pd.DataFrame(rng.uniform(0.1, 1, (90, 7)), columns=NUMERIC_COLUMNS)
            features.insert(0, "Sex", np.resize(["M", "F", "I"], len(features)))
            features.loc[0, "Height"] = 0
            features["Rings"] = rng.integers(5, 20, len(features))
            csv_path = root / "data.csv"
            features.to_csv(csv_path, index=False)

            data = prepare_training_data(csv_path)
            candidates = candidate_models()
            validation = {
                name: train_candidate(name, estimator, data, root / "mlruns", "stages")
                for name, estimator in candidates.items()
            }
            best_name = select_best_model(validation)
            model, result = refit_and_evaluate(
                best_name, candidates[best_name], data, validation, root / "mlruns", "stages"
            )
            model_path = root / "stages" / "model.pkl"
            results_path = root / "stages" / "results.json"
            self.assertFalse(model_path.exists())
            self.assertFalse(results_path.exists())
            self.assertIs(save_training_results(model, result, model_path, results_path), result)
            self.assertEqual(json.loads(results_path.read_text()), result)

            expected = train_and_save(
                csv_path, root / "entry.pkl", root / "mlruns", root / "entry.json"
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
                predict(load_model(model_path), features),
                predict(load_model(root / "entry.pkl"), features),
            )
            # Each stage clones its estimator rather than mutating the candidate configuration.
            for estimator in candidates.values():
                self.assertFalse(hasattr(estimator, "n_features_in_"))

            # Candidate validation must not read held-out test features or targets.
            data.splits = data.splits._replace(x_test=pd.DataFrame(), y_test=pd.Series(dtype=float))
            metrics = train_candidate(
                "linear_regression", LinearRegression(), data, root / "mlruns", "stages"
            )
            self.assertEqual(metrics, validation["linear_regression"])

    def test_selects_by_validation_mae_not_other_metrics(self) -> None:
        validation = {
            "first": {"mae": 2.0, "rmse": 2.0, "r2": 0.9},
            "second": {"mae": 1.0, "rmse": 3.0, "r2": 0.1},
        }
        self.assertEqual(select_best_model(validation), "second")


if __name__ == "__main__":
    unittest.main()
