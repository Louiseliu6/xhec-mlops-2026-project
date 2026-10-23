"""API checks against a small pipeline trained on synthetic data."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from src.modelling.predicting import predict
from src.modelling.preprocessing import NUMERIC_COLUMNS, make_pipeline
from src.modelling.utils import save_model
from src.web_service.main import app

PAYLOAD = {
    "Sex": "M",
    "Length": 0.455,
    "Diameter": 0.365,
    "Height": 0.095,
    "Whole weight": 0.514,
    "Shucked weight": 0.2245,
    "Viscera weight": 0.101,
    "Shell weight": 0.15,
}


class WebServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.uniform(0.1, 1, (60, 7)), columns=NUMERIC_COLUMNS)
        features.insert(0, "Sex", np.resize(["M", "F", "I"], len(features)))
        self.model = make_pipeline(LinearRegression()).fit(
            features, rng.integers(5, 20, len(features))
        )
        model_path = Path(temporary_dir.name) / "model.pkl"
        save_model(self.model, model_path)

        environment = mock.patch.dict(os.environ, {"MODEL_PATH": str(model_path)})
        environment.start()
        self.addCleanup(environment.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_health_check(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"health_check": "App up and running!"})

    def test_predict_matches_pipeline(self) -> None:
        response = self.client.post("/predict", json=PAYLOAD)

        self.assertEqual(response.status_code, 201)
        expected = predict(self.model, pd.DataFrame([PAYLOAD]))[0]
        self.assertAlmostEqual(response.json()["rings"], expected)

    def test_predict_rejects_invalid_input(self) -> None:
        invalid_payloads = [
            {**PAYLOAD, "Sex": "X"},
            {**PAYLOAD, "Length": -0.1},
            {key: value for key, value in PAYLOAD.items() if key != "Shell weight"},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)

    def test_predict_rejects_overflowing_numbers(self) -> None:
        # 1e309 is valid JSON syntax but overflows a Python float to infinity.
        for field in (key for key in PAYLOAD if key != "Sex"):
            with self.subTest(field=field):
                payload = {**PAYLOAD, field: "overflow-placeholder"}
                body = json.dumps(payload).replace('"overflow-placeholder"', "1e309")
                response = self.client.post(
                    "/predict", content=body, headers={"Content-Type": "application/json"}
                )
                self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
