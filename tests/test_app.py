import pandas as pd
from fastapi.testclient import TestClient

from churn_mlops.serving.app import app


class _FakeChurnModel:
    def predict_proba(self, context, df: pd.DataFrame):
        return [0.73] * len(df)


def _valid_churn_payload():
    return {
        "tenure": 5, "Contract": "Month-to-month", "MonthlyCharges": 50.0,
        "TotalCharges": 250.0, "InternetService": "DSL",
        "PaymentMethod": "Electronic check", "TechSupport": "No",
        "current_mrr": 50.0, "trailing_3mo_avg_mrr": 48.0,
    }


def test_predict_churn_returns_valid_response_when_model_loaded():
    app.state.churn_model = _FakeChurnModel()
    client = TestClient(app)  # no `with`: lifespan (real model loading) does not run

    response = client.post("/predict/churn", json=_valid_churn_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["churn_probability"] == 0.73
    assert body["churn_prediction"] is True


def test_predict_churn_returns_503_when_model_not_loaded():
    app.state.churn_model = None
    client = TestClient(app)

    response = client.post("/predict/churn", json=_valid_churn_payload())

    assert response.status_code == 503


def test_predict_churn_returns_422_on_missing_field():
    app.state.churn_model = _FakeChurnModel()
    client = TestClient(app)

    payload = _valid_churn_payload()
    del payload["tenure"]
    response = client.post("/predict/churn", json=payload)

    assert response.status_code == 422
