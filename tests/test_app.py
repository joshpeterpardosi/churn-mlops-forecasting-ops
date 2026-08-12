import pandas as pd
from fastapi.testclient import TestClient

import churn_mlops.serving.app as app_module
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


class _FakeForecastModel:
    def predict(self, context, df: pd.DataFrame):
        return df["lag_1"] + 1.0


def _write_forecast_features_parquet(tmp_path):
    df = pd.DataFrame({
        "customerID": ["a", "a", "b"],
        "month": [3, 4, 3],
        "lag_1": [50.0, 52.0, 30.0],
        "lag_2": [48.0, 50.0, 29.0],
        "lag_3": [47.0, 48.0, 28.0],
        "rolling_3mo_mean": [48.33, 50.0, 29.0],
        "Contract": ["Month-to-month", "Month-to-month", "One year"],
        "InternetService": ["DSL", "DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Electronic check", "Mailed check"],
        "TechSupport": ["No", "No", "Yes"],
        "target_mrr": [52.0, 53.0, 31.0],
    })
    path = tmp_path / "forecast_features.parquet"
    df.to_parquet(path)
    return str(path)


def test_forecast_mrr_returns_valid_response_when_model_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FORECAST_FEATURES_PATH", _write_forecast_features_parquet(tmp_path))
    app_module.app.state.forecast_model = _FakeForecastModel()
    client = TestClient(app_module.app)

    response = client.get("/forecast/mrr", params={"horizon": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 2
    # customer "a" latest row (month 4, lag_1=52.0) predicts 53.0 then 54.0;
    # customer "b" only row (month 3, lag_1=30.0) predicts 31.0 then 32.0
    assert body["forecast_total_mrr"] == 54.0 + 32.0


def _write_forecast_features_with_churned_customer_parquet(tmp_path):
    df = pd.DataFrame({
        "customerID": ["active", "churned"],
        "month": [4, 4],
        "lag_1": [50.0, 30.0],
        "lag_2": [48.0, 29.0],
        "lag_3": [47.0, 28.0],
        "rolling_3mo_mean": [48.33, 29.0],
        "Contract": ["Month-to-month", "One year"],
        "InternetService": ["DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "TechSupport": ["No", "Yes"],
        # "churned" customer's terminal row has target_mrr == 0 (the churn-drop row),
        # so it must be excluded from the forecast seed.
        "target_mrr": [52.0, 0.0],
    })
    path = tmp_path / "forecast_features.parquet"
    df.to_parquet(path)
    return str(path)


def test_forecast_mrr_excludes_churned_customer_seed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module, "FORECAST_FEATURES_PATH", _write_forecast_features_with_churned_customer_parquet(tmp_path)
    )
    app_module.app.state.forecast_model = _FakeForecastModel()
    client = TestClient(app_module.app)

    response = client.get("/forecast/mrr", params={"horizon": 1})

    assert response.status_code == 200
    body = response.json()
    # Only "active" (lag_1=50.0) should seed the forecast; "churned" (target_mrr=0.0)
    # must be excluded, so the total is just active's single-step prediction.
    assert body["forecast_total_mrr"] == 51.0


def test_forecast_mrr_returns_503_when_model_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FORECAST_FEATURES_PATH", _write_forecast_features_parquet(tmp_path))
    app_module.app.state.forecast_model = None
    client = TestClient(app_module.app)

    response = client.get("/forecast/mrr", params={"horizon": 2})

    assert response.status_code == 503


def test_forecast_mrr_returns_503_when_parquet_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FORECAST_FEATURES_PATH", str(tmp_path / "does_not_exist.parquet"))
    app_module.app.state.forecast_model = _FakeForecastModel()
    client = TestClient(app_module.app)

    response = client.get("/forecast/mrr", params={"horizon": 2})

    assert response.status_code == 503


def test_forecast_mrr_rejects_horizon_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "FORECAST_FEATURES_PATH", _write_forecast_features_parquet(tmp_path))
    app_module.app.state.forecast_model = _FakeForecastModel()
    client = TestClient(app_module.app)

    assert client.get("/forecast/mrr", params={"horizon": 0}).status_code == 422
    assert client.get("/forecast/mrr", params={"horizon": 25}).status_code == 422


def test_lifespan_sets_mlflow_tracking_uri_before_loading_models(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module.mlflow, "set_tracking_uri", lambda uri: calls.append(uri))

    class _FakeLoaded:
        def unwrap_python_model(self):
            return object()

    monkeypatch.setattr(app_module.mlflow.pyfunc, "load_model", lambda uri: _FakeLoaded())

    with TestClient(app_module.app):
        pass

    assert calls == [app_module.MLFLOW_TRACKING_URI]
