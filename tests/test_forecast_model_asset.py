from unittest.mock import patch

import numpy as np
import pandas as pd
from dagster import RunConfig, asset, materialize
from mlflow.tracking import MlflowClient

import churn_mlops.assets.forecast_model as forecast_model_module
from churn_mlops.assets.forecast_model import ForecastMlflowConfig, forecast_model


def _synthetic_forecast_table(n_customers=30, months_per_customer=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.uniform(20, 100)
        for m in range(months_per_customer):
            rows.append({
                "customerID": f"c{c}",
                "month": m + 3,
                "lag_1": base + rng.normal(0, 2),
                "lag_2": base + rng.normal(0, 2),
                "lag_3": base + rng.normal(0, 2),
                "rolling_3mo_mean": base,
                "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check"]),
                "TechSupport": rng.choice(["Yes", "No"]),
                "target_mrr": base + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


@asset(name="forecast_features")
def _stub_forecast_features():
    return _synthetic_forecast_table()


def test_first_run_registers_and_promotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_forecast_features, forecast_model],
        run_config=RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)}),
    )

    assert result.success
    output = result.output_for_node("forecast_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == output["version"]


def test_logged_model_has_signature(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_forecast_features, forecast_model],
        run_config=RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)}),
    )
    assert result.success
    output = result.output_for_node("forecast_model")

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    model_info = mlflow.models.get_model_info(f"models:/forecast_model/{output['version']}")
    assert model_info.signature is not None


def _fitted_stub_regressor():
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(n_estimators=2, max_depth=2)
    model.fit(pd.DataFrame({"a": [0, 1, 0, 1]}), [10.0, 20.0, 10.0, 20.0])
    return model


def _fixed_metrics(rmse):
    return {"rmse": rmse, "mae": rmse}


def test_worse_run_does_not_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(10.0), pd.DataFrame({"a": [0, 1]})),
    ):
        first = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert first.success
    first_version = first.output_for_node("forecast_model")["version"]

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(15.0), pd.DataFrame({"a": [0, 1]})),
    ):
        second = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("forecast_model")
    assert output["promoted"] is False

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == first_version
    assert production_version.version != output["version"]


def test_better_run_promotes_and_replaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(10.0), pd.DataFrame({"a": [0, 1]})),
    ):
        materialize([_stub_forecast_features, forecast_model], run_config=run_config)

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(5.0), pd.DataFrame({"a": [0, 1]})),
    ):
        second = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("forecast_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == output["version"]
