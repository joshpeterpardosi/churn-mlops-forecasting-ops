from unittest.mock import patch

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dagster import RunConfig, asset, materialize
from mlflow.tracking import MlflowClient

import churn_mlops.assets.churn_model as churn_model_module
from churn_mlops.assets.churn_model import MlflowConfig, churn_model
from churn_mlops.lib.churn_model import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def _synthetic_feature_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"c{i}" for i in range(n)],
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
        ),
        "TechSupport": rng.choice(["Yes", "No"], size=n),
        "Churn": rng.integers(0, 2, size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
    })


@asset(name="feature_table")
def _stub_feature_table():
    return _synthetic_feature_table()


def test_first_run_registers_and_promotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_feature_table, churn_model],
        run_config=RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)}),
    )

    assert result.success
    output = result.output_for_node("churn_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == output["version"]


def _fitted_stub_model():
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=2, max_depth=2)
    model.fit(pd.DataFrame({"a": [0, 1, 0, 1]}), [0, 1, 0, 1])
    return model


def _fixed_metrics(roc_auc):
    return {"roc_auc": roc_auc, "accuracy": roc_auc, "precision": roc_auc, "recall": roc_auc, "f1": roc_auc}


def _stub_train_return(roc_auc):
    model = _fitted_stub_model()
    X_test = pd.DataFrame({"a": [0, 1]})
    return model, _fixed_metrics(roc_auc), X_test


def test_worse_run_does_not_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=_stub_train_return(0.70),
    ):
        first = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert first.success
    first_version = first.output_for_node("churn_model")["version"]

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=_stub_train_return(0.60),
    ):
        second = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("churn_model")
    assert output["promoted"] is False

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == first_version
    assert production_version.version != output["version"]


def test_better_run_promotes_and_replaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=_stub_train_return(0.70),
    ):
        materialize([_stub_feature_table, churn_model], run_config=run_config)

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=_stub_train_return(0.85),
    ):
        second = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("churn_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == output["version"]


def test_logged_model_has_signature(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_feature_table, churn_model],
        run_config=RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)}),
    )

    assert result.success
    output = result.output_for_node("churn_model")

    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/churn_model/{output['version']}"
    model_info = mlflow.models.get_model_info(model_uri)
    assert model_info.signature is not None


def test_loaded_model_predicts_on_plain_dtype_input(tmp_path, monkeypatch):
    """Reproduces the reviewer's finding: predicting on plain object/string
    dtype input (how a real caller / the model's own logged input_example
    naturally looks) must succeed, not raise LightGBM's
    'train and valid dataset categorical_feature do not match'.
    """
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_feature_table, churn_model],
        run_config=RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)}),
    )
    assert result.success
    output = result.output_for_node("churn_model")

    mlflow.set_tracking_uri(tracking_uri)
    loaded = mlflow.pyfunc.load_model(f"models:/churn_model/{output['version']}")

    plain_input = _synthetic_feature_table(n=10, seed=99)[FEATURE_COLUMNS]
    for col in CATEGORICAL_COLUMNS:
        assert plain_input[col].dtype == object, f"{col} must be plain-dtype for this test"

    predictions = loaded.predict(plain_input)

    assert len(predictions) == 10
    assert set(np.unique(predictions)).issubset({0, 1})
