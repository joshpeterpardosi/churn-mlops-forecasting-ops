from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

from churn_mlops.lib.mlflow_registry import get_production_metric, promote_if_better


def test_get_production_metric_returns_none_when_no_production_version():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("not found")
    assert get_production_metric(client, "m", "roc_auc") is None


def test_get_production_metric_returns_value_when_production_exists():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.9}
    client.get_run.return_value = run
    assert get_production_metric(client, "m", "roc_auc") == 0.9


def test_promote_if_better_promotes_when_no_production_exists():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("not found")
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "1", higher_is_better=True)
    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("m", "production", "1")


def test_promote_if_better_higher_is_better_promotes_when_new_value_wins():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.7}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "2", higher_is_better=True)
    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("m", "production", "2")


def test_promote_if_better_higher_is_better_skips_when_new_value_loses():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.9}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "2", higher_is_better=True)
    assert promoted is False
    client.set_registered_model_alias.assert_not_called()


def test_promote_if_better_lower_is_better_promotes_when_new_value_wins():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"rmse": 10.0}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "rmse", 8.0, "2", higher_is_better=False)
    assert promoted is True


def test_promote_if_better_lower_is_better_skips_when_new_value_loses():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"rmse": 8.0}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "rmse", 10.0, "2", higher_is_better=False)
    assert promoted is False
